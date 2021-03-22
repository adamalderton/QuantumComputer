/****************************************************************************************
    Program -- qc_shor.c

    Author
        Adam Alderton (aa816@exeter.ac.uk)

    Version
        1.0.3

    Revision Date
        21/03/2021

    Purpose
        Simulate the execution of a quantum array compute as part of Shor's algorithm,
        which uses quantum computers to factorise potentially very large numbers.

    Requirements and Compilation
        Non-Standard Requirements:
            GNU Scientific Library
        
        Compilation:
            gcc qc_shor.c -Wall -IC:[include directory] -LC:[lib directory] -lgsl -lgslcblas -lm -o qc_shor.exe

    Usage
        Mandatory command line arguments:
            -C [positive integer]
                The number to be factorised by the program.

            -L [positive integer]
                The size of the L sub-register of the qubit register.
            
            -M [positive integer]
                The size of the M sub-register of the qubit register.
        
        Options:
            -i [positive integer]
                Force Shor's algorithm to use this integer only as the trial integer.
            
            -v
                Ensure the program gives a medium-level of update messages as the
                program executes. This is useful when executing the program with
                register sizes such that the program takes 10's of seconds or a few
                minutes to execute.
            
            -V
                Ensure the program gives a high-level of update messages as the program
                executes. This includes the functionality of the -v option. This option
                also contains messages of the groups of quantum gates currently being
                executed. This option is useful when executing the program with a register
                such that the program may take several minutes or more to execute.

    Limitations
        The parameters describing the number of continued fractions to analyse are not
        currently adjustable from run-to-run, as they are defined within the preprocessor
        below.

        Not all possible checks are carried out on the given command line arguments, only
        some neccessary ones. Therefore, some care should be taken when passing them.
        However, with competance and some knowledge of Shor's algorithm, no issues should
        be met.
    
        The absolute maximum of qubits that can be considered by this program is 32, as 
        governed by the amount of bits in the unsigned long type. However, this is an
        incredibly large number of qubits such that the simulation of this many qubits is
        simply unfeasible. If for some reason this needed to be extended, the unsigned
        long long int datatype would be able to consider 64 qubits when building the
        quantum gate matrices.

    Notes
        The program will display warnings for when the register sizes are not above certain
        limits, which provide confidence in finding periods. However, in many cases, the
        period can be found with smaller registers. For example, 15 can be factored with
        L = 3 and M = 4, and 35 can be consistently factored with L = 5, M = 5. Therefore,
        the warnings are simply there to remind the user that the results from the program
        may not be 100% reliable.

        Much of the notation and methodology within this program is derived from and
        consisten with the D. Candela reference below. An introduction to the notation 
        and to Shor's algorithm can be found there should there be any confusion.

    References
        D. Candela, "Undergraduate Computational Physics Projects on Quantum Computing", 
        American Journal of Physics 83, 688 (2015).

        M. Galassi, J. Davies, J. Theiler, B. Gough, G. Jungman, P. Alken, M. Booth,
        F. Rossi and R. Ulerich, "GNU Scientific Library 2.6 Reference Manual", 2019.

 ****************************************************************************************/

/***********************************PREPROCESSOR*****************************************/

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/time.h>   /* To time simulation. */
#include <time.h>       /* To time simulation. */

#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_spblas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_math.h>

#define VERSION "1.0.3"
#define REVISION_DATE "21/03/2021"

/* Used to correctly scale Hadamard matrices which is otherwise a matrix of integers. */
#define HADAMARD_SCALE 1.0/M_SQRT2

/* Used to inform operate_matrix() that no alternative complex element is needed. */
#define NULL_ALT_ELEMENT gsl_complex_rect(0.0, 0.0)

/* Used to inform operate_matrix() that a complex element is needed in place of this element. */
#define COMPLEX_ELEMENT -127

/* Defines the extent to which the continued fraction expansion is analysed. */
#define NUM_CONTINUED_FRACTIONS 15
#define TRIALS_PER_DENOMINATOR 10

/* 
   Many functions below return an errorcode. This macro is called after these functions return
   and checks for an error. If so, this ends the function in which the error
   occured returns the error code. If the error raising function is called within a function,
   that calling function is also returned with the same error code. This repeats until the scope of the main
   function is reached. Therefore, this macro passes the error code up the stack until it is eventually returned by main.
 */
#define ERROR_CHECK(error) \
    if (error != NO_ERROR) { \
        return error; \
    }

/* Checks the success of memory allocation functions such as malloc and similar functions within GSL. */
#define ALLOC_CHECK(alloc_pointer) \
    if (alloc_pointer == NULL) { \
        fprintf(stderr, "Error: Insufficient memory.\n"); \
    }

/* 
   Gets the binary 1 or 0 stored in the n'th bit of integer (from the right) 
   by bit masking followed by a bitwise and. That is, 00000001 is 
   leftshifted n places to address the n'th bit (from the right) in integer.
   The value resulting from the bitwise and will result in either 0,
   or a finite power of two, corresponding to the nth bit. That is, 2^n.
   Therefore, the result is rightshfted n places to yield either 1 or 0.
 */
#define GET_BIT(integer, n) \
    ( (integer & (1 << n)) >> n )

/*
    Finds the result of an integer raised to an integer power, returned as an unsigned integer.
    Therefore, care must be taken with the use of powers of negative numbers, which does not
    occur in this program.
*/
#define INT_POW(base, power) \
    ( (unsigned int) (pow(base, power) + 0.5) )

/***********************************TYPEDEFS AND GLOBALS*********************************/

/* Enum to store the various error codes than can be returned within this program. */
typedef enum {
    NO_ERROR = 0,
    INSUFFICIENT_MEMORY,
    BAD_ARGUMENTS,
    PERIOD_NOT_FOUND,
    UNKNOWN_ERROR,
} ErrorCode;

/*
    Struct to store matrices relevant to the program.

    Stored are two matrices, result_matrix and comp_matrix. Within this program,
    matrices are constructed within comp_matrix which is a sparse matrix of the
    coordinate type, and comp_matrix is then compressed into result_matrix by the
    function compress_comp_matrix(). The result matrix is then used in matrix-vector
    multiplications as its compressed row format is more efficient for matrix-vector
    multiplications, especially for large matrices.
*/
typedef struct {
    gsl_spmatrix_char *result_matrix;
    gsl_spmatrix_char *comp_matrix;
} GateMatrices;

/*
    A struct containing details of the qubit register, and its sub-registers L and M.
    Please see the 'Candela' reference cited for details on these separate registers.
    The number of qubits is also stored, and the corresponding number of states which
    is simply 2^(num_qubits).

    Also stored are two complex vectors state_a and state_b. These hold the vectorised
    form of the wavefunction. Two must be stored as many of the calculations carried out
    cannot be completed in-place, so the result of a calculation on state_a is stored in
    state_b.

    The two stored double pointers current_state and new_state track which of state_a
    and state_b hold the 'current state' of the system, and the 'new state' of the system
    which will be realised after a matrix-vector multiplication. After each operation, these
    pointers are swapped such that they point to the other of state_a or state_b. This small
    detail prevents the need to expensively copy an entire vector into another one, which can
    be expensive especially if the vectors are very large, which they can be in this program.
    Due to this pointer 'swapping' behaviour on each operation, the vectors state_a and 
    state_b should not be dealt with directly in future development, but the pointers
    current_state and new_state should be dereferenced to give the necessary vectors.
*/
typedef struct {
    int L_size;     /* Signed as to be able to handle invalid command line argument. */
    int M_size;     /* Signed as to be able to handle invalid command line argument. */
    unsigned int num_qubits;
    unsigned long int num_states;
    gsl_vector_complex **current_state;
    gsl_vector_complex **new_state;
    gsl_vector_complex *state_a;
    gsl_vector_complex *state_b;
} Register;

/* 
    A basis matrix used in the construction of larger Hadamard matrices.

    Does NOT contain the necessary scale of 1/sqrt(2),
    such that it can be stored as an integer matrix.
    This scalar factor is implemented later in appropriate functions.
 */
const char HADAMARD_BASE_MATRIX[2][2] = {
    {1, 1},
    {1, -1}
};

/*
    A basis matrix used in the contruction of phase-shift gate matrices.

    The matrices in this program do not store complex numbers in the interest of memory,
    so COMPLEX_ELEMENT is a placeholder for what will be a complex value. COMPLEX_ELEMENT
    is checked for in the operate_matrix() function in which the complex element which
    COMPLEX_ELEMENT is representing can be passed. In the case of the phase shift matrix,
    this complex element is z = r * e^(i\theta).
*/
const char C_PHASE_SHIFT_BASE_MATRIX[4][4] = {
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, COMPLEX_ELEMENT}
};

/* Used for the controlled display of progress messages throughout the program. */
bool verbose = false;
bool very_verbose = false;

/********** UTILITY FUNCTIONS **********/

/****************************************************************************************
    compress_comp_matrix -- Compress the coordinate based comp_matrix into the 
                            column compressed result_matrix. comp_matrix is also
                            zeroed out in preparation for the next matrix operation.
    
    Parameters:
        Assets *assets
            Contains comp_matrix and result_matrix pointers.

 ****************************************************************************************/
static void compress_comp_matrix(GateMatrices matrices)
{
    /* Compress comp_matrix into result_matrix in compressed column format. */
    gsl_spmatrix_char_csr(matrices.result_matrix, matrices.comp_matrix);

    /* Reset comp_matrix straight away as to reduce memory bloat. */
    gsl_spmatrix_char_set_zero(matrices.comp_matrix);
}


/****************************************************************************************
    swap_states -- Swaps the pointers current_state and new_state between state_a
                   and state_b.
    
    Parameters:
        Assets *assets
            Contains current_state, new_state, state_a and state_b.

 ****************************************************************************************/
static void swap_states(Register *reg)
{
    gsl_vector_complex **temp;

    temp = reg->new_state;
    reg->new_state = reg->current_state;
    reg->current_state = temp;
}


/****************************************************************************************
    measure_state -- Collapses the wavefunction current_state into a particular state,
                     by measurement. The state it collapses to is randomly determined.
    
    Parameters:
        Register reg
            Contains information regarding the qubit register.

        Assets *assets
            Contains the current state vector in *assets->current_state.
        
        gsl_rng *rng
            Random number generator implemented by GSL.
    
    Returns:
        unsigned long int state_num
            An integer corresponding to the index of the state measured in the
            wavefunction vector. The GET_BIT() macro can be used on this integer to
            determine whether a qubit was measured to be 1 or 0. That is, it is the
            decimal representation of the binary value stored in the measured qubit
            register.

 ****************************************************************************************/
static unsigned long int measure_state(Register reg, gsl_rng *rng)
{
    double r;                           /* Random number between 0.0 and 1.0, used to measure the state. */
    double cumulative_prob;             /* Quantity to track the cumulative probability considered so far. */
    unsigned long int state_num;        /* The index in the state vector of the state measured. */
    gsl_vector_complex *current_state;  /* To prevent repeated pointer dereference. */

    current_state = *reg.current_state;
    cumulative_prob = 0.0;
    r = gsl_rng_uniform(rng);

    for (state_num = 0; state_num < (reg.num_states - 1); state_num++) {

        /* Add the square of the absolute value of the state coefficient, as per the basics of quantum mechanics. */
        cumulative_prob += gsl_complex_abs2(gsl_vector_complex_get(current_state, state_num));

        /* The collapsed state is found, the number of which is stored in state_num. */
        if (cumulative_prob >= r) {
            break;
        }
    }

    /* 
        Now, set current_state to be the collapsed state.
        That is, set the state_num'th state to have a probability of 1.

        This step could be omitted if one wanted to repeatedly measure this state,
        but this is not in the spirit of true quantum mechanics hence quantum computers.
     */
    gsl_vector_complex_set_zero(current_state);
    gsl_vector_complex_set(current_state, state_num, gsl_complex_rect(1.0, 0.0));

    return state_num;
}


/****************************************************************************************
    reset_register -- Resets the qubit register to the state required to begin applying
                      the quantum circuit. That is, |000 ... 001>.
    
    Parameters:
        gsl_vector_complex *current_state
            Dereferenced pointer to the current state. That is, pass it as
            *assets.current_state.

 ****************************************************************************************/
static void reset_register(Register reg)
{
    gsl_vector_complex_set_zero(*reg.current_state);

    /* Sets register to |000 ... 001>. */
    gsl_vector_complex_set(*reg.current_state, 1, gsl_complex_polar(1.0, 0.0));
}


/****************************************************************************************
    issue_warnings -- With user parameters inputted such as the register sizes,
                      give warnings concerning the confidence of finding factors.
    
    Parameters:
        unsigned int C
            The number passed to the program to be factorised.
        
        Register reg
            An instance of the Register struct containing information about the 
            qubit register.

 ****************************************************************************************/
static void issue_warnings(unsigned int C, Register reg)
{
    if (C % 2 == 0) {
        printf(" --- *WARNING* Number to factorise C = %d is even, results may be unreliable.\n", C);
    }

    /* To find valid factors, 2^M must be greater than or equal to C. */
    if (INT_POW(2, reg.M_size) < C) {
        printf(" --- *WARNING* The M register is not large enough for reliable results. Ensure 2^M >= C. Minimum: M = %d.\n", ((int) (log2(C) + 0.5)) + 1);
    }

    /* To be confident of finding the period, 2^L should be greater than or equal to C. */
    if (INT_POW(2, reg.L_size) < C*C) {
        printf(" --- *WARNING* The L register is not large enough for full confidence in finding the period. Ensure 2^L >= C^2 for confidence. Suggested: L = %d.\n", (int) (log2(C*C) + 0.5));
    }
}


/********** QUANTUM GATE FUNCTIONS **********/


/****************************************************************************************
    operate_matrix -- With a matrix built and compressed in assets.result_matrix,
                      operate it on the vector wavefunction *assets.current_state.
                      The result is stored in *assets.current_state.
    
    Parameters:
        Register reg
            An instance of the Register struct containing information about the 
            qubit register.
        
        Assets *assets
            Contains the necessary matrices and vectors.
        
        double scale
            A scalar by which to scale the matrix-vector product. For example, when
            using the Hadamard matrix, a scale of 1/sqrt(2) should be used as it is
            stored as integers.
        
        gsl_complex alt_element
            An alternative complex matrix element to be used in the matrix-vector
            multiplication. For example, when applying the phase_shift matrix,
            COMPLEX_ELEMENT is substituted for z = r * e^(i\theta).
    
    Notes:
        Credit for the workings of the sparse matrix-vector multiplication should be
        given to the GSL source code 
        (https://github.com/ampl/gsl/blob/master/spblas/spdgemv.c). Their implementation
        has been adapted to be able to multiply complex elements.

 ****************************************************************************************/
static void operate_matrix(Register reg, GateMatrices matrices, double scale, gsl_complex alt_element)
{
    gsl_vector_complex *n_state;    /* To prevent repeated pointer dereference. */
    gsl_vector_complex *c_state;    /* To prevent repeated pointer dereference. */
    gsl_spmatrix_char *mat;         /* To prevent repeated pointer dereference. */
    int stride;                     /* Stride of state vectors. */
    double c_real;                  /* Real part of current state element. */
    double c_imag;                  /* Imaginary part of current state element. */
    double m_real;                  /* Real part of matrix element. */
    double m_imag;                  /* Imaginary part of matrix element. */

    n_state = *reg.new_state;       /* Dereference new_state double pointer. */
    c_state = *reg.current_state;   /* Dereference current_state double pointer. */
    stride = c_state->stride;       /* Assuming strides of current state and new state are equal. */
    mat = matrices.result_matrix;

    /* Reset new_state to zero. */
    gsl_vector_complex_set_zero(n_state);

    if (scale == 0.0) {
        return;
    }

    /* Apply scale to current state, as to reduce double multiplications in for loops below. */
    if (scale != 1.0) {
        gsl_vector_complex_scale(c_state, gsl_complex_rect(scale, 0.0));
    }

    for (unsigned long int j = 0; j < reg.num_states; j++) {

        /* pj is a pointer to the element at the start of the j'th column. */
        for (unsigned long int pj = mat->p[j]; pj < mat->p[j + 1]; pj++) {

            /* Retrieve matrix element. */
            if (m_real == COMPLEX_ELEMENT) {
                m_real = GSL_REAL(alt_element);
                m_imag = GSL_IMAG(alt_element);
            } else {
                m_real = mat->data[pj];
                m_imag = 0.0;
            }

            /* Retrieve corresponding element in current_state vector. */
            c_real = c_state->data[2 * stride * j];
            c_imag = c_state->data[2 * stride * j + 1];

            /* Real part. */
            n_state->data[2 * stride * mat->i[pj]] += (m_real * c_real) - (m_imag * c_imag);

            /* Imaginary part. */
            n_state->data[2 * stride * mat->i[pj] + 1] += (m_real * c_imag) + (m_imag * c_real);
        }
    }

    /* The result is stored in new_state, so complete by storing the result in current_state instead. */
    swap_states(&reg);
}


/****************************************************************************************
    hadamard_gate -- Apply the hadamard gate to a qubit in the qubit register.

    Parameters:
        unsigned int qubit_num
            The qubit number to apply the Hadamard gate to. Note: the qubit counting
            starts at 0.
        
        Register reg
            An instance of the Register struct containing information about the qubit
            register.
        
        Assets *assets
            Contains the necessary matrices and vectors.

    Notes:
        The construction of the matrix is as per the instruction in the cited Candela
        reference.

 ****************************************************************************************/
static void hadamard_gate(unsigned int qubit_num, Register reg, GateMatrices matrices)
{
    /* 
        Holds the result of the bitwise not operation applied to the 
        result of the bitwise xor operation between the matrix indices i and j
    */
    unsigned long int not_xor_ij;
    char element;                   /* Element of Hadamard matrix. */
    bool dirac_deltas_non_zero;     /* Determines whether any of the dirac deltas are zero or not. */

    /* Iterate over all possible elements of the matrix. */
    for (unsigned long int i = 0; i < reg.num_states; i++) {
        for (unsigned long int j = 0; j < reg.num_states; j++) {
            dirac_deltas_non_zero = true;

            not_xor_ij = ~(i ^ j);

            /* Check that all of the dirac-deltas are 1 before proceeding. */
            for (unsigned int b = 0; b < reg.num_qubits; b++) {

                if (b != qubit_num) {
                    if (GET_BIT(not_xor_ij, b) == 0) {
                        dirac_deltas_non_zero = false;
                        break;
                    }
                }
            }

            if (dirac_deltas_non_zero) {

                /* Retrieve element from base matrix. */
                element = HADAMARD_BASE_MATRIX[GET_BIT(i, qubit_num)][GET_BIT(j, qubit_num)];

                /* Insert element in comp_matrix. */
                gsl_spmatrix_char_set(matrices.comp_matrix, i, j, element);
            }
        }
    }

    /* Compress comp_matrix into result_matrix for efficient multiplication. */
    compress_comp_matrix(matrices);

    /* With the Hadamard matrix built and compressed, operate the matrix with the scale 1/sqrt(2). */
    operate_matrix(reg, matrices, HADAMARD_SCALE, NULL_ALT_ELEMENT);
}


/****************************************************************************************
    c_phase_shift_gate -- Apply the conditional phase shift gate on a qubit in the qubit
                          register.

    Parameters:
        unsigned int c_qubit_num
            The conditional qubit number. Note: the qubit counting starts at 0.

        unsigned int qubit_num
            The qubit number to apply the phase shift gate to. Note: the qubit counting
            starts at 0.
        
        double theta
            The phase shift to apply. The \theta in z = r* e^(i\theta).
        
        Register reg
            An instance of the Register struct containing information about the qubit
            register.
        
        Assets *assets
            Contains the necessary matrices and vectors.
    
    Notes:
        The construction of the matrix is as per the instruction in the cited Candela
        reference.

 ****************************************************************************************/
static void c_phase_shift_gate(unsigned int c_qubit_num, unsigned int qubit_num, double theta, Register reg, GateMatrices matrices)
{
    /* 
        Holds the result of the bitwise not operation applied to the 
        result of the bitwise xor operation between the matrix indices i and j
    */
    unsigned long int not_xor_ij;
    char element;                   /* Element of phase shift matrix. */
    bool dirac_deltas_non_zero;     /* Determines whether any of the dirac deltas are zero or not. */

    /* Iterate over all possible elements of the matrix. */
    for (unsigned long int i = 0; i < reg.num_states; i++) {
        for (unsigned long int j = 0; j < reg.num_states; j++) {
            dirac_deltas_non_zero = true;

            not_xor_ij = ~(i ^ j);

            /* Check that all of the dirac-deltas are 1 before proceeding. */
            for (unsigned int b = 0; b < reg.num_qubits; b++) {

                if ( (b != qubit_num) && (b != c_qubit_num) ) {
                    if (GET_BIT(not_xor_ij, b) == 0) {
                        dirac_deltas_non_zero = false;
                        break;
                    }
                }
            }

            if (dirac_deltas_non_zero) {

                /* Retrieve element from base matrix. */
                element = C_PHASE_SHIFT_BASE_MATRIX
                    [(2*GET_BIT(i, c_qubit_num)) + GET_BIT(i, qubit_num)]
                    [(2*GET_BIT(j, c_qubit_num)) + GET_BIT(j, qubit_num)];

                /* Insert element in comp_matrix. */
                gsl_spmatrix_char_set(matrices.comp_matrix, i, j, element);
            }
        }
    }

    /* Compress comp_matrix into result_matrix for efficient multiplication. */
    compress_comp_matrix(matrices);

    /*
        With the phase shift matrix built and compressed, operate it with
        the alt_element being z = e^(i\theta).
    */
    operate_matrix(reg, matrices, 1.0, gsl_complex_polar(1.0, theta));
}


/****************************************************************************************
    c_amodc_gate -- Apply the f(x) = a^x (mod C) gate, where x corresponds to 
                    2^(c_qubit_num), to the M register.

    Parameters:
        unsigned int C
            The number passed to the program to factorise.
        
        unsigned long long int atox
            The result of the trial integer a raised to the power of 2^(c_qubit_num),
            which can get large quickly. To minimise the possibility of an overflow
            error, the quantity is stored as an unsigned long long int.
        
        unsigned int c_qubit_num
            The conditional qubit num with which to determine the application of f(x).
        
        Register reg
            An instance of the Register struct containing information about the qubit
            register.
        
        Assets *assets
            Contains the necessary matrices and vectors.

    Notes:
        The construction of the matrix is as per the instruction in the cited Candela
        reference.

 ****************************************************************************************/
static void c_amodc_gate(unsigned int C, unsigned long long int atox, unsigned int c_qubit_num, Register reg, GateMatrices matrices)
{
    unsigned int A; /* Holds a^x (mod C). */
    unsigned int j; /* The column of the matrix in row k in which the 1 resides. */
    unsigned int f; /* Used in the calculation of the permutation matrix. */

    /* Compiler automatically casts A, atox and C to yield the correct A. */
    A = atox % C;

    /* Using notation from instruction document, loop over rows (k) of matrix. */
    for (unsigned long int k = 0; k < reg.num_states; k++) {

        /* If l_0 (c_qubit_num) is 0, j = k. */
        if (GET_BIT(k, c_qubit_num) == 0) {
            gsl_spmatrix_char_set(matrices.comp_matrix, k, k, 1);
            continue;
        
        /* If l_0 = 1 ... */
        } else {

            /* f must be calculated, which is the decimal value stored in the M register. */
            f = 0;
            for (unsigned int b = 0; b < reg.M_size; b++) {
                f += GET_BIT(k, b) << b;
            }

            /* 
             * The calculation above could have been omitted, and just the bits
             * above and including the nearest power of 2 to C, rounding up,
             * could have been tested for any values of 1. However, for simplicity
             * and readability, and considering the speed of bitwise calculations,
             * this approach of calculating f every time was taken.
             */
            if (f >= C) {

                gsl_spmatrix_char_set(matrices.comp_matrix, k, k, 1);
                continue;

            } else {

                /* Calculate f', which can be stored in f for simplicity. */
                f = (A * f) % C;

                /* Next, build up the integer value j using f'. */
                j = 0;

                /* M register (concerning f'). */
                for (unsigned int b = 0; b < reg.M_size; b++) {
                    j += GET_BIT(f, b) << b;
                }

                /* L register (concerning k). */
                for (unsigned int b = reg.M_size; b < reg.num_qubits; b++) {
                    j += GET_BIT(k, b) << b;
                }

                gsl_spmatrix_char_set(matrices.comp_matrix, j, k, 1);
            }
        }
    }

    compress_comp_matrix(matrices);

    operate_matrix(reg, matrices, 1.0, NULL_ALT_ELEMENT);
}


/****************************************************************************************
    inverse_QFT -- Perform an inverse quantum Fourier transform on the L register.
    
    Parameters:
        Register reg
            An instance of the Register struct containing information about the 
            qubit register.

        Assets *assets
            Contains the necessary matrices and vectors.

    Notes:
        The inverse quantum Fourier transform is peformed as per the instructions in
        the cited Candela reference.

 ****************************************************************************************/
static void inverse_QFT(Register reg, GateMatrices matrices)
{
    double theta;   /* The phase to apply within the phase shift gates. */

    for (int l = reg.L_size - 1; l >= 0; l--) {
        hadamard_gate(l, reg, matrices);

        for (int k = l - 1; k >= 0; k--) {
            theta = M_PI / (double) INT_POW(2, reg.L_size - k - 1);
            c_phase_shift_gate(l, k, theta, reg, matrices);
        }
    }
}


/****************************************************************************************
    quantum_computation -- Perform the series of quantum gates in the quantum circuit
                           as within the cited Candela reference.
    
    Parameters:
        unsigned int C
            The number passed to the program to be factorised.

        unsigned int a
            The current trial integer within Shor's algorithm, as per the Candela
            reference.

        Register reg
            An instance of the Register struct containing information about the 
            qubit register.

        Assets *assets
            Contains the necessary matrices and vectors.

 ****************************************************************************************/
static void quantum_computation(unsigned int C, unsigned int a, Register reg, GateMatrices matrices)
{
    unsigned int x = 1;

    /* Apply Hadamard gate to qubits in the L register. */
    if (very_verbose) {
        printf("         - Applying Hadamard matrices.\n");
    }
    for (unsigned int l = (reg.num_qubits - reg.L_size); l < reg.num_qubits; l++) {
        hadamard_gate(l, reg, matrices);
    }

    if (very_verbose) {
        printf("         - Applying a^x mod (C) gates.\n");
    }
    /* For each bit value in the L register, apply the conditional a^x (mod C) gate. */
    for (unsigned int l = (reg.num_qubits - reg.L_size); l < reg.num_qubits; l++) {
        c_amodc_gate(C, INT_POW(a, x), l, reg, matrices);
        x *= 2;
    }

    if (very_verbose) {
        printf("         - Performing inverse quantum Fourier transform.\n");
    }
    inverse_QFT(reg, matrices);
}


/********** SHOR'S ALGORITHM FUNCTIONS **********/


/****************************************************************************************
    greatest_common_divisor -- Find the greatest common divisor between two integers
                               using an iteration of Euclid's algorithm.
    
    Parameters:
        unsigned int a
            
        unsgined int b
    
    Returns:
        The greatest common divisor of a and b.

 ****************************************************************************************/
static unsigned int greatest_common_divisor(unsigned int a, unsigned int b)
{
    unsigned int temp;

    /* Trivial cases. */
    if (a == 0) {
        return b;
    }
    if (b == 0) {
        return a;
    }
    if (a == b) {
        return a;
    }

    /* Simple iterative version of Euclid's algorithm. */
    while ((a % b) > 0) {
        temp = a % b;
        a = b;
        b = temp;
    }

    return b;
}


/****************************************************************************************
    get_continued_fractions_denominators -- Get the denominators of the fractions
                                            resulting from the continued fraction
                                            expansion of omega = x_tilde / 2^L.
    
    Parameters:
        double omega
            The measured x_tilde register value divided by 2^L. It should be some
            harmonic of the base frequency hence omega, resulting from the 
            quantum Fourier transform.
            
        unsgined int num_fractions
            The number of fraction interations to produce using the continued fraction
            expansion.
        
        unsigned int *denominators
            A previously allocated array of unsigned int datatype with length
            num_fractions. On completion of the function, it will store the denominators
            of the fractions found.
    
    Notes:
        This function was built following the guidance of the cited Candela reference.

 ****************************************************************************************/
static void get_continued_fractions_denominators(double omega, unsigned int num_fractions, unsigned int *denominators)
{
    double omega_inv;           /* 1/omega for the current iteration. */
    unsigned int numerator;     
    unsigned int denominator;
    unsigned int temp;          /* Used in the swapping of the numerator and the denominator. */
    /*
        Coefficients used in the continued fraction expansion. In the example within the Candela
        reference paper, they are 1, 1, 1, 3, 1, 19 ...
    */
    unsigned int *coeffs;       

    coeffs = (unsigned int *) malloc(num_fractions * sizeof(unsigned int));
    ALLOC_CHECK(coeffs);

    for (unsigned int i = 0; i < num_fractions; i++) {
        omega_inv = 1.0 / omega;

        /* Omega for next loop iteration, which is the fractional part of omega_inv. */
        omega = omega_inv - (double) ( (unsigned int) omega_inv );

        /* Coefficient calculation uses next omega value. */
        coeffs[i] = (unsigned int) (omega_inv - omega);

        /*
            With the coefficient array built, use it (in reverse order) to build the
            numerator and denominator of the continued fraction approximation.
        */
        denominator = 1;
        numerator = 0;
        for (int coeff_num = i - 1; coeff_num >= 0; coeff_num--) {
            temp = denominator;
            denominator = numerator + (denominator * coeffs[coeff_num]);
            numerator = temp;
        }

        denominators[i] = denominator;
    }

    free(coeffs);
}


/****************************************************************************************
    read_omega -- Reads the value x_tilde from the x_tilde register and divides it by
                  2^L, to yield some harmonic of the fundamental frequency omega.
    
    Parameters:
        unsigned long int state_num
            An integer who's binary representation represents the measured state of
            the qubit register. Individual qubits can be addressed with the GET_BIT
            macro.

        Register reg
            Contains information about the qubit register including the size of the
            sub registers.
    
    Returns:
        double omega
            x_tilde / 2^L

 ****************************************************************************************/
static double read_omega(unsigned long int state_num, Register reg)
{
    unsigned int x_tilde; /* Quantity to build the result of the x_tilde register in. */
    unsigned int power;

    power = 0;

    /* Read x_tilde register in reverse order. */
    for (unsigned int i = reg.L_size + reg.M_size - 1; i >= reg.M_size; i--) {
        x_tilde += GET_BIT(state_num, i) << power;
        power++;
    }

    return (double) x_tilde / (double) INT_POW(2, reg.L_size);
}


/****************************************************************************************
    find_period -- With a trial integer a, attempt to find the period of the function
                   f(x) = a^x (mod C) using (simulated) quantum mechanical computations.
    
    Parameters:
        unsigned int *period
            The quantity in which the period is stored, if found.
        
        unsigned int a
            The trial integer passed to the period finding algorithm, as per the
            notation in the Candela reference.
        
        Register reg
            Contains information regarding the qubit register, such as the size of the
            sub registers.
        
        Assets *assets
            Contains the matrices and state vectors needed to perform the quantum
            computations.
        
        gsl_rng *rng
            The random number generator as implemented by GSL.
    
    Returns:
        ErrorCode error
            Describes if any errors have occured, including not finding a period.

 ****************************************************************************************/
static ErrorCode find_period(unsigned int *period, unsigned int C, unsigned int a, Register reg, GateMatrices matrices, gsl_rng *rng)
{
    unsigned int *denominators;             /* The denominators of the fractions in the continued expansion of omega. */
    bool period_found;                      /* True if the period has been found successfully, false if not. */
    unsigned long int measured_state_num;   /* The decimal representation of the binary number representing the measured qubit register. */
    double omega;                           /* The measured x_tilde / 2^L. */

    if (very_verbose) {
        printf("      - Performing quantum computation...\n");
    }
    reset_register(reg);
    quantum_computation(C, a, reg, matrices);

    if (very_verbose) {
        printf("      - Measuring state...\n");
    }

    measured_state_num = measure_state(reg, rng);
    omega = read_omega(measured_state_num, reg);

    if (very_verbose) {
        printf("      - Using continued fractions to guess period...\n");
    }

    denominators = (unsigned int *) malloc(NUM_CONTINUED_FRACTIONS * sizeof(unsigned int));
    ALLOC_CHECK(denominators);

    get_continued_fractions_denominators(omega, NUM_CONTINUED_FRACTIONS, denominators);

    /* With denominators found, trial multiples of them until the period is found. */
    for (unsigned int d = 0; d < NUM_CONTINUED_FRACTIONS; d++) {    /* d => denominator. */
        for (unsigned int m = 1; m < TRIALS_PER_DENOMINATOR + 1; m++) { /* m => multiple. */
            *period = m * denominators[d];

            if (INT_POW(a, *period) % C == 1 ) {
                period_found = true;
                break;
            }
        }

        if (period_found) {
            break;
        }
    }

    free(denominators);

    if (!period_found) {
        return PERIOD_NOT_FOUND;
    }

    return NO_ERROR;
}


/****************************************************************************************
    shors_algorithm -- Execute Shor's algorithm for the factorisation of a number. The
                       quantum mechanical aspects are simulated, and some of the checks
                       to see if a number can be factorised easily classically are
                       omitted such that the quantum aspects can be tested on these
                       classically factorisable integers.
    
    Parameters:
        unsigned int factors[2]
            The quantity in which the factors will be stored, if found, on completion
            of the function.
        
        unsigned int C
            The number passed to the program to factorise, as per the notation in the
            Candela reference.
        
        unsigned int forced_trial_int
            If this quantity is non-zero an attempt to find the period is made 
            with only this as the trial integer. If this quantity is 0, the possible
            trial integers are looped over until a period is found.
        
        Register reg
            Contains information regarding the qubit register, such as the size of the
            sub registers.
        
        Assets *assets
            Contains the matrices and state vectors needed to perform the quantum
            computations.
        
        gsl_rng *rng
            The random number generator as implemented by GSL.
    
    Returns:
        ErrorCode error
            Describes if any errors have occured, including not finding a period and
            hence a factorisation.

 ****************************************************************************************/
static ErrorCode shors_algorithm(unsigned int factors[2], unsigned int C, unsigned int forced_trial_int, Register reg, GateMatrices matrices, gsl_rng *rng)
{
    ErrorCode error;
    unsigned int period;
    struct timespec start, stop; /* Used to track the start and end times of the simulation. */
    double time_elapsed;         /* Holds the time elapsed by a simulation. */

    printf("\n --- Finding factors...\n\n");

    /* Start simulation timer. */
    clock_gettime(CLOCK_REALTIME, &start);

    /*
        If a trial integer has been forced by the user, only attempt to find the period
        with that integer, as below.
    */
    if (forced_trial_int != 0) {
        if (verbose) {
            printf(" --- Forced trial integer a = %d, finding period ...\n", forced_trial_int);
        }

        error = find_period(&period, C, forced_trial_int, reg, matrices, rng);
        if (error == PERIOD_NOT_FOUND) {
            printf(" --- A valid period was not found and hence C = %d could not be factorised.\n", C);
            return PERIOD_NOT_FOUND;
        }

        if (period % 2 != 0) {
            if (verbose) {
                printf(" --- Period was found to be %d, but it did not pass the validity requirements.\n", period);
            }
            printf(" --- A valid period was not found and hence C = %d could not be factorised.\n", C);
            return PERIOD_NOT_FOUND;

        } else if (INT_POW(forced_trial_int, period / 2) % C == C - 1) {
            if (verbose) {
                printf(" --- Period was found to be %d, but it did not pass the validity requirements\n", period);
            }
            printf(" --- A valid period was not found and hence C = %d could not be factorised.\n", C);
            return PERIOD_NOT_FOUND;
        }

        if (verbose) {
            printf(" --- A valid period = %d has been found so the factors of C = %d have been found quantum mechanically.\n\n", period, C);
        }

        factors[0] = greatest_common_divisor(INT_POW(forced_trial_int, period / 2) + 1, C);
        factors[1] = greatest_common_divisor(INT_POW(forced_trial_int, period / 2) - 1, C);

        if (factors[0] == 1 || factors[1] == 1) {
            printf(" --- The factors found are trivial, consider trying a different trial integer.\n");
        }

        /* Stop simulation timer. */
        clock_gettime(CLOCK_REALTIME, &stop);

        if (verbose) {
            /* Derive time_elapsed from the timespec struct instances. */
            time_elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) / 1e9;
            printf(" --- Time to run Shor's Algorithm: %.6fs.\n", time_elapsed);
        }

        return NO_ERROR;
    }

    /*
        If a trial integer has not been specified by the user, loop over valid integers
        1 < a < C until a valid period hence factors are found.
    */
    for (unsigned int trial_int = 2; trial_int < C - 1; trial_int++) {
        if (verbose) {
            printf(" --- Trial integer a = %d, finding period ...\n", trial_int);
        }

        error = find_period(&period, C, trial_int, reg, matrices, rng);
        if (error == PERIOD_NOT_FOUND) {
            if (verbose) {
                printf(" --- A valid period could not be found for a = %d.\n\n", trial_int);
            }
            continue;
        }

        if (period % 2 != 0) {
            if (verbose) {
                printf(" --- Period was found to be %d, but it did not pass the validity requirements.\n\n", period);
            }
            continue;

        } else if (INT_POW(forced_trial_int, period / 2) % C == C - 1) {
            if (verbose) {
                printf(" --- Period was found to be %d, but it did not pass the validity requirements.\n\n", period);
            }
            continue;
        }

        if (verbose) {
            printf(" --- A valid period = %d has been found so the factors of C = %d have been found quantum mechanically.\n\n", period, C);
        }

        factors[0] = greatest_common_divisor(INT_POW(trial_int, period / 2) + 1, C);
        factors[1] = greatest_common_divisor(INT_POW(trial_int, period / 2) - 1, C);

        if (factors[0] == 1 || factors[1] == 1) {
            printf(" --- Factors found are trivial. Continuing to find non-trivial factors.\n");
            continue;
        }

        /* Stop simulation timer. */
        clock_gettime(CLOCK_REALTIME, &stop);

        if (verbose) {
            /* Derive time_elapsed from the timespec struct instances. */
            time_elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) / 1e9;
            printf(" --- Time to run Shor's Algorithm: %.6fs.\n", time_elapsed);
        }

        return NO_ERROR;
    }

    printf(" --- A valid period was not found and hence C = %d could not be factorised.\n", C);

    /* Stop simulation timer. */
    clock_gettime(CLOCK_REALTIME, &stop);

    if (verbose) {
        /* Derive time_elapsed from the timespec struct instances. */
        time_elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) / 1e9;
        printf(" --- Time to run Shor's Algorithm: %.6fs.\n", time_elapsed);
    }

    return PERIOD_NOT_FOUND;
}


/*********** SETUP FUNCTIONS **********/


/****************************************************************************************
    parse_command_line_args -- Take the user's input and parse the parameters, such
                               as the number subject to Shor's algorithm.
    
    Parameters:
        int argc
            As usual, the number of command line arguments passed to the program.

        char *argv[]
            As usual, an array of strings containing the passed command line
            arguments.
        
        Register *reg
            A pointer to a struct of which to populate with information about the
            quantum register, as provided by the user.
        
        unsigned int *C
            A pointer to the storage of the  number to be factorised by Shor's
            algorithm.
        
        unsigned int *forced_trial_int
            A pointer to the storage of an integer forced to be trialed in Shor's
            algorithm. It is passed as 0, and if left that way, all possible trial
            integers will be looped over to find the period hence factors. If it
            is changed to be non-zero, only this integer will be used to attempt to
            find the period.
    
    Returns:
        ErrorCode error
            Describes if any errors have occured, including the passing of invalid
            arguments to the program.

 ****************************************************************************************/
static ErrorCode parse_command_line_args(int argc, char *argv[], Register *reg, unsigned int *C, unsigned int *forced_trial_int)
{
    extern char *optarg;    /* External variable used in getopt. Stores the argument passed. */
    extern int optind;      /* External variable used in getopt. Tracks the number of arguments passed. */
    const char *usage = "Usage: ./qc_shor.exe -C num -L L_reg_size -M M_reg_size [-i trial_int] [-v] [-V]\n";
    int arg;                /* The result of getopt, containing the flag of the argument passed. */

    /* To ensure the validty of C, the size of L and the size of M. */
    bool C_flag = false;
    bool L_flag = false;
    bool M_flag = false;

    while ((arg = getopt(argc, argv, "C:L:M:i:vV")) != -1) {
        switch(arg) {
            case 'C':
                *C = atoi(optarg);
                C_flag = true;
                break;
            
            case 'L':
                reg->L_size = atoi(optarg);
                L_flag = true;
                break;
            
            case 'M':
                reg->M_size = atoi(optarg);
                M_flag = true;
                break;
            
            case 'v':   /* Controls lower level of verbose. */
                verbose = true;
                break;
            
            case 'V':   /* Controls higher level of verbose. */
                verbose = true;
                very_verbose = true;
                break;
            
            case 'i':
                *forced_trial_int = atoi(optarg);
                break;
            
            case '?':
                /* Invalid option information printed internally by getopt, simply print usage after. */
                printf(usage);
                return BAD_ARGUMENTS;
        }
    }

    if (!C_flag) {
        fprintf(stderr, "Error: Number to be factorised 'C' not given.\n");
        printf(usage);
        return BAD_ARGUMENTS;
    }

    if (!L_flag) {
        fprintf(stderr, "Error: Size of L register not given.\n");
        printf(usage);
        return BAD_ARGUMENTS;
    }

    if (!M_flag) {
        fprintf(stderr, "Error: Size of M register not given.\n");
        printf(usage);
        return BAD_ARGUMENTS;
    }

    if (C <= 0) {
        fprintf(stderr, "Error: Number to be factorised C is invalid.\n");
        printf(usage);
    }

    if (reg->L_size <= 0) {
        fprintf(stderr, "Error: L is invalid.\n");
        printf(usage);
    }

    if (reg->M_size <= 0) {
        fprintf(stderr, "Error: M is invalid.\n");
        printf(usage);
    }

    reg->num_qubits = reg->M_size + reg->L_size;

    /* Find num states by 2^(num_qubits), but not limited by unsigned int data type is an INT_POW(). */
    reg->num_states = 1;
    for (unsigned int pow_ = 0; pow_ < reg->num_qubits; pow_++) {
        reg->num_states *= 2;
    }

    return NO_ERROR;
}


/****************************************************************************************
    main -- Facilitates the high-level governance of the program including setup,
            execution and cleanup.
    
    Parameters:
        int argc
            As usual, the number of command line arguments passed to the program.

        char *argv[]
            As usual, an array of strings containing the passed command line
            arguments.

    Returns:
        int error
            Describes if any errors have occured during the execution of the program.

 ****************************************************************************************/
int main(int argc, char *argv[])
{
    GateMatrices matrices;
    Register reg;                       /* Contains information regarding the qubit register. */
    ErrorCode error;                    /* Return code of the program. */
    const gsl_rng_type *rng_type;       /* The type of rng used in the program. Default is Mersenne twister. */
    gsl_rng *rng;                       /* Instance of random number generator. */
    unsigned int C;                     /* Number to be factorised by the program. */
    unsigned int factors[2];            /* Storage of the results of the program - the factors of C. */
    unsigned int forced_trial_int = 0;  /* Determines whether the trial integer should be forced, and what it should be. */

    /* Setup rng, seeded by an integer derived from the current time. */
    rng_type = gsl_rng_mt19937;
    rng = gsl_rng_alloc(rng_type);
    ALLOC_CHECK(rng);
    gsl_rng_set(rng, (unsigned) time(NULL));

    error = parse_command_line_args(argc, argv, &reg, &C, &forced_trial_int);
    ERROR_CHECK(error);

    /* Issue warnings concerning the reliablity of the results of the program, given the size of the register. */
    issue_warnings(C, reg);

    /* 
        Setup contents of assets, including matrices and vector states.
        The most memory consuming objects are allocated here straight away
        such that the program does not run out of memory mid-way through computation.

        The GSL error handler has not been turned off so will check the return codes
        automatically. However, should this be turned off in future, the ALLOC_CHECK
        macro is called to check the allocation.
    */
    reg.state_a = gsl_vector_complex_alloc(reg.num_states);
    ALLOC_CHECK(reg.state_a);
    reg.state_b = gsl_vector_complex_alloc(reg.num_states);
    ALLOC_CHECK(reg.state_b);
    matrices.comp_matrix = gsl_spmatrix_char_alloc(reg.num_states, reg.num_states);
    ALLOC_CHECK(matrices.comp_matrix);
    matrices.result_matrix = gsl_spmatrix_char_alloc_nzmax(reg.num_states, reg.num_states, reg.num_states, GSL_SPMATRIX_CSR);
    ALLOC_CHECK(matrices.result_matrix);

    reg.current_state = &reg.state_a;
    reg.new_state = &reg.state_b;

    /* With the setup complete, execute Shor's algorithm. */
    error = shors_algorithm(factors, C, forced_trial_int, reg, matrices, rng);

    /* Cleanup. */
    gsl_vector_complex_free(reg.state_a);
    gsl_vector_complex_free(reg.state_b);
    gsl_spmatrix_char_free(matrices.result_matrix);
    gsl_spmatrix_char_free(matrices.comp_matrix);
    gsl_rng_free(rng);

    if (error == NO_ERROR) {
        fprintf(stdout, " --- Factors of %d found: (%d, %d).\n", C, factors[0], factors[1]);
        if (C / factors[0] != factors[1]) {
            fprintf(stdout, " --- These factors are incorrect. Consider increasing register sizes as per the warnings.\n");
        }
    } else if (error == PERIOD_NOT_FOUND) {
        /* Error messages displayed within shors_algorithm(). */
        return PERIOD_NOT_FOUND;
    } else {
        return UNKNOWN_ERROR;
    }

    return NO_ERROR;
}