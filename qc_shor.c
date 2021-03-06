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
        Example:
            ./qc_shor.exe -C 33 -L 5 -M 5 -f 7

            This example should factorise 33 into 3 and 11, using 10 qubits and using 7
            as the forced trial integer.

        Mandatory command line arguments:
            -C [positive integer]
                The number to be factorised by the program.

            -L [positive integer]
                The size of the L sub-register of the qubit register.
            
            -M [positive integer]
                The size of the M sub-register of the qubit register.
        
        Options:
            -a [positive integer]
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
        quite unfeasible. If for some reason this needed to be extended, the unsigned
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

/* Used to insert complex elements into otherwise real matrices. */
#define COMPLEX_ELEMENT -DBL_MAX

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
    A basis matrix used in the contruction of conditional phase-shift gate matrices.

    Stored as doubles to prevent unnecessary casting.
*/
const double HADAMARD_BASE_MATRIX[2][2] = {
    {M_SQRT1_2, M_SQRT1_2},
    {M_SQRT1_2, -M_SQRT1_2}
};

/*
    A basis matrix used in the contruction of conditional phase-shift gate matrices.

    The COMPLEX_ELEMENT quantity is used here to mark elements which are not purely real.
*/
const double C_PHASE_SHIFT_BASE_MATRIX[4][4] = {
    {1.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0},
    {0.0, 0.0, 0.0, COMPLEX_ELEMENT}
};

/* Used for the controlled display of progress messages throughout the program. */
bool verbose = false;
bool very_verbose = false;

/********** UTILITY FUNCTIONS **********/

/****************************************************************************************
    swap_states -- Swaps the pointers current_state and new_state between state_a
                   and state_b.
    
    Parameters:
        Register *reg
            Contains current_state, new_state double pointers which are swapped.

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
            Contains current_state and other register size information.
        
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
    r = gsl_rng_uniform(rng); /* Generate random number. */

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
        That is, set the state_num'th state to have a probability of 1 and all
        the others to have a probability of 0.

        This step could be omitted if one wanted to repeatedly measure this state with re-doing
        the computation but this is not in the spirit of true quantum mechanics hence quantum computers.
     */
    gsl_vector_complex_set_zero(current_state);
    gsl_vector_complex_set(current_state, state_num, gsl_complex_rect(1.0, 0.0));

    return state_num;
}


/****************************************************************************************
    reset_register -- Resets the qubit register to the state required to begin applying
                      the quantum circuit. That is, |000 ... 001>.
    
    Parameters:
        Register reg
            Contains the current state of the register which is to be reset.

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
    operate_matrix -- With a complex sparse matrix built in matrix, operate it on
                      the current state of the register stored in *reg->current_state.
                      The result of the calcuation is stored int *reg->current_state.
    
    Parameters:
        Register *reg
            A pointer to the qubit register which contains the states to be operated on.
        
        gsl_spmatrix_complex *matrix
            The complex sparse matrix to operate.

 ****************************************************************************************/
static void operate_matrix(gsl_spmatrix_complex *matrix, Register *reg)
{
    double *current_state_data; /* To prevent repeated pointer dereference. */
    double *new_state_data;     /* To prevent repeated pointer dereference. */
    double *matrix_data;        /* To prevent repeated pointer dereference. */
    int *element_rows;          /* To prevent repeated pointer dereference. */
    int *element_cols;          /* To prevent repeated pointer dereference. */

    unsigned int row;           /* Contains the row in which the current element under consideration is in. */
    unsigned int col;           /* Contains the column in which the current element under consideration is in. */
    double matrix_real;         /* Real part of matrix element. */
    double matrix_imag;         /* Imaginary part of matrix element. */
    double current_real;        /* Real part of current_state element. */
    double current_imag;        /* Imaginary part of current_state element. */

    current_state_data = (*reg->current_state)->data;
    new_state_data = (*reg->new_state)->data;
    matrix_data = matrix->data;

    element_rows = matrix->i;
    element_cols = matrix->p;

    /* Reset new state to zero. */
    gsl_vector_complex_set_zero(*reg->new_state);

    /* Loop over non-zero (nz) elements. */
    for (unsigned long int nz = 0; nz < (unsigned long int) matrix->nz; nz++) {
        row = element_rows[nz];
        col = element_cols[nz];

        /* Extract matrix element. */
        matrix_real = matrix_data[2 * nz];
        matrix_imag = matrix_data[2 * nz + 1];

        /* Extract current state element. */
        current_real = current_state_data[2 * col];
        current_imag = current_state_data[2 * col + 1];

        /* Real part of new state element. */
        new_state_data[2 * row] += (matrix_real * current_real) - (matrix_imag * current_imag);

        /* Imaginary part of new state element. */
        new_state_data[2 * row + 1] += (matrix_real * current_imag) + (matrix_imag * current_real);
    }

    /* Reset matrix in preparation for the next calcuation. */
    gsl_spmatrix_complex_set_zero(matrix);

    /* Ensure result is stored in current state on completion. */
    swap_states(reg);
}


/****************************************************************************************
    hadamard_gate -- Apply the hadamard gate to a qubit in the qubit register.

    Parameters:
        unsigned int qubit_num
            The qubit number to apply the Hadamard gate to. Note: the qubit counting
            starts at 0.
        
        Register *reg
            A pointer to the qubit register.
        
        gsl_spmatrix_complex *matrix
            An allocated sparse complex matrix in which to store the Hadamard matrix.

    Notes:
        The construction of the matrix is as per the instruction in the cited Candela
        reference.

 ****************************************************************************************/
static void hadamard_gate(unsigned int qubit_num, Register *reg, gsl_spmatrix_complex *matrix)
{
    /* 
        Holds the result of the bitwise not operation applied to the 
        result of the bitwise xor operation between the matrix indices i and j
    */
    unsigned long int not_xor_ij;
    gsl_complex element;            /* Element of Hadamard matrix. */
    bool dirac_deltas_non_zero;     /* Determines whether any of the dirac deltas are zero or not. */

    /* Hadamard matrix elements are purely real. */
    GSL_SET_IMAG(&element, 0.0);

    /* Iterate over all possible elements of the matrix. */
    for (unsigned long int i = 0; i < reg->num_states; i++) {
        for (unsigned long int j = 0; j < reg->num_states; j++) {
            dirac_deltas_non_zero = true;

            not_xor_ij = ~(i ^ j);

            /* Check that all of the dirac-deltas are 1 before proceeding. */
            for (unsigned int b = 0; b < reg->num_qubits; b++) {

                if (b != qubit_num) {
                    if (GET_BIT(not_xor_ij, b) == 0) {
                        dirac_deltas_non_zero = false;
                        break;
                    }
                }
            }

            if (dirac_deltas_non_zero) {

                /* Retrieve element from base matrix. */
                GSL_SET_REAL(&element, HADAMARD_BASE_MATRIX[GET_BIT(i, qubit_num)][GET_BIT(j, qubit_num)]);

                gsl_spmatrix_complex_set(matrix, i, j, element);
            }
        }
    }

    operate_matrix(matrix, reg);
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
        
        Register *reg
            A pointer to the qubit register.
        
        gsl_spmatrix_complex *matrix
            An allocated sparse complex matrix in which to store the phase-shift matrix.
    
    Notes:
        The construction of the matrix is as per the instruction in the cited Candela
        reference.

 ****************************************************************************************/
static void c_phase_shift_gate(unsigned int c_qubit_num, unsigned int qubit_num, double theta, Register *reg, gsl_spmatrix_complex *matrix)
{
    /* 
        Holds the result of the bitwise not operation applied to the 
        result of the bitwise xor operation between the matrix indices i and j
    */
    unsigned long int not_xor_ij;
    bool dirac_deltas_non_zero;     /* Determines whether any of the dirac deltas are zero or not. */

    double base_matrix_element;     /* Used to extract element from C_PHASE_SHIFT_BASE_MATRIX. */
    gsl_complex e_i_theta;          /* z = e^{i\theta}, as per the phase shift matrix. */
    gsl_complex element;            /* Element of phase shift matrix. */

    e_i_theta = gsl_complex_polar(1.0, theta);

    /* Iterate over all possible elements of the matrix. */
    for (unsigned long int i = 0; i < reg->num_states; i++) {
        for (unsigned long int j = 0; j < reg->num_states; j++) {
            dirac_deltas_non_zero = true;

            not_xor_ij = ~(i ^ j);

            /* Check that all of the dirac-deltas are 1 before proceeding. */
            for (unsigned int b = 0; b < reg->num_qubits; b++) {

                if ( (b != qubit_num) && (b != c_qubit_num) ) {
                    if (GET_BIT(not_xor_ij, b) == 0) {
                        dirac_deltas_non_zero = false;
                        break;
                    }
                }
            }

            if (dirac_deltas_non_zero) {

                /* Retrieve element from base matrix. */
                base_matrix_element = C_PHASE_SHIFT_BASE_MATRIX
                    [(2*GET_BIT(i, c_qubit_num)) + GET_BIT(i, qubit_num)]
                    [(2*GET_BIT(j, c_qubit_num)) + GET_BIT(j, qubit_num)];
                
                if (base_matrix_element == COMPLEX_ELEMENT) {
                    element = e_i_theta;
                } else {
                    GSL_SET_COMPLEX(&element, base_matrix_element, 0.0);
                }

                gsl_spmatrix_complex_set(matrix, i, j, element);
            }
        }
    }

    operate_matrix(matrix, reg);
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
        
        Register *reg
            A pointer to the qubit register.
        
        gsl_spmatrix_complex *matrix
            An allocated sparse complex matrix in which to store the a^x (mod C) matrix.

    Notes:
        The construction of the matrix is as per the instruction in the cited Candela
        reference.

 ****************************************************************************************/
static void c_amodc_gate(unsigned int C, unsigned long long int atox, unsigned int c_qubit_num, Register *reg, gsl_spmatrix_complex *matrix)
{
    gsl_complex complex_1;  /* The complex representation of 1, to be inserted in the matrix. */
    unsigned int A;         /* Holds a^x (mod C). */
    unsigned int j;         /* The column of the matrix in row k in which the 1 resides. */
    unsigned int f;         /* Used in the calculation of the permutation matrix. */

    complex_1 = gsl_complex_rect(1.0, 0.0);

    /* Compiler automatically casts A, atox and C to yield the correct A. */
    A = atox % C;

    /* Using notation from instruction document, loop over rows (k) of matrix to build it. */
    for (unsigned long int k = 0; k < reg->num_states; k++) {

        /* If l_0 (c_qubit_num) is 0, j = k. */
        if (GET_BIT(k, c_qubit_num) == 0) {
            gsl_spmatrix_complex_set(matrix, k, k, complex_1);
            continue;
        
        /* If l_0 = 1 ... */
        } else {

            /* f must be calculated, which is the decimal value stored in the M register. */
            f = 0;
            for (unsigned int b = 0; b < reg->M_size; b++) {
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

                gsl_spmatrix_complex_set(matrix, k, k, complex_1);
                continue;

            } else {

                /* Calculate f', which can be stored in f for simplicity. */
                f = (A * f) % C;

                /* Next, build up the integer value j using f'. */
                j = 0;

                /* M register (concerning f'). */
                for (unsigned int b = 0; b < reg->M_size; b++) {
                    j += GET_BIT(f, b) << b;
                }

                /* L register (concerning k). */
                for (unsigned int b = reg->M_size; b < reg->num_qubits; b++) {
                    j += GET_BIT(k, b) << b;
                }

                gsl_spmatrix_complex_set(matrix, j, k, complex_1);
            }
        }
    }

    operate_matrix(matrix, reg);
}


/****************************************************************************************
    inverse_QFT -- Perform an inverse quantum Fourier transform on the L register.
    
    Parameters:
        Register *reg
            A pointer to the qubit register.
        
        gsl_spmatrix_complex *matrix
            An allocated sparse complex matrix.

    Notes:
        The inverse quantum Fourier transform is peformed as per the instructions in
        the cited Candela reference.

 ****************************************************************************************/
static void inverse_QFT(Register *reg, gsl_spmatrix_complex *matrix)
{
    double theta;   /* The phase to apply within the phase shift gates. */

    for (int l = reg->L_size + reg->M_size - 1; l >= reg->M_size; l--) {
        hadamard_gate(l, reg, matrix);

        for (int k = l - 1; k >= reg->M_size; k--) {
            theta = M_PI / INT_POW(2, l - k);
            c_phase_shift_gate(l, k, theta, reg, matrix);
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

        Register *reg
            A pointer to the qubit register.
        
        gsl_spmatrix_complex *matrix
            An allocated sparse complex matrix in which to store the a^x (mod C) matrix.

 ****************************************************************************************/
static void quantum_computation(unsigned int C, unsigned int a, Register *reg, gsl_spmatrix_complex *matrix)
{
    unsigned int x = 1; /* The x in f(x) = a^x (mod C). */

    /* Apply Hadamard gate to qubits in the L register. */
    if (very_verbose) {
        printf("         - Applying Hadamard matrices.\n");
    }
    for (unsigned int l = (reg->num_qubits - reg->L_size); l < reg->num_qubits; l++) {
        hadamard_gate(l, reg, matrix);
    }

    if (very_verbose) {
        printf("         - Applying a^x mod (C) gates.\n");
    }
    /* For each bit value in the L register, apply the conditional a^x (mod C) gate. */
    for (unsigned int l = (reg->num_qubits - reg->L_size); l < reg->num_qubits; l++) {
        c_amodc_gate(C, INT_POW(a, x), l, reg, matrix);
        x *= 2;
    }

    if (very_verbose) {
        printf("         - Performing inverse quantum Fourier transform.\n");
    }
    inverse_QFT(reg, matrix);
}


/********** SHOR'S ALGORITHM FUNCTIONS **********/


/****************************************************************************************
    greatest_common_divisor -- Find the greatest common divisor between two integers
                               using an iterative version of Euclid's algorithm.
    
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

    x_tilde = 0;
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
        
        Register *reg
            A pointer to the qubit register.
        
        gsl_spmatrix_complex *matrix
            An allocated sparse complex matrix.
        
        gsl_rng *rng
            The random number generator as implemented by GSL.
    
    Returns:
        ErrorCode error
            Describes if any errors have occured, including not finding a period.

 ****************************************************************************************/
static ErrorCode find_period(unsigned int *period, unsigned int C, unsigned int a, Register *reg, gsl_spmatrix_complex *matrix, gsl_rng *rng)
{
    unsigned int *denominators;             /* The denominators of the fractions in the continued expansion of omega. */
    bool period_found;                      /* True if the period has been found successfully, false if not. */
    unsigned long int measured_state_num;   /* The decimal representation of the binary number representing the measured qubit register. */
    double omega;                           /* The measured x_tilde / 2^L. */

    if (very_verbose) {
        printf("      - Performing quantum computation...\n");
    }
    reset_register(*reg);
    quantum_computation(C, a, reg, matrix);

    if (very_verbose) {
        printf("      - Measuring state...\n");
    }
    measured_state_num = measure_state(*reg, rng);
    omega = read_omega(measured_state_num, *reg);

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

            /* a^p = 1 (mod C) is the valid period condition. */
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

        Register *reg
            A pointer to the qubit register.
        
        gsl_spmatrix_complex *matrix
            An allocated sparse complex matrix.
        
        gsl_rng *rng
            The random number generator as implemented by GSL.
    
    Returns:
        ErrorCode error
            Describes if any errors have occured, including not finding a period and
            hence a factorisation.

 ****************************************************************************************/
static ErrorCode shors_algorithm(unsigned int factors[2], unsigned int C, unsigned int forced_trial_int, Register *reg, gsl_spmatrix_complex *matrix, gsl_rng *rng)
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

        error = find_period(&period, C, forced_trial_int, reg, matrix, rng);
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
                printf(" --- Period was found to be %d, but it did not pass the validity requirements.\n", period);
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

        error = find_period(&period, C, trial_int, reg, matrix, rng);
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
            A pointer to a Register struct to populate with information about the
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
    const char *usage = "Usage: ./qc_shor.exe -C num -L L_reg_size -M M_reg_size [-f trial_int] [-v] [-V]\n";
    int arg;                /* The result of getopt, containing the flag of the argument passed. */

    /* To ensure the validty of C, the size of L and the size of M. */
    bool C_flag = false;
    bool L_flag = false;
    bool M_flag = false;

    while ((arg = getopt(argc, argv, "C:L:M:a:vV")) != -1) {
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
            
            case 'a':
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
    gsl_spmatrix_complex *matrix;       /* Used to build and operate matrices representing gates. */
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
    matrix = gsl_spmatrix_complex_alloc_nzmax(reg.num_states, reg.num_states, 2 * reg.num_states, GSL_SPMATRIX_COO);
    ALLOC_CHECK(matrix);

    reg.current_state = &reg.state_a;
    reg.new_state = &reg.state_b;

    /* With the setup complete, execute Shor's algorithm. */
    error = shors_algorithm(factors, C, forced_trial_int, &reg, matrix, rng);

    /* Cleanup. */
    gsl_vector_complex_free(reg.state_a);
    gsl_vector_complex_free(reg.state_b);
    gsl_spmatrix_complex_free(matrix);
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