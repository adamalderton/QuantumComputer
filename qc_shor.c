/****************************************************************************************
 * Program --  qc_shor.c                                                                *
 *                                                                                      *
 * Author                                                                               *
 *      Adam Alderton (aa816@exeter.ac.uk)                                              *
 *                                                                                      *
 * Version                                                                              *
 *                                                                                      *
 * Revision Date                                                                        *
 *                                                                                      *
 * Purpose                                                                              *
 * 
 * Requirements and Compilation
 *                                                                                      *
 * Usage                                                                                *
 *                                                                                      *
 * Options                                                                              *
 *                                                                                      *
 * Notes                                                                                *
 *                                                                                      *
 * Limitations                                                                          *
 *                                                                                      *
 * References                                                                           *
 *      M. Galassi, J. Davies, J. Theiler, B. Gough, G. Jungman, P. Alken, M. Booth,    *
 *      F. Rossi and R. Ulerich, "GNU Scientific Library 2.6 Reference Manual", 2019.   *
 *                                                                                      *
 ****************************************************************************************/

/***********************************PREPROCESSOR*****************************************/

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_spblas.h>

#define VERSION "1.0.0"
#define REVISION_DATE "04/02/2021"

#define LINE_BUFFER_LENGTH 1024     /* Large buffer to read in lines of files. */
#define MAX_FILENAME_LENGTH 128

#define HADAMARD_SCALE 1.0/M_SQRT2
#define NULL_ALT_ELEMENT gsl_complex_rect(0.0, 0.0)
#define NO_CONDITIONAL_QUBIT -1

#define LOW_POWER_TOLERANCE 5
#define ALT_ELMNT -INT_MAX

/* 
 * Many functions below return an errorcode. This macro is called after these functions return
 * and checks for an error. If so, this ends the function in which the error
 * occured returns the error code. If the error raising function is called within a function,
 * that calling function is also returned with the same error code. This repeats until the scope of the "main"
 * function is reached. Therefore, this macro passes the error code up the stack until it is eventually returned by main.
 */
#define ERROR_CHECK(error) \
    if (error != NO_ERROR) { \
        return error; \
    }


/* Checks for the success of the opening of a file. */
#define FILE_CHECK(file, filename) \
    if (file == NULL) { \
        fprintf(stderr, "Error: Unable to open file \"%s\".\n", filename);\
        return BAD_FILENAME; \
    }

/* Checks for the success of a line read from a file. */
#define READLINE_CHECK(result, filename, in_file) \
    if (result == NULL) { \
        fprintf(stderr, "Error: Could not read file \"%s\".\n", filename); \
        fclose(in_file); \
        return BAD_FILE; \
    }

/* 
 * Gets the binary 1 or 0 stored in the n'th bit of int (from the right) 
 * by bit masking followed by a bitwise and. That is, 00000001 is 
 * leftshifted n places to address the n'th bit (from the right) in int.
 * THe value resulting from the bitwise and will result in either 0,
 * or a finite power of two, corresponding to the nth bit. That is, 2^n.
 * Therefore, the result is rightshfted n places to yield either 1 or 0.
 */
#define GET_BIT(int, n) \
    ( (int & (1 << n)) >> n )

#define INT_POW(base, power) \
    ( (int) (pow(base, power) + 0.5) )

#define NON_ZERO_ESTIMATE(num_qubits) \
    ( num_qubits )

/***********************************TYPEDEFS AND GLOBALS*********************************/

/* Enum to store the various error codes than can be returned within this program. */
typedef enum {
    NO_ERROR = 0,
    GSL_ERROR,
    BAD_ARGUMENTS,
    BAD_FILENAME,
    BAD_FILE,
    UNKNOWN_ERROR,
} ErrorCode;

typedef enum {
    HADAMARD_GATE_TYPE,
    C_PHASE_SHIFT_GATE_TYPE
} GateType;

typedef struct {
    GateType type;
    int base[4][4];
} Gate;

typedef struct {
    int register_size[2];
    gsl_vector_complex **current_state;
    gsl_vector_complex **new_state;
    gsl_vector_complex *state_a;
    gsl_vector_complex *state_b;
    gsl_spmatrix_int *result_matrix;
    gsl_spmatrix_int *comp_matrix;
} Assets;

/* Global Variables */

/* 
 * Does NOT contain the necessary scale of 1/sqrt(2),
 * such that it can be stored as an integer matrix.
 * This scalar factor is implemented later in appropriate functions.
 */
const Gate HADAMARD_GATE = {
    HADAMARD_GATE_TYPE,
    {
        {1, 1, 0, 0},
        {1, -1, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0}
    }
};

const Gate C_PHASE_SHIFT_GATE = {
    C_PHASE_SHIFT_GATE_TYPE,
    {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, ALT_ELMNT}
    }
};

int num_qubits;
int num_states;

/***********************************FUNCTION PROTOTYPES**********************************/
static void swap_states(Assets *assets);
static void display_state(gsl_vector_complex *state);
static void operate_matrix(Assets *assets, double scale, gsl_complex alt_element);

/****************************************************************************************/

static void build_matrix(Assets *assets, Gate gate, int c_qubit_num, int qubit_num)
{
    int not_xor_ij;
    int element;
    bool dirac_deltas_non_zero;

    int c_q_address = (num_qubits - 1) - c_qubit_num;
    int q_address = (num_qubits - 1) - qubit_num;

    for (int i = 0; i < num_states; i++) {
        for (int j = 0; j < num_states; j++) {
            dirac_deltas_non_zero = true;

            not_xor_ij = ~(i ^ j);

            for (int b = 0; b < num_states; b++) {

                if ( (b != q_address) && (b != c_q_address) ) {
                    if (GET_BIT(not_xor_ij, b) == 0) {
                        dirac_deltas_non_zero = false;
                        break;
                    }
                }
            }

            if (dirac_deltas_non_zero) {

                if (gate.type == HADAMARD_GATE_TYPE) {
                    element = gate.base[GET_BIT(i, q_address)][GET_BIT(j, q_address)];
                } else {
                    element = gate.base
                        [(2*GET_BIT(i, c_q_address)) + GET_BIT(i, q_address)]
                        [(2*GET_BIT(j, c_q_address)) + GET_BIT(j, q_address)];
                }

                gsl_spmatrix_int_set(assets->comp_matrix, i, j, element);
            }
        }
    }

    gsl_spmatrix_int_csc(assets->result_matrix, assets->comp_matrix);
}

static void c_phase_shift_gate(Assets *assets, int c_qubit_num, int qubit_num, double theta)
{
    build_matrix(assets, C_PHASE_SHIFT_GATE, c_qubit_num, qubit_num);

    operate_matrix(assets, 1.0, gsl_complex_polar(1.0, theta));

    swap_states(assets);
}

static void hadamard_gate(Assets *assets, int qubit_num)
{

    build_matrix(assets, HADAMARD_GATE, NO_CONDITIONAL_QUBIT, qubit_num);

    operate_matrix(assets, HADAMARD_SCALE, NULL_ALT_ELEMENT);

    swap_states(assets);
}

/* 
 * This gate is built differently to other gates, so it is built within this function.
 * Therefore, the build_matrix function is not used.
 */
static void c_amodc_gate(Assets *assets, int c_qubit_num, int atox, int C)
{
    int A; /* Holds a^x (mod C). */
    int M_size;
    int j; /* The column of the matrix in row k in which the 1 resides. */
    int f; /* Used in the calculation of the permutation matrix. */

    A = atox % C;
    M_size = assets->register_size[1];

    /* Using notation from instruction document, loop over rows (k) of matrix. */
    for (int k = 0; k < num_states; k++) {

        /* If l_0 (c_qubit_num) is 0, j = k. */
        if (GET_BIT(k, c_qubit_num) == 0) {
            gsl_spmatrix_int_set(assets->comp_matrix, k, k, 1);
            continue;
        
        /* If l_0 = 1 ... */
        } else {

            /* f must be calculated, which is the decimal value stored in the M register. */
            f = 0;
            for (int b = 0; b < M_size; b++) {
                f += GET_BIT(k, b) << b;
            }

            /* 
             * The calculation above could have been omitted, and just the bits
             * above and including the nearest power of 2 to C, rounding up,
             * could have been tested for any values of 1. However, for simplicity
             * and readability, and considering the speed of bitwise calculations,
             * this approach of calculating f every time was taken.
             */
            if (j >= C) {

                gsl_spmatrix_int_set(assets->comp_matrix, k, k, 1);
                continue;

            } else {

                /* Calculate f', which can be stored in f for simplicity. */
                f = (A * f) % C;

                /* Next, build up the integer value j using f'. */
                j = 0;

                /* M register (concerning f'). */
                for (int b = 0; b < M_size; b++) {
                    j += GET_BIT(f, b) << b;
                }

                /* L register (concerning k). */
                for (int b = M_size; b < num_qubits; b++) {
                    j += GET_BIT(k, b) << b;
                }

                gsl_spmatrix_int_set(assets->comp_matrix, k, j, 1);
            }
        }
    }

    /* Finally, compress the matrix stored in comp_matrix into result_matrix. */
    gsl_spmatrix_int_csc(assets->result_matrix, assets->comp_matrix);



    /* Now the matrix has been built, operate it as with any other gate. */
    //operate_matrix(assets, 1.0, NULL_ALT_ELEMENT);

    swap_states(assets);
}

static void display_state(gsl_vector_complex *state)
{
    for (int i = 0; i < num_states; i++) {
        printf("|");
        //printf("|%d%d%d> ", GET_BIT(i, 2), GET_BIT(i, 1), GET_BIT(i, 0));
        for (int b = num_qubits - 1; b >= 0; b--) {
            printf("%d", GET_BIT(i, b));
        }
        printf("> ");

        printf("%.3f\n", gsl_complex_abs(gsl_vector_complex_get(state, i)));
    }

}

static void swap_states(Assets *assets)
{
    gsl_vector_complex **temp;

    temp = assets->new_state;
    assets->new_state = assets->current_state;
    assets->current_state = temp;

    /* Reset comp_matrix. */
    gsl_spmatrix_int_set_zero(assets->comp_matrix);
}

static void operate_matrix(Assets *assets, double scale, gsl_complex alt_element)
{
    gsl_vector_complex *n_state;
    gsl_vector_complex *c_state;
    gsl_spmatrix_int *mat;
    int n_stride;
    int c_stride;
    double c_real;
    double c_imag;
    double m_real;
    double m_imag;

    n_state = *assets->new_state;
    c_state = *assets->current_state;
    mat = assets->result_matrix;

    n_stride = n_state->stride;
    c_stride = c_state->stride;

    if (scale == 0.0) {
        gsl_vector_complex_set_zero(n_state);
        return;
    }

    /* Apply scale to current state, as to reduce double multiplications in for loops below. */
    if (scale != 1.0) {
        gsl_vector_complex_scale(c_state, gsl_complex_rect(scale, 0.0));
    }

    gsl_vector_complex_set_zero(n_state);

    for (int j = 0; j < num_states; j++) {
        for (int p = mat->p[j]; p < mat->p[j + 1]; p++) {

            /* Retrieve matrix element. */
            m_real = mat->data[p];
            m_imag = 0.0;

            if ((int) m_real == ALT_ELMNT) {
                m_real = GSL_REAL(alt_element);
                m_imag = GSL_IMAG(alt_element);
            }

            /* Retrieve corresponding element in current_state vector. */
            c_real = c_state->data[2 * c_stride * j];
            c_imag = c_state->data[2 * c_stride * j + 1];

            /* Real part. */
            n_state->data[2 * n_stride * mat->i[p]] += (m_real * c_real) - (m_imag * c_imag);

            /* Imaginary part. */
            n_state->data[2 * n_stride * mat->i[p] + 1] += (m_real * c_imag) + (m_imag * c_real);
        }
    }

}

static int greatest_common_divisor(int a, int b)
{
    int temp;

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

static bool is_power(int small_int, int C)
{
    /* If C is divisible by small_int, it is a power of small_int */
    if (C % small_int == 0) {
        return true;
    } else {
        return false;
    }
}

static int find_period(Assets *assets, int a, int C, int L, int M)
{
    int period;

    /* Apply Hadamard gate to qubits in the L register. */
    for (int i = 0; i < L; i++) {
        hadamard_gate(assets, i);
    }

    /* 
     * Apply conditional a^x (mod C) gate to M register. 
     * Here, x is the power of 2 corresponding to the qubit
     */
    for (int l = 0; l < L; l++) {
        for (int m = L; m < num_qubits; m++) {
            //c_amodc_gate(assets,);
            1 == 1;
        }
    }

    /* Inverse quantum Fourier transform (7 qubits only). */


    return period;
}

static ErrorCode shors_algorithm(Assets *assets, int factors[2], int C, int L, int M)
{
    int period;
    int gcd;

    for (int i = 2; i < LOW_POWER_TOLERANCE; i++) {
        if (is_power(i, C)) {
            factors[0] = i;
            factors[1] = C / i;
            return NO_ERROR;
        }
    }

    for (int trial_int = LOW_POWER_TOLERANCE; trial_int < C; trial_int++) {
        
        gcd = greatest_common_divisor(trial_int, C);

        if (gcd > 1) {
            factors[0] = gcd;
            factors[1] = C / gcd;
            
            return NO_ERROR;
        }

        period = find_period(assets, trial_int, C, L, M);

        if ( (period % 2 == 0) && ( (INT_POW(trial_int, period / 2) + 1) % C == 0 ) ) {
            continue;
        } else {
            continue;
        }

        factors[0] = greatest_common_divisor(INT_POW(trial_int, period / 2) + 1, C);
        factors[1] = greatest_common_divisor(INT_POW(trial_int, period / 2) - 1, C);

        break;
    }

    return NO_ERROR;
}

int main(int argc, char *argv[])
{
    ErrorCode error;
    Assets assets;
    int factors[2];

    // parse_command_line_args(argc, char *argv[]);
    
    num_qubits = 7;
    num_states = INT_POW(2, num_qubits);

    assets.register_size[0] = 3; /* L register. */
    assets.register_size[1] = 4; /* M register. */

    assets.state_a = gsl_vector_complex_alloc(num_states);
    assets.state_b = gsl_vector_complex_alloc(num_states);
    assets.comp_matrix = gsl_spmatrix_int_alloc(num_states, num_states);
    assets.result_matrix = gsl_spmatrix_int_alloc_nzmax(num_states, num_states, NON_ZERO_ESTIMATE(num_qubits), GSL_SPMATRIX_CSC);

    assets.current_state = &assets.state_a;
    assets.new_state = &assets.state_b;

    /* Move me into shors algorithm. */
    gsl_vector_complex_set_zero(*assets.current_state);
    gsl_vector_complex_set(*assets.current_state, 0, gsl_complex_polar(1.0, 0.0));

    //error = shors_algorithm(&assets, factors, 15, 3, 4);
    //ERROR_CHECK(error);


    gsl_vector_complex_free(assets.state_a);
    gsl_vector_complex_free(assets.state_b);
    gsl_spmatrix_int_free(assets.result_matrix);
    gsl_spmatrix_int_free(assets.comp_matrix);

    fprintf(stdout, "Factors: (%d, %d)\n", factors[0], factors[1]);

    return NO_ERROR;
}