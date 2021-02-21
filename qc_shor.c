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

#define LOW_POWER_TOLERANCE 5

#define ALT_ELMT -1

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
    ( (int) M_SQRT2 * num_qubits )

/***********************************TYPEDEFS AND GLOBALS*********************************/

/* Enum to store the various error codes than can be returned within this program. */
typedef enum {
    NO_ERROR = 0,
    GSL_ERROR = 1,
    BAD_ARGUMENTS = 2,
    BAD_FILENAME = 3,
    BAD_FILE = 4,
    UNKNOWN_ERROR = 5,
} ErrorCode;

typedef struct {
    gsl_vector_complex **current_state;
    gsl_vector_complex **new_state;
    gsl_vector_complex *state_a;
    gsl_vector_complex *state_b;
    gsl_spmatrix *result_matrix;
    gsl_spmatrix *comp_matrix;
} Assets;

/* Global Variables */

const double HADAMARD_2x2[2][2] = {
    {(1.0/M_SQRT2) * 1.0, (1.0/M_SQRT2) * 1.0},
    {(1.0/M_SQRT2) * 1.0, (1.0/M_SQRT2) * -1.0}
};

/* TODO: EXPLAIN WHY IT IS DOUBLE TYPE. */
const int CNOT_4x4[4][4] = {
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 0, 1},
    {0, 0, 1, 0}
};

const int aTOx_MODC_4x4[4][4] = {
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, ALT_ELMT, 0},
    {0, 0, 0, ALT_ELMT}
};

const int PHASE_CHANGE_4x4[4][4] = {
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, ALT_ELMT}
};

int num_qubits;
int num_states;


/***********************************FUNCTION PROTOTYPES**********************************/
static void swap_states(Assets *assets);
static void display_state(gsl_vector_complex *state);

static void operate_matrix(Assets *assets, double scale, gsl_complex alt_element);

/****************************************************************************************/

static void build_hadamard_matrix(Assets *assets, int qubit_num)
{
    int not_xor_ij;
    int nth_address;
    bool dirac_deltas_non_zero;

    nth_address = (num_qubits - 1) - qubit_num;

    /* Set all element in comp_matrix to 0. */
    gsl_spmatrix_set_zero(assets->comp_matrix);

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            dirac_deltas_non_zero = true;
            
            /* 
             * Bitwise EXCLUSIVE OR of integers i and j, followed by a
             * bitwise NOT operation.
             * This is similar to the AND operation, except 1 is returned for 
             * two 1s and for two 0s, as per the nature of the Kronecker delta.
             */
            not_xor_ij = ~(i ^ j);

            /* Loop over bits in not_xor_ij. */
            for (int b = 0; b < 2*2*2; b++) {
    
                /* 
                 * If at least one bit in not_xor_ij is 0, the (i,j)'th element in the matrix will be 0.
                 * All elements of hadamard matrix passed are already initialised to 0, so break 'b' loop.
                 */

                if (b != nth_address) {
                    if (GET_BIT(not_xor_ij, b) == 0) {
                        dirac_deltas_non_zero = false;
                        break;
                    }
                }
            }

            if (dirac_deltas_non_zero) {

                /* 
                 * If all the Kronecker deltas in this term are 1, what's left is to actually find the 
                 * H value. To do this, extract the nth_address'th digits in the binary representations
                 * of i and j. These values, each 0 or 1, are the indices of the value to extract from
                 * the 2x2 Hadamard materix, HADAMARD_2x2.
                 */

                gsl_spmatrix_set(assets->comp_matrix, i, j, HADAMARD_2x2[GET_BIT(i, nth_address)][GET_BIT(j, nth_address)]);
            }
        }
    }

    gsl_spmatrix_csc(assets->result_matrix, assets->comp_matrix);
}

static void hadamard_gate(Assets *assets, int qubit_num)
{
    build_hadamard_matrix(assets, qubit_num);

    operate_matrix(assets, 1.0/M_SQRT2, gsl_complex_rect(0.0, 0.0));

    swap_states(assets);
}

static void build_cnot_matrix(Assets *assets, int c_qubit_num, int qubit_num)
{
    int not_xor_ij;
    int element;
    bool dirac_deltas_non_zero;

    int c_q_address = (num_qubits - 1) - c_qubit_num;
    int q_address = (num_qubits - 1) - qubit_num;

    /* Set all element in comp_matrix to 0. */
    gsl_spmatrix_set_zero(assets->comp_matrix);

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            dirac_deltas_non_zero = true;

            not_xor_ij = ~(i ^ j);

            for (int b = 0; b < 2*2*2; b++) {

                if ( (b != q_address) && (b != c_q_address) ) {
                    if (GET_BIT(not_xor_ij, b) == 0) {
                        dirac_deltas_non_zero = false;
                        break;
                    }
                }
            }

            if (dirac_deltas_non_zero) {

                element = CNOT_4x4
                [(2*GET_BIT(i, c_q_address)) + GET_BIT(i, q_address)]
                [(2*GET_BIT(j, c_q_address)) + GET_BIT(j, q_address)];

                gsl_spmatrix_set(assets->comp_matrix, i, j, element);
            }

        }
    }

    gsl_spmatrix_csc(assets->result_matrix, assets->comp_matrix);
}

static void cnot_gate(Assets *assets, int c_qubit_num, int qubit_num)
{
    /*
     * The cnot matrix is to be less efficiently stored as a matrix of doubles,
     * even though its elements will be integers. This is as such to agree with the
     * sparse matrix-vector multiplication methods supplied by GSL. Fortunately,
     * due to the nature of sparse matrices, the memory consumption of the matrix 
     * will remain reasonably low.
     */

    gsl_vector_view new_state_real;
    gsl_vector_view new_state_imag;
    gsl_vector_view state_real;
    gsl_vector_view state_imag;

    build_cnot_matrix(assets, c_qubit_num, qubit_num);

    new_state_real = gsl_vector_complex_real(*assets->new_state);
    new_state_imag = gsl_vector_complex_imag(*assets->new_state);
    state_real = gsl_vector_complex_real(*assets->current_state);
    state_imag = gsl_vector_complex_imag(*assets->current_state);

    gsl_spblas_dgemv(CblasNoTrans, 1.0, assets->result_matrix, &state_real.vector, 0.0, &new_state_real.vector);
    gsl_spblas_dgemv(CblasNoTrans, 1.0, assets->result_matrix, &state_imag.vector, 0.0, &new_state_imag.vector);

    swap_states(assets);
}

static void build_atox_modC_gate(Assets *assets, int c_qubit_num, int qubit_num, int a, int x, int C)
{
    int not_xor_ij;
    int element;
    bool dirac_deltas_non_zero;

    int c_q_address = (num_qubits - 1) - c_qubit_num;
    int q_address = (num_qubits - 1) - qubit_num;

    /* Reset elements in comp_matrix to 0. */
    gsl_spmatrix_set_zero(assets->comp_matrix);

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

                element = aTOx_MODC_4x4
                [(2*GET_BIT(i, c_q_address)) + GET_BIT(i, q_address)]
                [(2*GET_BIT(j, c_q_address)) + GET_BIT(j, q_address)];

                if (element == ALT_ELMT) {
                    element = INT_POW(a, x) % C;
                }

                gsl_spmatrix_set(assets->comp_matrix, i, j, element);
            }
        }
    }

    gsl_spmatrix_csc(assets->result_matrix, assets->comp_matrix);
}

static void atox_modC_gate(Assets *assets, int c_qubit_num, int qubit_num, int a, int x, int C)
{
    gsl_vector_view new_state_real;
    gsl_vector_view new_state_imag;
    gsl_vector_view state_real;
    gsl_vector_view state_imag;

    build_atox_modC_gate(assets, c_qubit_num, qubit_num, a, x, C);

    new_state_real = gsl_vector_complex_real(*assets->new_state);
    new_state_imag = gsl_vector_complex_imag(*assets->new_state);
    state_real = gsl_vector_complex_real(*assets->current_state);
    state_imag = gsl_vector_complex_imag(*assets->current_state);

    gsl_spblas_dgemv(CblasNoTrans, 1.0, assets->result_matrix, &state_real.vector, 0.0, &new_state_real.vector);
    gsl_spblas_dgemv(CblasNoTrans, 1.0, assets->result_matrix, &state_imag.vector, 0.0, &new_state_imag.vector);

    swap_states(assets);
}

// static void phase_change_gate(gsl_vector_complex *state, double theta)
// {
//     gsl_complex temp;

//     /* assets.current_state[i] *= exp(i \theta) */
//     for (int i = 0; i < num_states; i+=2) {
//         temp = gsl_vector_complex_get(state, i);
//         gsl_vector_complex_set(state, i, gsl_complex_mul(temp, gsl_complex_polar(1.0, theta)));
//     }
// }

static void build_phase_change_gate(Assets *assets, int c_qubit_num, int qubit_num, double part)
{
    int not_xor_ij;
    int element;
    bool dirac_deltas_non_zero;

    int c_q_address = (num_qubits - 1) - c_qubit_num;
    int q_address = (num_qubits - 1) - qubit_num;

    /* Reset elements in comp_matrix to 0. */
    gsl_spmatrix_set_zero(assets->comp_matrix);

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

                element = aTOx_MODC_4x4
                [(2*GET_BIT(i, c_q_address)) + GET_BIT(i, q_address)]
                [(2*GET_BIT(j, c_q_address)) + GET_BIT(j, q_address)];

                if (element == ALT_ELMT) {
                    element = part;
                }

                gsl_spmatrix_set(assets->comp_matrix, i, j, element);
            }
        }
    }

    gsl_spmatrix_csc(assets->result_matrix, assets->comp_matrix);
}

static void phase_change_gate(Assets *assets, int c_qubit_num, int qubit_num, double theta)
{
    /* 
     * Need to effectively apply two gates, one for real part and one for the imaginary part.
     * This is done such that the matrix does not need to be stored as a complex matrix.
     */
    gsl_vector_view new_state_real;
    gsl_vector_view new_state_imag;
    gsl_vector_view state_real;
    gsl_vector_view state_imag;

    new_state_real = gsl_vector_complex_real(*assets->new_state);
    new_state_imag = gsl_vector_complex_imag(*assets->new_state);
    state_real = gsl_vector_complex_real(*assets->current_state);
    state_imag = gsl_vector_complex_imag(*assets->current_state);

    gsl_complex mul_factor;

    mul_factor = gsl_complex_polar(1.0, theta);

    /***** Real Part *****/

    build_phase_change_gate(assets, c_qubit_num, qubit_num, GSL_REAL(mul_factor));

    swap_states(assets);

    /***** Imaginary Part *****/

    build_phase_change_gate(assets, c_qubit_num, qubit_num, GSL_IMAG(mul_factor));

    swap_states(assets);
}

static void display_state(gsl_vector_complex *state)
{
    for (int i = 0; i < 8; i++) {
        printf("|%d%d%d> ", GET_BIT(i, 2), GET_BIT(i, 1), GET_BIT(i, 0));

        printf("%.3f\n", gsl_complex_abs(gsl_vector_complex_get(state, i)));
    }

}

static void swap_states(Assets *assets)
{
    gsl_vector_complex **temp;

    temp = assets->new_state;
    assets->new_state = assets->current_state;
    assets->current_state = temp;
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

    /* Simple iterative version of Euclid algorithm. */
    while ((a % b) > 0) {
        temp = a % b;
        a = b;
        b = temp;
    }

    return b;
}

static bool is_power(int small_integer, int C)
{
    // TODO: THIS
    return false;
}

static int find_p(Assets *assets, int a, int C, int L, int M)
{
    int p;

    for (int i = 0; i < L; i++) {
        hadamard_gate(assets, i);
    }

    for (int l = 0; l < L; l++) {
        for (int m = L; m < num_qubits; m++) {
            atox_modC_gate(assets, l, m, a, INT_POW(2, l), C);
        }
    }

    return p;
}

static ErrorCode shors_algorithm(Assets *assets, int *factors, int C, int L, int M)
{
    for (int i = 2; i < LOW_POWER_TOLERANCE; i++) {
        if (is_power(i, C)) {
            factors[0] = i;
            factors[1] = C / i;
            return NO_ERROR;
        }
    }

    for (int a = 2; a < C; a++) {
        int p;
        int gcd = greatest_common_divisor(a, C);

        if (gcd > 1) {
            factors[0] = gcd;
            factors[1] = C / gcd;
            
            return NO_ERROR;
        }

        p = find_p(assets, a, C, L, M);

        if ( (p % 2 == 0) && ( (INT_POW(a, p / 2) + 1) % C == 0 ) ) {
            continue;
        } else {
            continue;
        }

        factors[0] = greatest_common_divisor(INT_POW(a, p / 2) + 1, C);
        factors[1] = greatest_common_divisor(INT_POW(a, p / 2) - 1, C);

        break;
    }

    return NO_ERROR;
}

static void operate_matrix(Assets *assets, double scale, gsl_complex alt_element)
{
    gsl_vector_complex *n_state;
    gsl_vector_complex *c_state;
    gsl_spmatrix *mat;
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

    gsl_vector_complex_set_zero(n_state);

    for (int j = 0; j < num_states; j++) {
        for (int p = mat->p[j]; p < mat->p[j + 1]; p++) {

            /* Retrieve matrix element. */
            m_real = mat->data[p];
            m_imag = 0.0;

            if (m_real == (double) ALT_ELMT) {
                m_real = GSL_REAL(alt_element);
                m_imag = GSL_IMAG(alt_element);
            }

            /* Retrieve corresponding element in current_state vector. */
            c_real = c_state->data[2 * c_stride * j];
            c_imag = c_state->data[2 * c_stride * j + 1];

            /* Real part. */
            n_state->data[2 * n_stride * mat->i[p]] += ( (m_real * c_real) - (m_imag * c_imag) );

            /* Imaginary part. */
            n_state->data[2 * n_stride * mat->i[p] + 1] += ( (m_real * c_imag) + (m_imag * c_real) );
        }
    }

}

/****************************************************************************************
 * main -- Parse command line arguments and execute N body simulation.                  *
 *                                                                                      *
 * Returns                                                                              *
 *      ErrorCode error                                                                 *
 *          Value from the "Errorcode" enum describing what error occurred, if any.     *
 *                                                                                      *
 ****************************************************************************************/
int main(int argc, char *argv[])
{
    ErrorCode error;
    Assets assets;
    int factors[2];

    // parse_command_line_args(argc, char *argv[]);
    
    num_qubits = 3;
    num_states = INT_POW(2, num_qubits);

    assets.state_a = gsl_vector_complex_alloc(num_states);
    assets.state_b = gsl_vector_complex_alloc(num_states);
    assets.comp_matrix = gsl_spmatrix_alloc(num_states, num_states);
    assets.result_matrix = gsl_spmatrix_alloc_nzmax(num_states, num_states, NON_ZERO_ESTIMATE(num_qubits), GSL_SPMATRIX_CSC);

    assets.current_state = &assets.state_a;
    assets.new_state = &assets.state_b;

    gsl_vector_complex_set_zero(*assets.current_state);
    gsl_vector_complex_set(*assets.current_state, 0, gsl_complex_polar(1.0, 0.0));

    //error = shors_algorithm(&assets, factors, 15, 3, 4);
    //ERROR_CHECK(error);
    hadamard_gate(&assets, 0);

    display_state(*assets.current_state);

    gsl_vector_complex_free(assets.state_a);
    gsl_vector_complex_free(assets.state_b);
    gsl_spmatrix_free(assets.result_matrix);
    gsl_spmatrix_free(assets.comp_matrix);

    fprintf(stdout, "Factors: (%d, %d)\n", factors[0], factors[1]);

    return NO_ERROR;
}