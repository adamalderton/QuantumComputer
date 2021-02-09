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
    ( (int & (1<<n)) >> n )

#define INT_POW(base, power) \
    ( (int) (pow(base, power) + 0.5) ) 

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

/* Global Variables */

const double HADAMARD_2x2[2][2] = {
    {(1.0/M_SQRT2) * 1.0, (1.0/M_SQRT2) * 1.0},
    {(1.0/M_SQRT2) * 1.0, (1.0/M_SQRT2) * -1.0}
};

/* TODO: EXPLAIN WHY IT IS DOUBLE TYPE. */
const double CNOT_4x4[4][4] = {
    {1.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 1.0},
    {0.0, 0.0, 1.0, 0.0}
};

int num_qubits;


/***********************************FUNCTION PROTOTYPES**********************************/


/****************************************************************************************/

static void build_hadamard_matrix(gsl_spmatrix **hadamard, int qubit_num)
{
    int not_xor_ij;
    int nth_address;
    bool dirac_deltas_non_zero;
    gsl_spmatrix *temp_hadamard_coo;
    
    nth_address = (num_qubits - 1) - qubit_num;

    temp_hadamard_coo = gsl_spmatrix_alloc(8, 8);

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

                gsl_spmatrix_set(temp_hadamard_coo, i, j, HADAMARD_2x2[GET_BIT(i, nth_address)][GET_BIT(j, nth_address)]);
            }
        }
    }

    *hadamard = gsl_spmatrix_compress(temp_hadamard_coo, GSL_SPMATRIX_CSC);

    gsl_spmatrix_free(temp_hadamard_coo);

}

static void hadamard_gate(gsl_vector_complex *state, int qubit_num)
{
    gsl_spmatrix *hadamard;
    gsl_vector_complex *new_state;
    gsl_vector_view new_state_real;
    gsl_vector_view new_state_imag;
    gsl_vector_view state_real;
    gsl_vector_view state_imag;

    hadamard = gsl_spmatrix_alloc(8, 8);

    build_hadamard_matrix(&hadamard, qubit_num);

    new_state = gsl_vector_complex_calloc(8);

    new_state_real = gsl_vector_complex_real(new_state);
    new_state_imag = gsl_vector_complex_imag(new_state);
    state_real = gsl_vector_complex_real(state);
    state_imag = gsl_vector_complex_imag(state);

    gsl_spblas_dgemv(CblasNoTrans, 1.0, hadamard, &state_real.vector, 0.0, &new_state_real.vector);
    gsl_spblas_dgemv(CblasNoTrans, 1.0, hadamard, &state_imag.vector, 0.0, &new_state_imag.vector);

    /* 
     * Copy new_state into state.
     * This could be more elegantly achieved with pointer manipulation
     * but there are logistical issues with GSL's internal tracking of vectors.
     * Additionally, the state vectors are small so the vector copying functions
     * are not too laborious to contrast the benefits of readability.
     */
    gsl_vector_complex_memcpy(state, new_state);

    gsl_spmatrix_free(hadamard);
    gsl_vector_complex_free(new_state);
}

static void build_cnot_matrix(gsl_spmatrix **cnot, int c_qubit_num, int qubit_num)
{
    int not_xor_ij;
    int element;
    bool dirac_deltas_non_zero;
    gsl_spmatrix *temp_cnot_coo;

    int c_q_address = (num_qubits - 1) - c_qubit_num;
    int q_address = (num_qubits - 1) - qubit_num;

    temp_cnot_coo = gsl_spmatrix_alloc(8, 8);

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

                gsl_spmatrix_set(temp_cnot_coo, i, j, element);
            }

        }
    }

    *cnot = gsl_spmatrix_compress(temp_cnot_coo, GSL_SPMATRIX_CSC);

    gsl_spmatrix_free(temp_cnot_coo);
}

static void cnot_gate(gsl_vector_complex *state, int c_qubit_num, int qubit_num)
{
    /*
     * The cnot matrix is to be less efficiently stored as a matrix of doubles,
     * even though its elements will be integers. This is as such to agree with the
     * sparse matrix-vector multiplication methods supplied by GSL. Fortunately,
     * due to the nature of sparse matrices, the memory consumption of the matrix 
     * will remain reasonably low.
     */
    gsl_spmatrix *cnot;

    gsl_vector_complex *new_state;
    gsl_vector_view new_state_real;
    gsl_vector_view new_state_imag;
    gsl_vector_view state_real;
    gsl_vector_view state_imag;

    cnot = gsl_spmatrix_alloc(8, 8);

    build_cnot_matrix(&cnot, c_qubit_num, qubit_num);

    new_state = gsl_vector_complex_calloc(8);

    new_state_real = gsl_vector_complex_real(new_state);
    new_state_imag = gsl_vector_complex_imag(new_state);
    state_real = gsl_vector_complex_real(state);
    state_imag = gsl_vector_complex_imag(state);

    gsl_spblas_dgemv(CblasNoTrans, 1.0, cnot, &state_real.vector, 0.0, &new_state_real.vector);
    gsl_spblas_dgemv(CblasNoTrans, 1.0, cnot, &state_imag.vector, 0.0, &new_state_imag.vector);

    /* 
     * Copy new_state into state.
     * This could be more elegantly achieved with pointer manipulation
     * but there are logistical issues with GSL's internal tracking of vectors.
     * Additionally, the state vectors are small so the vector copying functions
     * are not too laborious to contrast the benefits of readability.
     */
    gsl_vector_complex_memcpy(state, new_state);

    gsl_spmatrix_free(cnot);
    gsl_vector_complex_free(new_state);
}

static void phase_change_gate(gsl_vector_complex *state, double theta)
{
    gsl_complex temp;

    for (int i = 0; i < 8; i+=2) {
        temp = gsl_vector_complex_get(state, i);
        gsl_vector_complex_set(state, i, gsl_complex_mul(temp, gsl_complex_polar(1.0, theta)));
    }
}

static void display_state(gsl_vector_complex *state)
{
    for (int i = 0; i < 8; i++) {
        printf("|%d%d%d> ", GET_BIT(i, 2), GET_BIT(i, 1), GET_BIT(i, 0));

        printf("%g\n", gsl_complex_abs(gsl_vector_complex_get(state, i)));
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

static int find_p(int a, int C)
{

    return 4;
}

static ErrorCode shors_algorithm(int *factors, int C, int L, int M)
{
    for (int i = 2; i < LOW_POWER_TOLERANCE; i++) {
        if (is_power(i, C)) {
            factors[0] = i;
            factors[1] = C / i;
            return NO_ERROR;
        }
    }

    gsl_vector_complex *state;
    state = gsl_vector_complex_alloc(INT_POW(2, num_qubits));

    for (int a = 2; a < C; a++) {
        int p;
        int gcd = greatest_common_divisor(a, C);

        if (gcd > 1) {
            factors[0] = gcd;
            factors[1] = C / gcd;

            gsl_vector_complex_free(state);
            
            return NO_ERROR;
        }

        p = find_p(a, C);

        if ( (p % 2 == 0) &&
           ( (INT_POW(a, p/2) + 1) % C == 0 ) ) {
            continue;
        } else {
            continue;
        }

        factors[0] = greatest_common_divisor(INT_POW(a, p/2) + 1, C);
        factors[1] = greatest_common_divisor(INT_POW(a, p/2) - 1, C);

    }

    gsl_vector_complex_free(state);

    return NO_ERROR;
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
/*
TODO:
    - One (two) matrix, similar to state, at to prevent many mallocs and frees.
    - Use clever pointer stuff to swap state and new state.
    - (potentially) normalise as you go.
    - Verbose option
    - ALL comments.
    - Find reference for gcd algorithm.
    - Put checks on gcd algorithm?
*/
    ErrorCode error;
    int factors[2];

    // parse_command_line_args(argc, char *argv[]);
    
    num_qubits = 3;

    error = shors_algorithm(factors, 15, 3, 4);
    ERROR_CHECK(error);

    fprintf(stdout, "Factors: (%d, %d)\n", factors[0], factors[1]);

    return NO_ERROR;
}