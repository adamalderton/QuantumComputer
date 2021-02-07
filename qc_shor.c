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

// const double HADAMARD_2x2[2][2] = {
//     {(1.0/M_SQRT2) * 1.0, (1.0/M_SQRT2) * 1.0},
//     {(1.0/M_SQRT2) * 1.0, (1.0/M_SQRT2) * -1.0}
// };

const int HADAMARD_2x2[2][2] = {
    {1, 1},
    {1, -1}
};

int num_qubits = 3;


/***********************************FUNCTION PROTOTYPES**********************************/


/****************************************************************************************/

static ErrorCode build_hadamard_matrix(gsl_spmatrix_complex *hadamard, int qubit_num)
{
    unsigned char not_xor_ij;
    gsl_complex element;
    
    /* Needed as binary numbers will be addressed right to left. */
    int nth_address = (num_qubits - 1) - qubit_num;

    bool element_non_zero;
    int spmatrix_operation_success;

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            element_non_zero = true;
            
            /* 
             * Bitwise EXCLUSIVE OR of integers i and j, followed by a
             * bitwise NOT operation.
             * This is similar to the AND operation, except 1 is returned for 
             * two 1s and for two 0s, as per the nature of the Kronecker delta.
             */
            not_xor_ij = ~(i ^ j);

            /* Loop over bits in not_xor_ij. */
            for (int b = 0; b < 8; b++) {
    
                /* 
                 * If at least one bit in not_xor_ij is 0, the (i,j)'th element in the matrix will be 0.
                 * All elements of hadamard matrix passed are already initialised to 0, so break 'b' loop.
                 */

                if (b != nth_address) {
                    if (GET_BIT(not_xor_ij, b) == 0) {
                        element_non_zero = false;
                        break;
                    }
                }
            }

            if (element_non_zero) {

                /* 
                 * If all the Kronecker deltas in this term are 1, what's left is to actually find the 
                 * H value. To do this, extract the qubit_num'th digits in the binary representations
                 * of i and j. These values, each 0 or 1, are the indices of the value to extract from
                 * the 2x2 Hadamard materix, HADAMARD_2x2.
                 */
                int hi = GET_BIT(i, nth_address);
                int hj = GET_BIT(j, nth_address);


                GSL_SET_REAL(&element, HADAMARD_2x2[hi][hj]);
                GSL_SET_IMAG(&element, 0);

                spmatrix_operation_success = gsl_spmatrix_complex_set(hadamard, i, j, element);
                if (spmatrix_operation_success != GSL_SUCCESS) {
                    return GSL_ERROR;
                }

            }
        }
    }

    /* Convert matrix to CSC format for efficient calculations in future. */
    // spmatrix_operation_success = gsl_spmatrix_complex_csc(hadamard, temp_hadamard_coo);
    // if (spmatrix_operation_success != GSL_SUCCESS) {
    //     return GSL_ERROR;
    // }

    return NO_ERROR;
}

static ErrorCode temp(int n)
{
    ErrorCode error;
    gsl_vector_complex *state;
    gsl_spmatrix_complex *hadamard;

    /* Initialises hadamard (sparse) matrix and sets all elements to 0. */
    hadamard = gsl_spmatrix_complex_alloc(8, 8);

    state = gsl_vector_complex_alloc(8);

    gsl_complex element = gsl_complex_rect( (1.0/sqrt(8.0)), 0.0 );

    for (int i = 0; i < 8; i++) {
        gsl_vector_complex_set(state, i, element);
    }

    error = build_hadamard_matrix(hadamard, n); // 1 is second qubit
    ERROR_CHECK(error);

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            printf("%d  ", (int) (GSL_REAL(gsl_spmatrix_complex_get(hadamard, i, j))));
        }
        printf("\n");
    }

    gsl_spmatrix_complex_free(hadamard);
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
    /* #states = 2^(#qubits) */

    /*
     * For writing own complex sparse matrix multiplication:
     * https://github.com/ampl/gsl/blob/master/spblas/spdgemv.c
     * 
     * Ideally, if possible, would be good to have in-place computation.
     */

    int n = atoi(argv[1]);

    temp(n);



    return NO_ERROR;
}