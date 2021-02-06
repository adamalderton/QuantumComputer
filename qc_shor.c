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

#define H_NZMAX_ESTIMATE(N) N

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

const int SIZE_OF_INT = sizeof(int);


/***********************************FUNCTION PROTOTYPES**********************************/


/****************************************************************************************/

static ErrorCode build_hadamard_matrix(gsl_spmatrix_complex *hadamard, int qubit_num)
{
    int not_xor_ij;
    gsl_complex element;

    bool element_non_zero;
    int element_set_success;
    

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
            for (int b = 0; b < SIZE_OF_INT; b++) {
    
                /* 
                 * If at least one bit in not_xor_ij is 0, the (i,j)'th element in the matrix will be 0.
                 * All elements of hadamard matrix passed are already initialised to 0, so break 'b' loop.
                 */

                if (GET_BIT(not_xor_ij, b) == 0) {
                    element_non_zero = false;
                    break;
                }
            }

            if (element_non_zero) {

                /* 
                 * If all the Kronecker deltas in this term are 1, what's left is to actually find the 
                 * H value. To do this, extract the qubit_num'th digits in the binary representations
                 * of i and j. These values, each 0 or 1, are the indices of the value to extract from
                 * the 2x2 Hadamard materix, HADAMARD_2x2.
                 */

                GSL_SET_REAL(&element, GET_BIT(i, qubit_num));
                GSL_SET_IMAG(&element, GET_BIT(j, qubit_num));

                element_set_success = gsl_spmatrix_complex_set(hadamard, i, j, element);

                if (element_set_success != GSL_SUCCESS) {
                    return GSL_ERROR;
                }

            }
        }
    }

    /* Convert matrix to CSC format for efficient calculations in future. */

    return NO_ERROR;
}



static ErrorCode temp()
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

    error = build_hadamard_matrix(hadamard, 2);
    ERROR_CHECK(error);

    gsl_spmatrix_complex_free(hadamard);
    gsl_vector_complex_free(state);
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

    temp();



    return NO_ERROR;
}