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

#include <gsl\gsl_spmatrix.h>

#define VERSION "1.0.0"
#define REVISION_DATE "04/02/2021"

#define LINE_BUFFER_LENGTH 1024     /* Large buffer to read in lines of files. */
#define MAX_FILENAME_LENGTH 128

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



/***********************************FUNCTION PROTOTYPES**********************************/


/****************************************************************************************/

/****************************************************************************************
 * parse_command_line_args -- Parse the command line arguments.                         *
 *                                                                                      *
 * Parameters                                                                           *
 *      int argc                                                                        *
 *          As usual, number of command line arguments.                                 *
 *                                                                                      *
 *      char *argv                                                                      *
 *          As usual, array of command line arguments (strings).                        *
 *                                                                                      *
 * Returns                                                                              *
 *      ErrorCode error                                                                 *
 *          Value from the "Errorcode" enum describing what error occurred, if any.     *
 *                                                                                      *
 ****************************************************************************************/
static ErrorCode parse_command_line_args(int argc, char *argv[])
{

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

    



    return NO_ERROR;
}