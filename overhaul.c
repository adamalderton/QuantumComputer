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

#define HADAMARD_SCALE 1.0/M_SQRT2
#define COMPLEX_ELEMENT -DBL_MAX

#define GET_BIT(integer, n) \
    ( (integer & (1 << n)) >> n )

#define INT_POW(base, power) \
    ( (unsigned int) (pow(base, power) + 0.5) )

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

/* Stored as doubles as to prevent frequent unneccessary casting. */
const double HADAMARD_BASE_MATRIX[2][2] = {
    {M_SQRT1_2, M_SQRT1_2},
    {M_SQRT1_2, -M_SQRT1_2}
};

const double C_PHASE_SHIFT_BASE_MATRIX[4][4] = {
    {1.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0},
    {0.0, 0.0, 0.0, COMPLEX_ELEMENT}
};

const double PHASE_BASE[2][2] = {
    {1.0, 0.0},
    {0.0, COMPLEX_ELEMENT}
};

static void swap_states(Register *reg)
{
    gsl_vector_complex **temp;

    temp = reg->new_state;
    reg->new_state = reg->current_state;
    reg->current_state = temp;

}

static void display_state(gsl_vector_complex *state, Register reg)
{
    double prob;
    int x;
    int fx;

    for (int i = 0; i < reg.num_states; i++) {

        prob = gsl_complex_abs(gsl_vector_complex_get(state, i));
        x = 0;
        fx = 0;

        //if (prob != 0.0) {

            printf("|");
            for (int b = reg.num_qubits - 1; b >= 0; b--) {
                printf("%d", GET_BIT(i, b));
            }
            printf("> ");

            printf("%.2f\n", prob);
        //}
    }
}

static void operate_matrix(gsl_spmatrix_complex *matrix, Register *reg)
{
    //gsl_blas_zgemv(CblasNoTrans, gsl_complex_rect(1.0, 0.0), matrix, *reg.current_state, gsl_complex_rect(0.0, 0.0), *reg.new_state);

    double *current_state_data;
    double *new_state_data;
    double *matrix_data;
    int *element_rows;
    int *element_cols;

    unsigned int row;
    unsigned int col;
    double matrix_real;
    double matrix_imag;
    double current_real;
    double current_imag;

    current_state_data = (*reg->current_state)->data;
    new_state_data = (*reg->new_state)->data;
    matrix_data = matrix->data;

    element_rows = matrix->i;
    element_cols = matrix->p;

    gsl_vector_complex_set_zero(*reg->new_state);

    /* Loop over non-zero (nz) elements. */
    for (unsigned long int nz = 0; nz < (unsigned long int) matrix->nz; nz++) {
        row = element_rows[nz];
        col = element_cols[nz];

        matrix_real = matrix_data[2 * nz];
        matrix_imag = matrix_data[2 * nz + 1];

        current_real = current_state_data[2 * col];
        current_imag = current_state_data[2 * col + 1];

        new_state_data[2 * row] += (matrix_real * current_real) - (matrix_imag * current_imag);
        new_state_data[2 * row + 1] += (matrix_real * current_imag) + (matrix_imag * current_real);
    }

    gsl_spmatrix_complex_set_zero(matrix);

    swap_states(reg);
}

static void c_phase_shift_gate(unsigned int c_qubit_num, unsigned int qubit_num, double theta, gsl_spmatrix_complex *matrix, Register *reg)
{
    /* 
        Holds the result of the bitwise not operation applied to the 
        result of the bitwise xor operation between the matrix indices i and j
    */
    unsigned long int not_xor_ij;
    double base_matrix_element;
    gsl_complex element;                   /* Element of phase shift matrix. */
    bool dirac_deltas_non_zero;     /* Determines whether any of the dirac deltas are zero or not. */

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
                    element = gsl_complex_polar(1.0, theta);
                } else {
                    GSL_SET_COMPLEX(&element, base_matrix_element, 0.0);
                }

                /* Insert element in build_matrix. */
                gsl_spmatrix_complex_set(matrix, i, j, element);
            }
        }
    }

    operate_matrix(matrix, reg);
}

static void phase_gate(unsigned int qubit_num, double theta, gsl_spmatrix_complex *matrix, Register *reg)
{
    /* 
        Holds the result of the bitwise not operation applied to the 
        result of the bitwise xor operation between the matrix indices i and j
    */
    unsigned long int not_xor_ij;
    double base_matrix_element;
    gsl_complex element;                   /* Element of phase shift matrix. */
    bool dirac_deltas_non_zero;     /* Determines whether any of the dirac deltas are zero or not. */

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
                base_matrix_element = PHASE_BASE[GET_BIT(i, qubit_num)][GET_BIT(j, qubit_num)];

                
                if (base_matrix_element == COMPLEX_ELEMENT) {
                    element = gsl_complex_polar(1.0, theta);
                } else {
                    GSL_SET_COMPLEX(&element, base_matrix_element, 0.0);
                }

                /* Insert element in build_matrix. */
                gsl_spmatrix_complex_set(matrix, i, j, element);
            }
        }
    }

    operate_matrix(matrix, reg);
}

static void hadamard_gate(unsigned int qubit_num, gsl_spmatrix_complex *matrix, Register *reg)
{
    /* 
        Holds the result of the bitwise not operation applied to the 
        result of the bitwise xor operation between the matrix indices i and j
    */
    unsigned long int not_xor_ij;
    gsl_complex element;            /* Element of Hadamard matrix. */
    bool dirac_deltas_non_zero;     /* Determines whether any of the dirac deltas are zero or not. */

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

                /* Insert element in build_matrix. */
                gsl_spmatrix_complex_set(matrix, i, j, element);
            }
        }
    }

    operate_matrix(matrix, reg);
}

int main()
{
    Register reg;
    gsl_spmatrix_complex *matrix;

    reg.num_qubits = 3;
    reg.num_states = INT_POW(2, reg.num_qubits);

    reg.state_a = gsl_vector_complex_calloc(reg.num_states);
    reg.state_b = gsl_vector_complex_calloc(reg.num_states);
    reg.current_state = &reg.state_a;
    reg.new_state = &reg.state_b;

    matrix = gsl_spmatrix_complex_alloc_nzmax(reg.num_states, reg.num_states, 2 * reg.num_states, GSL_SPMATRIX_COO);

    /**********************/
    gsl_vector_complex_set_zero(*reg.current_state);
    gsl_vector_complex_set(*reg.current_state, 0, gsl_complex_rect(1.0, 0.0));
    /**********************/

    hadamard_gate(0, matrix, &reg);
    phase_gate(0, M_PI, matrix, &reg);
    hadamard_gate(0, matrix, &reg);

    display_state(*reg.current_state, reg);

    // for (int i = 0; i < reg.num_states; i++) {
    //     for (int j = 0; j < reg.num_states; j++) {
    //         printf("%.1f ", gsl_spmatrix_complex_get(matrix, i, j));
    //     }
    //     printf("\n");
    // }

    /* Cleanup. */
    gsl_vector_complex_free(reg.state_a);
    gsl_vector_complex_free(reg.state_b);
    gsl_spmatrix_complex_free(matrix);

    return 0;
}