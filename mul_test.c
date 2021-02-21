#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_spblas.h>

static void display_vec(gsl_vector_int *vec)
{
    for (int i = 0; i < 8; i++) {
        printf("%d\n", gsl_vector_int_get(vec, i));
    }
}

static void operate_matrix(gsl_spmatrix_int *matrix, gsl_vector_int *vec, gsl_vector_int *result)
{
    int *Ap;
    int *Ai;
    int *Ad;
    int *res;
    int *vect;

    Ap = matrix->p;
    Ai = matrix->i;
    Ad = matrix->data;

    res = result->data;
    vect = vec->data;

    for (int j = 0; j < 8; j++) {
        for (int p = Ap[j]; p < Ap[j + 1]; p++) {
            res[Ai[p]] += Ad[p] * vect[j];
        }
    }
}


int main()  {

    gsl_vector_int *vec;
    gsl_vector_int *result;
    gsl_spmatrix_int *comp_matrix;
    gsl_spmatrix_int *result_matrix;


    vec = gsl_vector_int_alloc(8);
    result = gsl_vector_int_calloc(8);
    comp_matrix = gsl_spmatrix_int_alloc(8, 8);
    result_matrix = gsl_spmatrix_int_alloc_nzmax(8, 8, 8, GSL_SPMATRIX_CSC);

    for (int i = 0; i < 8; i++) {
        gsl_vector_int_set(vec, i, i);
    }

    /* Set matrix */
    gsl_spmatrix_int_set_zero(comp_matrix);
    for (int i = 0; i < 4; i++) {
        gsl_spmatrix_int_set(comp_matrix, i, i, -1);
    }
    for (int i = 4; i < 8; i++) {
        gsl_spmatrix_int_set(comp_matrix, i, i, 0);
    }

    gsl_spmatrix_int_csc(result_matrix, comp_matrix);

    display_vec(vec);
    printf("----\n");

    operate_matrix(result_matrix, vec, result);

    display_vec(result);


    gsl_vector_int_free(vec);
    gsl_vector_int_free(result);
    gsl_spmatrix_int_free(comp_matrix);
    gsl_spmatrix_int_free(result_matrix);

    return 0;
}