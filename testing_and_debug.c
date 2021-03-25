/*
    These are functions used within development for debugging and testing the program.
    These are here to be used in future development or debugging.
    These are not linked to by qc_shor.c so should be inserted and altered by hand.
*/

static void display_state(Register reg)
{
    double prob;

    for (int i = 0; i < reg.num_states; i++) {

        prob = gsl_complex_abs(gsl_vector_complex_get(*reg.current_state, i));

        if (prob != 0.0) {

            printf("|");
            for (int b = reg.num_qubits - 1; b >= 0; b--) {
                printf("%d", GET_BIT(i, b));
            }
            printf("> ");

            printf("%.2f\n", prob);
        }
    }
}

static void check_normalisation(Register reg)
{
    double sum_of_sq = 0.0;

    for (unsigned int i = 0; i < reg.num_states; i++) {
        sum_of_sq += gsl_complex_abs2(gsl_vector_complex_get(*reg.current_state, i));
    }

    printf("Total Probability: %.16f\n", sum_of_sq);
}
