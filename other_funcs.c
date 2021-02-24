
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

static void build_phase_change_gate(Assets *assets, int c_qubit_num, int qubit_num)
{
    int not_xor_ij;
    int element;
    bool dirac_deltas_non_zero;

    int c_q_address = (num_qubits - 1) - c_qubit_num;
    int q_address = (num_qubits - 1) - qubit_num;

    /* Reset elements in comp_matrix to 0. */
    gsl_spmatrix_int_set_zero(assets->comp_matrix);

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

                element = PHASE_CHANGE_BASE
                [(2*GET_BIT(i, c_q_address)) + GET_BIT(i, q_address)]
                [(2*GET_BIT(j, c_q_address)) + GET_BIT(j, q_address)];

                gsl_spmatrix_set(assets->comp_matrix, i, j, element);
            }
        }
    }

    gsl_spmatrix_csc(assets->result_matrix, assets->comp_matrix);
}
