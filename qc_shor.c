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
#include <unistd.h>
#include <stdbool.h>
#include <time.h>

#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_spblas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_math.h>

#define VERSION "1.0.0"
#define REVISION_DATE "04/02/2021"

#define LINE_BUFFER_LENGTH 1024     /* Large buffer to read in lines of files. */
#define MAX_FILENAME_LENGTH 128

#define HADAMARD_SCALE 1.0/M_SQRT2
#define NULL_ALT_ELEMENT gsl_complex_rect(0.0, 0.0)
#define NO_CONDITIONAL_QUBIT -1
#define NON_INT_ELEMENT -INT_MAX

/* Tweakable parameters. */
#define SMALL_POWER_TOLERANCE 5
#define NUM_CONTINUED_FRACTIONS 7
#define TRIALS_PER_DENOMINATOR 5


#define L 0
#define M 1

/* 
   Many functions below return an errorcode. This macro is called after these functions return
   and checks for an error. If so, this ends the function in which the error
   occured returns the error code. If the error raising function is called within a function,
   that calling function is also returned with the same error code. This repeats until the scope of the main
   function is reached. Therefore, this macro passes the error code up the stack until it is eventually returned by main.
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
   Gets the binary 1 or 0 stored in the n'th bit of int (from the right) 
   by bit masking followed by a bitwise and. That is, 00000001 is 
   leftshifted n places to address the n'th bit (from the right) in int.
   The value resulting from the bitwise and will result in either 0,
   or a finite power of two, corresponding to the nth bit. That is, 2^n.
   Therefore, the result is rightshfted n places to yield either 1 or 0.
 */
#define GET_BIT(int, n) \
    ( (int & (1 << n)) >> n )

#define INT_POW(base, power) \
    ( (int) (pow(base, power) + 0.5) )

/***********************************TYPEDEFS AND GLOBALS*********************************/

/* Enum to store the various error codes than can be returned within this program. */
typedef enum {
    NO_ERROR = 0,
    BAD_ARGUMENTS,
    PERIOD_NOT_FOUND,
    UNKNOWN_ERROR,
} ErrorCode;

typedef struct {
    int register_size[2];
    gsl_vector_complex **current_state;
    gsl_vector_complex **new_state;
    gsl_vector_complex *state_a;
    gsl_vector_complex *state_b;
    gsl_spmatrix_int *result_matrix;
    gsl_spmatrix_int *comp_matrix;
} Assets;

typedef struct {
    int L_size;
    int M_size;
    int num_qubits;
    int num_states;
} Register;

/* Global Variables */

/* 
   Does NOT contain the necessary scale of 1/sqrt(2),
   such that it can be stored as an integer matrix.
   This scalar factor is implemented later in appropriate functions.
 */
const int HADAMARD_BASE_MATRIX[2][2] = {
    {1, 1},
    {1, -1}
};

const int C_PHASE_SHIFT_BASE_MATRIX[4][4] = {
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, NON_INT_ELEMENT}
};

bool verbose = false;

/********** UTILITY FUNCTIONS **********/
int hi = {1
// static void display_state(Assets *assets)
// {
//     gsl_vector_complex *state;
//     double prob;
//     int x;
//     int fx;

//     state = *assets->current_state;

//     for (int i = 0; i < num_states; i++) {

//         prob = gsl_complex_abs(gsl_vector_complex_get(state, i));
//         x = 0;
//         fx = 0;

//         if (prob != 0.0) {

//             printf("|");
//             for (int b = num_qubits - 1; b >= 0; b--) {
//                 printf("%d", GET_BIT(i, b));
//             }
//             printf("> ");

//             printf("%.2f", prob);

//             x += GET_BIT(i, 6) << 2;
//             x += GET_BIT(i, 5) << 1;
//             x += GET_BIT(i, 4) << 0;

//             fx += GET_BIT(i, 0) << 0;
//             fx += GET_BIT(i, 1) << 1;
//             fx += GET_BIT(i, 2) << 2;
//             fx += GET_BIT(i, 3) << 3;

//             printf("x = %d, f(x) = %d", x, fx);
//         }
//     }
// }

// static void check_norm(gsl_vector_complex *state)
// {
//     double sum = 0.0;

//     for (int i = 0; i < num_states; i++) {
//         sum += gsl_complex_abs2(gsl_vector_complex_get(state, i));
//     }

//     printf("%.4f\n", sum);
// }

};


static void swap_states(Assets *assets)
{
    gsl_vector_complex **temp;

    temp = assets->new_state;
    assets->new_state = assets->current_state;
    assets->current_state = temp;

    /* Reset comp_matrix. */
    gsl_spmatrix_int_set_zero(assets->comp_matrix);
}

static int measure_state(Register reg, Assets *assets, gsl_rng *rng)
{
    double r;
    double cumulative_prob;
    int state_num;
    gsl_vector_complex *current_state; /* To prevent repeated pointer dereference. */

    current_state = *assets->current_state;
    cumulative_prob = 0.0;
    r = gsl_rng_uniform(rng);

    for (state_num = 0; state_num < (reg.num_states - 1); state_num++) {

        /* Add the square of the absolute value of the state coefficient, as per the basics of quantum mechanics. */
        cumulative_prob += gsl_complex_abs2(gsl_vector_complex_get(current_state, state_num));

        /* The collapsed state is found, the number of which is stored in state_num. */
        if (cumulative_prob >= r) {
            break;
        }

        /* if r = 1.0, this is handled automatically by the remaining probability. */
    }

    /* 
       Now, set new state to be the collapsed state.
       That is, set the state_num'th state to have a probability of 1.
     */
    gsl_vector_complex_set_zero(*assets->current_state);
    gsl_vector_complex_set(*assets->current_state, state_num, gsl_complex_rect(1.0, 0.0));

    return state_num;
}

static void reset_register(gsl_vector_complex *current_state)
{
    gsl_vector_complex_set_zero(current_state);

    /* Sets register to |000 ... 001>. */
    gsl_vector_complex_set(current_state, 1, gsl_complex_polar(1.0, 0.0));
}

/********** QUANTUM GATE FUNCTIONS **********/

static void operate_matrix(Register reg, Assets *assets, double scale, gsl_complex alt_element)
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

    for (int j = 0; j < reg.num_states; j++) {
        for (int p = mat->p[j]; p < mat->p[j + 1]; p++) {

            /* Retrieve matrix element. */
            m_real = mat->data[p];
            m_imag = 0.0;

            if ((int) m_real == NON_INT_ELEMENT) {
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

    swap_states(assets);
}

static void hadamard_gate(int qubit_num, Register reg, Assets *assets)
{
    int not_xor_ij;
    int element;
    bool dirac_deltas_non_zero;

    for (int i = 0; i < reg.num_states; i++) {
        for (int j = 0; j < reg.num_states; j++) {
            dirac_deltas_non_zero = true;

            not_xor_ij = ~(i ^ j);

            for (int b = 0; b < reg.num_qubits; b++) {

                if (b != qubit_num) {
                    if (GET_BIT(not_xor_ij, b) == 0) {
                        dirac_deltas_non_zero = false;
                        break;
                    }
                }
            }

            if (dirac_deltas_non_zero) {
                element = HADAMARD_BASE_MATRIX[GET_BIT(i, qubit_num)][GET_BIT(j, qubit_num)];

                gsl_spmatrix_int_set(assets->comp_matrix, i, j, element);
            }
        }
    }

    gsl_spmatrix_int_csc(assets->result_matrix, assets->comp_matrix);

    operate_matrix(reg, assets, HADAMARD_SCALE, NULL_ALT_ELEMENT);
}

static void c_phase_shift_gate(int c_qubit_num, int qubit_num, double theta, Register reg, Assets *assets)
{
    int not_xor_ij;
    int element;
    bool dirac_deltas_non_zero;

    for (int i = 0; i < reg.num_states; i++) {
        for (int j = 0; j < reg.num_states; j++) {
            dirac_deltas_non_zero = true;

            not_xor_ij = ~(i ^ j);

            for (int b = 0; b < reg.num_qubits; b++) {

                if ( (b != qubit_num) && (b != c_qubit_num) ) {
                    if (GET_BIT(not_xor_ij, b) == 0) {
                        dirac_deltas_non_zero = false;
                        break;
                    }
                }
            }

            if (dirac_deltas_non_zero) {

                element = C_PHASE_SHIFT_BASE_MATRIX
                    [(2*GET_BIT(i, c_qubit_num)) + GET_BIT(i, qubit_num)]
                    [(2*GET_BIT(j, c_qubit_num)) + GET_BIT(j, qubit_num)];

                gsl_spmatrix_int_set(assets->comp_matrix, i, j, element);
            }
        }
    }

    gsl_spmatrix_int_csc(assets->result_matrix, assets->comp_matrix);

    operate_matrix(reg, assets, 1.0, gsl_complex_polar(1.0, theta));
}

static void c_amodc_gate(int C, int atox, int c_qubit_num, Register reg, Assets *assets)
{
    int A; /* Holds a^x (mod C). */
    int M_size;
    int j; /* The column of the matrix in row k in which the 1 resides. */
    int f; /* Used in the calculation of the permutation matrix. */

    A = atox % C;
    M_size = assets->register_size[M];

    /* Using notation from instruction document, loop over rows (k) of matrix. */
    for (int k = 0; k < reg.num_states; k++) {

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
            if (f >= C) {

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
                for (int b = M_size; b < reg.num_qubits; b++) {
                    j += GET_BIT(k, b) << b;
                }

                gsl_spmatrix_int_set(assets->comp_matrix, j, k, 1);
            }
        }
    }

    /* Finally, compress the matrix stored in comp_matrix into result_matrix. */
    gsl_spmatrix_int_csc(assets->result_matrix, assets->comp_matrix);

    operate_matrix(reg, assets, 1.0, NULL_ALT_ELEMENT);
}

static void inverse_QFT(Register reg, Assets *assets)
{
    double theta;

    for (int l = reg.L_size - 1; l >= 0; l--) {
        hadamard_gate(l, reg, assets);

        for (int k = l - 1; k >= 0; k--) {
            theta = M_PI / (double) INT_POW(2, reg.L_size - k - 1);
            c_phase_shift_gate(l, k, theta, reg, assets);
        }
    }
}

static void quantum_computation(int C, int a, Register reg, Assets *assets)
{
    int x;

    /* Apply Hadamard gate to qubits in the L register. */
    if (verbose) {
        printf("            - Applying Hadamard gates.\n");
    }
    for (int l = (reg.num_qubits - reg.L_size); l < reg.num_qubits; l++) {
        hadamard_gate(l, reg, assets);
    }

    /* For each bit value in the L register, apply the conditional a^x (mod C) gate. */
    if (verbose) {
        printf("            - Applying f(x) gates.\n");
    }
    x = 1;
    for (int l = (reg.num_qubits - reg.L_size); l < reg.num_qubits; l++) {
        c_amodc_gate(C, INT_POW(a, x), l, reg, assets);
        x *= 2;
    }

    if (verbose) {
        printf("            - Performing inverse quantum Fourier transform.\n");
    }
    inverse_QFT(reg, assets);
}

/********** SHOR'S ALGORITHM FUNCTIONS **********/

static void get_continued_fractions_denominators(double omega, int num_fractions, int *denominators)
{
    double omega_inv;
    int numerator;
    int denominator;
    int temp;
    int *coeffs;

    coeffs = (int *) malloc(num_fractions * sizeof(int));
    // CHECK_ALLOC();

    for (int i = 0; i < num_fractions; i++) {
        omega_inv = 1.0 / omega;

        /* Omega for next loop iteration, which is the fractional part of omega_inv. */
        omega = omega_inv - (double) ( (int) omega_inv );

        /* Coefficient calculation uses next omega value. */
        coeffs[i] = (int) (omega_inv - omega);

        /*
         * With the coefficient array built, use it (in reverse order) to build the
         * numerator and denominator of the continued fraction approximation.
         */
        denominator = 1;
        numerator = 0;
        for (int coeff_num = i - 1; coeff_num >= 0; coeff_num--) {
            temp = denominator;
            denominator = numerator + (denominator * coeffs[coeff_num]);
            numerator = temp;
        }

        denominators[i] = denominator;
    }

    free(coeffs);
}

static double read_omega(int state_num, Register reg)
{
    int x_tilde;
    int power;

    power = 0;

    /* Read x_tilde register in reverse order. */
    for (int i = reg.L_size + reg.M_size - 1; i >= reg.M_size; i--) {
        x_tilde += GET_BIT(state_num, i) << power;
        power++;
    }

    return (double) x_tilde / (double) INT_POW(2, reg.L_size);
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

static int small_power_checks(int C, int max_small_power)
{
    for (int i = 2; i < max_small_power; i++) {
        if (C % i == 0) {
            return i;
        }
    }
    return -1;
}

static int find_period(int *period, int C, int a, Register reg, Assets *assets, gsl_rng *rng)
{
    int *denominators;
    bool period_found;
    int measured_state_num;
    double omega;

    if (verbose) {
        printf("        - Performing quantum computation...\n");
    }
    reset_register(*assets->current_state);
    quantum_computation(C, a, reg, assets);

    if (verbose) {
        printf("        - Measuring state.\n");
    }
    measured_state_num = measure_state(reg, assets, rng);
    omega = read_omega(measured_state_num, reg);

    if (verbose) {
        printf("        - Finding period with continued fractions.\n");
    }
    denominators = (int *) malloc(NUM_CONTINUED_FRACTIONS * sizeof(int));
    get_continued_fractions_denominators(omega, NUM_CONTINUED_FRACTIONS, denominators);

    /* With denominators found, trial multiples of them until the period is found. */
    for (int d = 0; d < NUM_CONTINUED_FRACTIONS; d++) {    /* d => denominator. */
        for (int m = 1; m < TRIALS_PER_DENOMINATOR + 1; m++) { /* m => multiple. */
            *period = m * denominators[d];

            if (INT_POW(a, *period) % C == 1 ) {
                period_found = true;
                break;
            }
        }

        if (period_found) {
            break;
        }
    }

    free(denominators);

    if (!period_found) {
        return PERIOD_NOT_FOUND;
    }

    return NO_ERROR;
}

static ErrorCode shors_algorithm(int factors[2], int C, Register reg, Assets *assets, gsl_rng *rng, bool force_quantum, int forced_trial_int)
{
    ErrorCode error;
    int period;
    int small_power_factor;
    int gcd;

    /* 
        If a trial integer is forced by the user and quantum mechanical means
        is also forced, the period only needs to be found wwhen considering that trial integer.
    */
    if (force_quantum && (forced_trial_int != 0)) {
        error = find_period(&period, C, forced_trial_int, reg, assets, rng);
        ERROR_CHECK(error);

        if (period % 2 != 0) {
            if (verbose) {
                printf(" --- Period was found to be %d, but it did not pass the validity requirements.\n", period);
            }
            return PERIOD_NOT_FOUND;

        } else if (INT_POW(forced_trial_int, period / 2) % C == -1) {
            if (verbose) {
                printf(" --- Period was found to be %d, but it did not pass the validity requirements\n", period);
            }
            return PERIOD_NOT_FOUND;
        }

        if (verbose) {
            printf(" --- A valid period = %d has been found so the factors of C = %d have been found quantum mechanically.\n", period, C);
        }

        factors[0] = greatest_common_divisor(INT_POW(forced_trial_int, period / 2) + 1, C);
        factors[1] = greatest_common_divisor(INT_POW(forced_trial_int, period / 2) - 1, C);

        return NO_ERROR;
    }

    /* 
        If quantum mechanical factorisation is forced but a trial integer is 
    */
    else if (force_quantum && (forced_trial_int == 0)) {
        for (int trial_int = SMALL_POWER_TOLERANCE; trial_int < C; trial_int++) {

            error = find_period(&period, C, forced_trial_int, reg, assets, rng);
            ERROR_CHECK(error);

            if (period % 2 != 0) {
                if (verbose) {
                    printf(" --- Period was found to be %d, but it did not pass the validity requirements.\n", period);
                }
                continue;

            } else if (INT_POW(forced_trial_int, period / 2) % C == -1) {
                if (verbose) {
                    printf(" --- Period was found to be %d, but it did not pass the validity requirements\n", period);
                }
                continue;
            }

            if (verbose) {
                printf(" --- A valid period = %d has been found so the factors of C = %d have been found quantum mechanically.\n", period, C);
            }

            factors[0] = greatest_common_divisor(INT_POW(forced_trial_int, period / 2) + 1, C);
            factors[1] = greatest_common_divisor(INT_POW(forced_trial_int, period / 2) - 1, C);

            return NO_ERROR;
        }
    }

    /* 
       If finding factors quantum mechanically is not forced,
       classical methods can be used as below, un
    */
    else {
        small_power_factor = small_power_checks(C, SMALL_POWER_TOLERANCE);

        if (small_power_factor != -1) {
            if (verbose) {
                printf(" --- C = %d is small power of %d, hence factors were found classically.\n", C, small_power_factor);
            }

            factors[0] = small_power_factor;
            factors[1] = C / small_power_factor;

            return NO_ERROR;
        }

        for (int trial_int = SMALL_POWER_TOLERANCE; trial_int < C; trial_int++) {
            gcd = greatest_common_divisor(trial_int, C);

            if (gcd > 1) {
                factors[0] = gcd;
                factors[1] = C / gcd;

                if (verbose) {
                    printf(" --- Greatest common divisor between C = %d and a = %d is %d, hence factors were found classically.\n", C, trial_int, gcd);
                }

                return NO_ERROR;
            }

            if (verbose) {
                printf(" --- Trial integer a = %d has passed classical tests - finding factors quantum mechanically...\n", trial_int);
            }

            error = find_period(&period, C, trial_int, reg, assets, rng);
            ERROR_CHECK(error);

            if (period % 2 != 0) {
                if (verbose) {
                    printf(" --- Period was found to be %d, but it did not pass the validity requirements. Trying another trial integer...\n", period);
                }
                continue;
            } else if (INT_POW(trial_int, period / 2) % C == -1) {
                if (verbose) {
                    printf(" --- Period was found to be %d, but it did not pass the validity requirements. Trying another trial integer...\n", period);
                }
                continue;
            }

            if (verbose) {
                printf(" --- A valid period = %d has been found so the factors of C = %d have been found quantum mechanically.\n", period, C);
            }

            factors[0] = greatest_common_divisor(INT_POW(trial_int, period / 2) + 1, C);
            factors[1] = greatest_common_divisor(INT_POW(trial_int, period / 2) - 1, C);
            
            return NO_ERROR;
        }
    }

    return PERIOD_NOT_FOUND;
}

/*********** SETUP FUNCTIONS **********/

static ErrorCode parse_command_line_args(int argc, char *argv[], Register *reg, int *C, bool *force_quantum, int *forced_trial_int)
{
    /* Todo:
        - take continued fraction parameters (current #define d)
        - warning if registers are too small to be confident about finding a factor
    */

    extern char *optarg;
    extern int optind;
    const char *usage = "Usage: ./qc_shor.exe -C num -L L_reg_size -M M_reg_size [-i trial_int] [-v] [-q]\n";
    int arg;

    bool C_flag = false;
    bool L_flag = false;
    bool M_flag = false;

    while ((arg = getopt(argc, argv, "C:L:M:i:vq")) != -1) {
        switch(arg) {
            case 'C':
                *C = atoi(optarg);
                C_flag = true;
                break;
            
            case 'L':
                reg->L_size = atoi(optarg);
                L_flag = true;
                break;
            
            case 'M':
                reg->M_size = atoi(optarg);
                M_flag = true;
                break;
            
            case 'v':
                verbose = true;
                break;
            
            case 'q':
                *force_quantum = true;
                break;
            
            case 'i':
                *forced_trial_int = atoi(optarg);
                break;
            
            case '?':
                /* Invalid option information printed internally by getopt, simply print usage after. */
                fprintf(stdout, usage);
                return BAD_ARGUMENTS;
        }
    }

    if (!C_flag) {
        fprintf(stderr, "Error: Number to be factorised 'C' not given.\n");
        fprintf(stdout, usage);
        return BAD_ARGUMENTS;
    }

    if (!L_flag) {
        fprintf(stderr, "Error: Size of L register not given.\n");
        fprintf(stdout, usage);
        return BAD_ARGUMENTS;
    }

    if (!M_flag) {
        fprintf(stderr, "Error: Size of M register not given.\n");
        fprintf(stdout, usage);
        return BAD_ARGUMENTS;
    }

    reg->num_qubits = reg->M_size + reg->L_size;
    reg->num_states = INT_POW(2, reg->num_qubits);

    return NO_ERROR;
}

int main(int argc, char *argv[])
{
    Assets assets;
    Register reg;
    ErrorCode error;
    const gsl_rng_type *rng_type;
    gsl_rng *rng;
    int C;
    int factors[2];
    bool force_quantum = false;
    int forced_trial_int = 0;

    error = parse_command_line_args(argc, argv, &reg, &C, &force_quantum, &forced_trial_int);
    ERROR_CHECK(error);

    /* Setup rng, seeded by integer derived from the current time. */
    rng_type = gsl_rng_mt19937;
    rng = gsl_rng_alloc(rng_type);
    gsl_rng_set(rng, (unsigned) time(NULL));

    /* Setup contents of assets, including matrices and vector states. */
    assets.state_a = gsl_vector_complex_alloc(reg.num_states);
    assets.state_b = gsl_vector_complex_alloc(reg.num_states);
    assets.comp_matrix = gsl_spmatrix_int_alloc(reg.num_states, reg.num_states);
    assets.result_matrix = gsl_spmatrix_int_alloc_nzmax(reg.num_states, reg.num_states, reg.num_states, GSL_SPMATRIX_CSC);
    assets.current_state = &assets.state_a;
    assets.new_state = &assets.state_b;

    error = shors_algorithm(factors, C, reg, &assets, rng, force_quantum, forced_trial_int);
    ERROR_CHECK(error);

    /* Free assets and rng generator. */
    gsl_vector_complex_free(assets.state_a);
    gsl_vector_complex_free(assets.state_b);
    gsl_spmatrix_int_free(assets.result_matrix);
    gsl_spmatrix_int_free(assets.comp_matrix);
    gsl_rng_free(rng);

    fprintf(stdout, " --- Factors of %d: (%d, %d).\n", C, factors[0], factors[1]);

    return NO_ERROR;
}