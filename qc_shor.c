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
#define COMPLEX_ELEMENT -127

/* Tweakable parameters. */
#define NUM_CONTINUED_FRACTIONS 15
#define TRIALS_PER_DENOMINATOR 10

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

#define ALLOC_CHECK(alloc_pointer) \
    if (alloc_pointer == NULL) { \
        fprintf(stderr, "Error: Insufficient memory.\n"); \
    }

/* 
   Gets the binary 1 or 0 stored in the n'th bit of integer (from the right) 
   by bit masking followed by a bitwise and. That is, 00000001 is 
   leftshifted n places to address the n'th bit (from the right) in integer.
   The value resulting from the bitwise and will result in either 0,
   or a finite power of two, corresponding to the nth bit. That is, 2^n.
   Therefore, the result is rightshfted n places to yield either 1 or 0.
 */
#define GET_BIT(integer, n) \
    ( (integer & (1 << n)) >> n )

#define INT_POW(base, power) \
    ( (unsigned int) (pow(base, power) + 0.5) )

#define LONG_INT_POW(base_ptr, power) \
    for (unsigned int i = 0; i < power; i++) { \
        (*base_ptr) * (*base_ptr) \
    }

/***********************************TYPEDEFS AND GLOBALS*********************************/

/* Enum to store the various error codes than can be returned within this program. */
typedef enum {
    NO_ERROR = 0,
    INSUFFICIENT_MEMORY,
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
    gsl_spmatrix_char *result_matrix;
    gsl_spmatrix_char *comp_matrix;
} Assets;

typedef struct {
    unsigned int L_size;
    unsigned int M_size;
    unsigned int num_qubits;
    unsigned long int num_states;
} Register;

/* 
   Does NOT contain the necessary scale of 1/sqrt(2),
   such that it can be stored as an integer matrix.
   This scalar factor is implemented later in appropriate functions.
 */
const char HADAMARD_BASE_MATRIX[2][2] = {
    {1, 1},
    {1, -1}
};

const char C_PHASE_SHIFT_BASE_MATRIX[4][4] = {
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, COMPLEX_ELEMENT}
};

bool verbose = false;
bool very_verbose = false;

/********** UTILITY FUNCTIONS **********/

static void compress_comp_matrix(Assets *assets)
{
    gsl_spmatrix_char_csc(assets->result_matrix, assets->comp_matrix);

    /* Reset comp_matrix straight away as to reduce memory bloat. */
    gsl_spmatrix_char_set_zero(assets->comp_matrix);
}

static void swap_states(Assets *assets)
{
    gsl_vector_complex **temp;

    temp = assets->new_state;
    assets->new_state = assets->current_state;
    assets->current_state = temp;
}

static unsigned long int measure_state(Register reg, Assets *assets, gsl_rng *rng)
{
    double r;
    double cumulative_prob;
    unsigned long int state_num;
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

static void issue_warnings(unsigned int C, Register reg)
{
    if (C % 2 == 0) {
        printf(" --- *WARNING* Number to factorise C = %d is even, results may be unreliable.\n", C);
    }

    /* To find valid factors, 2^M must be greater than or equal to C. */
    if (INT_POW(2, reg.M_size) < C) {
        printf(" --- *WARNING* The M register is not large enough for reliable results. Ensure 2^M >= C. Minimum: M = %d.\n", ((int) (log2(C) + 0.5)) + 1);
    }

    /* To be confident of finding the period, 2^L should be greater than or equal to C. */
    if (INT_POW(2, reg.L_size) < C*C) {
        printf(" --- *WARNING* The L register is not large enough for full confidence in finding the period. Ensure 2^L >= C^2 for confidence. Suggested: L = %d.\n", (int) (log2(C*C) + 0.5));
    }
}

/********** QUANTUM GATE FUNCTIONS **********/

static void operate_matrix(Register reg, Assets *assets, double scale, gsl_complex alt_element)
{
    gsl_vector_complex *n_state;
    gsl_vector_complex *c_state;
    gsl_spmatrix_char *mat;
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

    for (unsigned long int j = 0; j < reg.num_states; j++) {
        for (unsigned long int p = mat->p[j]; p < mat->p[j + 1]; p++) {

            /* Retrieve matrix element. */
            m_real = mat->data[p];
            m_imag = 0.0;

            if ((int) m_real == COMPLEX_ELEMENT) {
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

static void hadamard_gate(unsigned int qubit_num, Register reg, Assets *assets)
{
    unsigned long int not_xor_ij;
    char element;
    bool dirac_deltas_non_zero;

    for (unsigned long int i = 0; i < reg.num_states; i++) {
        for (unsigned long int j = 0; j < reg.num_states; j++) {
            dirac_deltas_non_zero = true;

            not_xor_ij = ~(i ^ j);

            for (unsigned int b = 0; b < reg.num_qubits; b++) {

                if (b != qubit_num) {
                    if (GET_BIT(not_xor_ij, b) == 0) {
                        dirac_deltas_non_zero = false;
                        break;
                    }
                }
            }

            if (dirac_deltas_non_zero) {
                element = HADAMARD_BASE_MATRIX[GET_BIT(i, qubit_num)][GET_BIT(j, qubit_num)];

                gsl_spmatrix_char_set(assets->comp_matrix, i, j, element);
            }
        }
    }

    compress_comp_matrix(assets);

    operate_matrix(reg, assets, HADAMARD_SCALE, NULL_ALT_ELEMENT);
}

static void c_phase_shift_gate(unsigned int c_qubit_num, unsigned int qubit_num, double theta, Register reg, Assets *assets)
{
    unsigned long int not_xor_ij;
    char element;
    bool dirac_deltas_non_zero;

    for (unsigned long int i = 0; i < reg.num_states; i++) {
        for (unsigned long int j = 0; j < reg.num_states; j++) {
            dirac_deltas_non_zero = true;

            not_xor_ij = ~(i ^ j);

            for (unsigned int b = 0; b < reg.num_qubits; b++) {

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

                gsl_spmatrix_char_set(assets->comp_matrix, i, j, element);
            }
        }
    }

    compress_comp_matrix(assets);

    operate_matrix(reg, assets, 1.0, gsl_complex_polar(1.0, theta));
}

static void c_amodc_gate(unsigned int C, unsigned long long int atox, unsigned int c_qubit_num, Register reg, Assets *assets)
{
    unsigned int A; /* Holds a^x (mod C). */
    unsigned int j; /* The column of the matrix in row k in which the 1 resides. */
    unsigned int f; /* Used in the calculation of the permutation matrix. */

    /* Compiler automatically casts A, atox and C to yield the correct A. */
    A = atox % C;

    /* Using notation from instruction document, loop over rows (k) of matrix. */
    for (unsigned long int k = 0; k < reg.num_states; k++) {

        /* If l_0 (c_qubit_num) is 0, j = k. */
        if (GET_BIT(k, c_qubit_num) == 0) {
            gsl_spmatrix_char_set(assets->comp_matrix, k, k, 1);
            continue;
        
        /* If l_0 = 1 ... */
        } else {

            /* f must be calculated, which is the decimal value stored in the M register. */
            f = 0;
            for (unsigned int b = 0; b < reg.M_size; b++) {
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

                gsl_spmatrix_char_set(assets->comp_matrix, k, k, 1);
                continue;

            } else {

                /* Calculate f', which can be stored in f for simplicity. */
                f = (A * f) % C;

                /* Next, build up the integer value j using f'. */
                j = 0;

                /* M register (concerning f'). */
                for (unsigned int b = 0; b < reg.M_size; b++) {
                    j += GET_BIT(f, b) << b;
                }

                /* L register (concerning k). */
                for (unsigned int b = reg.M_size; b < reg.num_qubits; b++) {
                    j += GET_BIT(k, b) << b;
                }

                gsl_spmatrix_char_set(assets->comp_matrix, j, k, 1);
            }
        }
    }

    compress_comp_matrix(assets);

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

static void quantum_computation(unsigned int C, unsigned int a, Register reg, Assets *assets)
{
    unsigned int x = 1;

    /* Apply Hadamard gate to qubits in the L register. */
    if (very_verbose) {
        printf("         - Applying Hadamard matrices.\n");
    }
    for (unsigned int l = (reg.num_qubits - reg.L_size); l < reg.num_qubits; l++) {
        hadamard_gate(l, reg, assets);
    }

    if (very_verbose) {
        printf("         - Applying a^x mod (C) gates.\n");
    }
    /* For each bit value in the L register, apply the conditional a^x (mod C) gate. */
    for (unsigned int l = (reg.num_qubits - reg.L_size); l < reg.num_qubits; l++) {
        c_amodc_gate(C, INT_POW(a, x), l, reg, assets);
        x *= 2;
    }

    if (very_verbose) {
        printf("         - Performing inverse quantum Fourier transform.\n");
    }
    inverse_QFT(reg, assets);
}

/********** SHOR'S ALGORITHM FUNCTIONS **********/

static unsigned int greatest_common_divisor(unsigned int a, unsigned int b)
{
    unsigned int temp;

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

static void get_continued_fractions_denominators(double omega, unsigned int num_fractions, unsigned int *denominators)
{
    double omega_inv;
    unsigned int numerator;
    unsigned int denominator;
    unsigned int temp;
    unsigned int *coeffs;

    coeffs = (unsigned int *) malloc(num_fractions * sizeof(unsigned int));
    ALLOC_CHECK(coeffs);

    for (unsigned int i = 0; i < num_fractions; i++) {
        omega_inv = 1.0 / omega;

        /* Omega for next loop iteration, which is the fractional part of omega_inv. */
        omega = omega_inv - (double) ( (unsigned int) omega_inv );

        /* Coefficient calculation uses next omega value. */
        coeffs[i] = (unsigned int) (omega_inv - omega);

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

static double read_omega(unsigned long int state_num, Register reg)
{
    unsigned int x_tilde;
    unsigned int power;

    power = 0;

    /* Read x_tilde register in reverse order. */
    for (unsigned int i = reg.L_size + reg.M_size - 1; i >= reg.M_size; i--) {
        x_tilde += GET_BIT(state_num, i) << power;
        power++;
    }

    return (double) x_tilde / (double) INT_POW(2, reg.L_size);
}

static int find_period(unsigned int *period, unsigned int C, unsigned int a, Register reg, Assets *assets, gsl_rng *rng)
{
    unsigned int *denominators;
    bool period_found;
    unsigned long int measured_state_num;
    double omega;

    if (very_verbose) {
        printf("      - Performing quantum computation...\n");
    }

    reset_register(*assets->current_state);
    quantum_computation(C, a, reg, assets);

    if (very_verbose) {
        printf("      - Measuring state...\n");
    }

    measured_state_num = measure_state(reg, assets, rng);
    omega = read_omega(measured_state_num, reg);

    if (very_verbose) {
        printf("      - Using continued fractions to guess period...\n");
    }

    denominators = (unsigned int *) malloc(NUM_CONTINUED_FRACTIONS * sizeof(unsigned int));
    ALLOC_CHECK(denominators);

    get_continued_fractions_denominators(omega, NUM_CONTINUED_FRACTIONS, denominators);

    /* With denominators found, trial multiples of them until the period is found. */
    for (unsigned int d = 0; d < NUM_CONTINUED_FRACTIONS; d++) {    /* d => denominator. */
        for (unsigned int m = 1; m < TRIALS_PER_DENOMINATOR + 1; m++) { /* m => multiple. */
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

static ErrorCode shors_algorithm(unsigned int factors[2], unsigned int C, Register reg, Assets *assets, gsl_rng *rng, unsigned int forced_trial_int)
{
    ErrorCode error;
    unsigned int period;

    printf("\n --- Finding factors...\n\n");

    /*
        If a trial integer has been forced by the user, only attempt to find the period
        with that integer, as below.
    */
    if (forced_trial_int != 0) {
        if (verbose) {
            printf(" --- Forced trial integer a = %d, finding period ...\n", forced_trial_int);
        }

        error = find_period(&period, C, forced_trial_int, reg, assets, rng);
        if (error == PERIOD_NOT_FOUND) {
            printf(" --- A valid period was not found and hence C = %d could not be factorised.\n", C);
            return PERIOD_NOT_FOUND;
        }

        if (period % 2 != 0) {
            if (verbose) {
                printf(" --- Period was found to be %d, but it did not pass the validity requirements.\n", period);
            }
            printf(" --- A valid period was not found and hence C = %d could not be factorised.\n", C);
            return PERIOD_NOT_FOUND;

        } else if (INT_POW(forced_trial_int, period / 2) % C == -1) {
            if (verbose) {
                printf(" --- Period was found to be %d, but it did not pass the validity requirements\n", period);
            }
            printf(" --- A valid period was not found and hence C = %d could not be factorised.\n", C);
            return PERIOD_NOT_FOUND;
        }

        if (verbose) {
            printf(" --- A valid period = %d has been found so the factors of C = %d have been found quantum mechanically.\n\n", period, C);
        }

        factors[0] = greatest_common_divisor(INT_POW(forced_trial_int, period / 2) + 1, C);
        factors[1] = greatest_common_divisor(INT_POW(forced_trial_int, period / 2) - 1, C);

        if (factors[0] == 1 || factors[1] == 1) {
            printf(" --- The factors found are trivial, consider trying a different trial integer.\n");
        }

        return NO_ERROR;
    }

    /*
        If a trial integer has not been specified by the user, loop over valid integers
        1 < a < C until a valid period hence factors are found.
    */
    for (unsigned int trial_int = 2; trial_int < C - 1; trial_int++) {
        if (verbose) {
            printf(" --- Trial integer a = %d, finding period ...\n", trial_int);
        }

        error = find_period(&period, C, trial_int, reg, assets, rng);
        if (error == PERIOD_NOT_FOUND) {
            if (verbose) {
                printf(" --- A valid period could not be found for a = %d.\n\n", trial_int);
            }
            continue;
        }

        if (period % 2 != 0) {
            if (verbose) {
                printf(" --- Period was found to be %d, but it did not pass the validity requirements.\n\n", period);
            }
            continue;
        } else if (INT_POW(trial_int, period / 2) % C == -1) {
            if (verbose) {
                printf(" --- Period was found to be %d, but it did not pass the validity requirements.\n\n", period);
            }
            continue;
        }

        if (verbose) {
            printf(" --- A valid period = %d has been found so the factors of C = %d have been found quantum mechanically.\n\n", period, C);
        }

        factors[0] = greatest_common_divisor(INT_POW(trial_int, period / 2) + 1, C);
        factors[1] = greatest_common_divisor(INT_POW(trial_int, period / 2) - 1, C);

        if (factors[0] == 1 || factors[1] == 1) {
            printf(" --- Factors found are trivial. Continuing to find non-trivial factors.\n");
            continue;
        }

        return NO_ERROR;
    }

    printf(" --- A valid period was not found and hence C = %d could not be factorised.\n", C);

    return PERIOD_NOT_FOUND;
}

/*********** SETUP FUNCTIONS **********/

static ErrorCode parse_command_line_args(int argc, char *argv[], Register *reg, unsigned int *C, unsigned int *forced_trial_int)
{
    extern char *optarg;
    extern int optind;
    const char *usage = "Usage: ./qc_shor.exe -C num -L L_reg_size -M M_reg_size [-i trial_int] [-v] [-V]\n";
    int arg;

    bool C_flag = false;
    bool L_flag = false;
    bool M_flag = false;

    while ((arg = getopt(argc, argv, "C:L:M:i:vV")) != -1) {
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
            
            case 'V':
                verbose = true;
                very_verbose = true;
                break;
            
            case 'i':
                *forced_trial_int = atoi(optarg);
                break;
            
            case '?':
                /* Invalid option information printed internally by getopt, simply print usage after. */
                printf(usage);
                return BAD_ARGUMENTS;
        }
    }

    if (!C_flag) {
        fprintf(stderr, "Error: Number to be factorised 'C' not given.\n");
        printf(usage);
        return BAD_ARGUMENTS;
    }

    if (!L_flag) {
        fprintf(stderr, "Error: Size of L register not given.\n");
        printf(usage);
        return BAD_ARGUMENTS;
    }

    if (!M_flag) {
        fprintf(stderr, "Error: Size of M register not given.\n");
        printf(usage);
        return BAD_ARGUMENTS;
    }

    if (C <= 0) {
        fprintf(stderr, "Error: Number to be factorised C is invalid.\n");
        printf(usage);
    }

    if (reg->L_size <= 0) {
        fprintf(stderr, "Error: L is invalid.\n");
        printf(usage);
    }

    if (reg->M_size <= 0) {
        fprintf(stderr, "Error: M is invalid.\n");
        printf(usage);
    }

    reg->num_qubits = reg->M_size + reg->L_size;

    /* Find num states by 2^(num_qubits), but not limited by unsigned int data type is an INT_POW(). */
    reg->num_states = 1;
    for (unsigned int pow_ = 0; pow_ < reg->num_qubits; pow_++) {
        reg->num_states *= 2;
    }

    return NO_ERROR;
}

int main(int argc, char *argv[])
{
    Assets assets;
    Register reg;
    ErrorCode error;
    const gsl_rng_type *rng_type;
    gsl_rng *rng;
    unsigned int C;
    unsigned int factors[2];
    unsigned int forced_trial_int = 0;

    /* Setup rng, seeded by integer derived from the current time. */
    rng_type = gsl_rng_mt19937;
    rng = gsl_rng_alloc(rng_type);
    ALLOC_CHECK(rng);

    gsl_rng_set(rng, (unsigned) time(NULL));

    error = parse_command_line_args(argc, argv, &reg, &C, &forced_trial_int);
    ERROR_CHECK(error);

    issue_warnings(C, reg);

    /* 
        Setup contents of assets, including matrices and vector states.
        The most memory consuming objects are allocated here straight away
        such that the program does not run out of memory mid-way through computation.
    */
    assets.state_a = gsl_vector_complex_alloc(reg.num_states);
    ALLOC_CHECK(assets.state_a);
    assets.state_b = gsl_vector_complex_alloc(reg.num_states);
    ALLOC_CHECK(assets.state_b);
    assets.comp_matrix = gsl_spmatrix_char_alloc(reg.num_states, reg.num_states);
    ALLOC_CHECK(assets.comp_matrix);
    assets.result_matrix = gsl_spmatrix_char_alloc_nzmax(reg.num_states, reg.num_states, reg.num_states, GSL_SPMATRIX_CSC);
    ALLOC_CHECK(assets.result_matrix);

    assets.current_state = &assets.state_a;
    assets.new_state = &assets.state_b;

    error = shors_algorithm(factors, C, reg, &assets, rng, forced_trial_int);
    ERROR_CHECK(error);

    /* Free assets and rng generator. */
    gsl_vector_complex_free(assets.state_a);
    gsl_vector_complex_free(assets.state_b);
    gsl_spmatrix_char_free(assets.result_matrix);
    gsl_spmatrix_char_free(assets.comp_matrix);
    gsl_rng_free(rng);

    if (error != PERIOD_NOT_FOUND) {
        fprintf(stdout, " --- Factors of %d found: (%d, %d).\n", C, factors[0], factors[1]);
    }

    return NO_ERROR;
}