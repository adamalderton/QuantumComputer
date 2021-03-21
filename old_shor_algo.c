static ErrorCode shors_algorithm(int factors[2], int C, Register reg, Assets *assets, gsl_rng *rng, bool force_quantum, int forced_trial_int)
{
    ErrorCode error;
    int period;
    int gcd;

    /* Use classical means to analyse small trial integers, unless forced not to. */
    if (!force_quantum) {
        for (int i = 2; i < SMALL_POWER_TOLERANCE; i++) {
            if (is_power(i, C)) {
                factors[0] = i;
                factors[1] = C / i;

                if (verbose) {
                    printf(" --- C = %d is small power of %d, hence factors were found classically.\n", C, i);
                }

                return NO_ERROR;
            }
        }
    }

    /********** If a trial integer is forced, only attempt to find a period with this integer. *********/
    if (forced_trial_int != 0) {
        if (!force_quantum) {
            gcd = greatest_common_divisor(forced_trial_int, C);

            if (gcd > 1) {
                factors[0] = gcd;
                factors[1] = C / gcd;
                
                if (verbose) {
                    printf(" --- Greatest common divisor between C = %d and a = %d is %d, hence factors were found classically.\n", C, forced_trial_int, gcd);
                }

                return NO_ERROR;
            }
        }

        if (verbose) {
            printf(" --- Forced trial integer a = %d, finding factors quantum mechanically...\n", forced_trial_int);
        }

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
    

    /********** If trial integer is not forced by user, iterate over possible trial integers until period is found. **********/
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
        
        break;
    }

    return NO_ERROR;
}
