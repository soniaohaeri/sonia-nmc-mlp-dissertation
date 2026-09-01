# ============================================================
# Nested Monte Carlo simulation
# ============================================================
# Take the Day-10 outer states, run the inner Hull-White
# simulations, price the cap under each outer scenario,
# and return the Day-10 cap values.
# ============================================================

import time
import numpy as np
import pandas as pd

from market_data import (
    a,
    sigma,
    strike,
    notional,
    number_of_outer_paths,
    number_of_inner_paths,
    number_of_caplets
)

from hull_white import (
    outer_paths,
    day_10_rates,
    inner_t,
    inner_forward,
    simulate_inner_hull_white
)

from caplet_pricing import (
    remaining_caplet_boundaries,
    compound_sonia_business_days,
    calculate_caplet_cashflow,
    calculate_discount_factor
)


# ============================================================
# 1. Value the cap at Day 10 for one outer scenario
# ============================================================

def value_cap_at_day_10(
    outer_path: np.ndarray,
    inner_paths: np.ndarray,
    strike: float,
    notional: float
) -> float:

    # --------------------------------------------------------
    # Outer path:
    # 11 observations -> 10 already-realised intervals
    # --------------------------------------------------------

    realised_rates = outer_path[:-1]

    realised_factor = np.prod(
        1.0 + realised_rates / 252.0
    )


    # --------------------------------------------------------
    # Inner paths:
    # remaining future intervals from Day 10 to maturity
    # --------------------------------------------------------

    interval_rates = inner_paths[:-1, :]

    number_of_inner_paths = inner_paths.shape[1]

    pathwise_cap_values = np.zeros(
        number_of_inner_paths
    )


    # --------------------------------------------------------
    # Loop through the four caplets
    # --------------------------------------------------------

    for q in range(number_of_caplets):

        start = remaining_caplet_boundaries[q]
        end = remaining_caplet_boundaries[q + 1]

        caplet_rates = interval_rates[
            start:end,
            :
        ]


        # ----------------------------------------------------
        # Caplet 1:
        # 10 realised business days
        # + 53 simulated business days
        # ----------------------------------------------------

        if q == 0:

            future_factor = np.prod(
                1.0 + caplet_rates / 252.0,
                axis=0
            )

            total_factor = (
                realised_factor
                * future_factor
            )

            accrual_fraction = 63 / 252.0

            compounded_rate = (
                total_factor - 1.0
            ) / accrual_fraction


        # ----------------------------------------------------
        # Caplets 2, 3 and 4:
        # full 63 business days simulated
        # ----------------------------------------------------

        else:

            compounded_rate, accrual_fraction = (
                compound_sonia_business_days(
                    caplet_rates
                )
            )


        # ----------------------------------------------------
        # Calculate caplet cashflow
        # ----------------------------------------------------

        caplet_cashflow = calculate_caplet_cashflow(
            compounded_rate=compounded_rate,
            strike=strike,
            notional=notional,
            accrual_fraction=accrual_fraction
        )


        # ----------------------------------------------------
        # Discount caplet value back to Day 10
        # ----------------------------------------------------

        discount_factor = calculate_discount_factor(
            interval_rates[:end, :]
        )

        discounted_caplet_value = (
            caplet_cashflow
            * discount_factor
        )

        pathwise_cap_values += (
            discounted_caplet_value
        )


    # --------------------------------------------------------
    # Average across inner path values
    # --------------------------------------------------------

    day_10_cap_value = (
        pathwise_cap_values.mean()
    )

    return day_10_cap_value


# ============================================================
# 2. Run the full nested Monte Carlo
# ============================================================

def run_nested_monte_carlo():

    cap_values_day_10 = np.zeros(
        number_of_outer_paths
    )

    start_time = time.perf_counter()

    for outer_index in range(
        number_of_outer_paths
    ):

        # ----------------------------------------------------
        # Complete 10-day outer path
        # ----------------------------------------------------

        outer_path = outer_paths[
            :,
            outer_index
        ]


        # ----------------------------------------------------
        # Generate inner continuations
        # ----------------------------------------------------

        inner_paths = simulate_inner_hull_white(
            r_start=day_10_rates[outer_index],
            a=a,
            sigma=sigma,
            t=inner_t,
            forward=inner_forward,
            number_of_paths=number_of_inner_paths,
            seed=10_000 + outer_index
        )


        # ----------------------------------------------------
        # Price cap at Day 10
        # ----------------------------------------------------

        cap_values_day_10[outer_index] = (
            value_cap_at_day_10(
                outer_path=outer_path,
                inner_paths=inner_paths,
                strike=strike,
                notional=notional
            )
        )

    runtime = (
        time.perf_counter()
        - start_time
    )

    return cap_values_day_10, runtime


# ============================================================
# 3. Create NMC results table
# ============================================================

def create_nmc_results():

    cap_values_day_10, runtime = (
        run_nested_monte_carlo()
    )

    results = pd.DataFrame({
        "Outer Scenario": np.arange(
            1,
            number_of_outer_paths + 1
        ),

        "Day 10 Short Rate": (
            day_10_rates
        ),

        "Day 10 Short Rate (%)": (
            day_10_rates * 100
        ),

        "Day 10 Cap Value (£)": (
            cap_values_day_10
        )
    })

    return results, runtime