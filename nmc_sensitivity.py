# ============================================================
# NMC Label Sensitivity Analysis
# ============================================================
# Revalue the same 100,000 outer scenarios using smaller
# inner simulation allocations.
#
# Purpose:
# Examine how surrogate accuracy and optimiser selection
# change as the accuracy of the NMC training labels changes.
# ============================================================

import os
import time
import numpy as np
import pandas as pd

from market_data import (
    a,
    sigma,
    strike,
    notional
)

from hull_white import (
    inner_t,
    inner_forward,
    simulate_inner_hull_white
)

from nmc_simulation import (
    value_cap_at_day_10
)

from nmc_cache import (
    load_outer_states
)


# ============================================================
# 1. Sensitivity output folder
# ============================================================

SENSITIVITY_DIR = "nmc_sensitivity_all"

os.makedirs(
    SENSITIVITY_DIR,
    exist_ok=True
)

# ============================================================
# 3. Reduced inner allocations
# ============================================================

inner_allocations = [
    500,
    250,
    100
]


# ============================================================
# 4. Run one inner allocation
# ============================================================

def run_nmc_allocation(
    n_inner,
    sensitivity_outer_paths,
    sensitivity_day_10_rates
):

    number_of_outer_paths = (
        sensitivity_outer_paths.shape[1]
    )

    cap_values = np.zeros(
        number_of_outer_paths
    )

    start_time = time.perf_counter()

    for outer_index in range(
        number_of_outer_paths
    ):

        # Complete 10-day outer path
        outer_path = sensitivity_outer_paths[
            :,
            outer_index
        ]

        # Generate inner continuations
        inner_paths = simulate_inner_hull_white(
            r_start=sensitivity_day_10_rates[
                outer_index
            ],
            a=a,
            sigma=sigma,
            t=inner_t,
            forward=inner_forward,
            number_of_paths=n_inner,
            seed=10_000 + outer_index
        )

        # Revalue cap at Day 10
        cap_values[
            outer_index
        ] = value_cap_at_day_10(
            outer_path=outer_path,
            inner_paths=inner_paths,
            strike=strike,
            notional=notional
        )

    runtime = (
        time.perf_counter()
        - start_time
    )

    return (
        cap_values,
        runtime
    )

# ============================================================
# 5. Load or run one allocation
# ============================================================

def load_or_run_allocation(
    n_inner
):

    results_file = os.path.join(
        SENSITIVITY_DIR,
        f"nmc_results_100k_{n_inner}inner.csv"
    )

    runtime_file = os.path.join(
        SENSITIVITY_DIR,
        f"nmc_runtime_100k_{n_inner}inner.txt"
    )


    # --------------------------------------------------------
    # Load previously saved results if available
    # --------------------------------------------------------

    if (
        os.path.exists(results_file)
        and os.path.exists(runtime_file)
    ):

        print(
            f"Existing {n_inner}-inner "
            f"sensitivity results found."
        )

        print(
            "Skipping NMC simulation."
        )

        results = pd.read_csv(
            results_file
        )

        with open(
            runtime_file,
            "r"
        ) as file:

            runtime = float(
                file.read()
            )

        return (
            results,
            runtime
        )


    # --------------------------------------------------------
    # Otherwise run the NMC sensitivity allocation
    # --------------------------------------------------------

        print(
        f"Running 100,000 outer × "
        f"{n_inner} inner..."
    )
    
    
    # Load benchmark outer states only when required
    
    sensitivity_outer_paths, sensitivity_day_10_rates = (
        load_outer_states()
    )
    
    number_of_outer_paths = (
        sensitivity_outer_paths.shape[1]
    )
    
    
    cap_values, runtime = (
        run_nmc_allocation(
            n_inner=n_inner,
            sensitivity_outer_paths=sensitivity_outer_paths,
            sensitivity_day_10_rates=sensitivity_day_10_rates
        )
    )


    # --------------------------------------------------------
    # Create results table
    # --------------------------------------------------------

    results = pd.DataFrame(
        {
            "Outer Scenario":
                np.arange(
                    1,
                    number_of_outer_paths + 1
                ),

            "Day 10 Short Rate":
                sensitivity_day_10_rates,

            "Day 10 Short Rate (%)":
                sensitivity_day_10_rates * 100,

            "Day 10 Cap Value (£)":
                cap_values
        }
    )


    # --------------------------------------------------------
    # Save results and runtime
    # --------------------------------------------------------

    results.to_csv(
        results_file,
        index=False
    )

    with open(
        runtime_file,
        "w"
    ) as file:

        file.write(
            str(runtime)
        )

    print(
        f"Saved {n_inner}-inner "
        f"sensitivity results."
    )

    return (
        results,
        runtime
    )