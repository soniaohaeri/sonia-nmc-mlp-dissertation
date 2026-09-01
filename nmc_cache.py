# ============================================================
# Nested Monte Carlo Result Cache
# ============================================================
# Prevents expensive NMC simulations from being rerun when
# results have already been generated and saved.
#
# The saved outer Hull-White states are also reused in the
# NMC label sensitivity analysis.
# ============================================================

import os
import numpy as np
import pandas as pd

from market_data import (
    number_of_outer_paths,
    number_of_inner_paths
)

from nmc_simulation import (
    create_nmc_results
)

from hull_white import (
    outer_paths,
    day_10_rates
)


# ============================================================
# 1. File names
# ============================================================

outer_label = (
    f"{number_of_outer_paths // 1000}k"
)

NMC_RESULTS_FILE = (
    f"nmc_results_{outer_label}_"
    f"{number_of_inner_paths}inner.csv"
)

NMC_RUNTIME_FILE = (
    f"nmc_runtime_{outer_label}_"
    f"{number_of_inner_paths}inner.txt"
)

OUTER_PATHS_FILE = (
    f"outer_paths_{outer_label}.npy"
)

DAY_10_RATES_FILE = (
    f"day_10_rates_{outer_label}.npy"
)


# ============================================================
# 2. Save NMC results
# ============================================================

def save_nmc_results(
    nmc_results,
    nmc_runtime
):

    # --------------------------------------------------------
    # Save NMC valuation results
    # --------------------------------------------------------

    nmc_results.to_csv(
        NMC_RESULTS_FILE,
        index=False
    )


    # --------------------------------------------------------
    # Save NMC runtime
    # --------------------------------------------------------

    with open(
        NMC_RUNTIME_FILE,
        "w"
    ) as file:

        file.write(
            str(nmc_runtime)
        )


    # --------------------------------------------------------
    # Save common outer Hull-White paths
    # --------------------------------------------------------

    np.save(
        OUTER_PATHS_FILE,
        outer_paths
    )


    # --------------------------------------------------------
    # Save Day-10 short-rate states
    # --------------------------------------------------------

    np.save(
        DAY_10_RATES_FILE,
        day_10_rates
    )

    print(
        "NMC results saved successfully."
    )


# ============================================================
# 3. Load existing NMC results
# ============================================================

def load_nmc_results():

    nmc_results = pd.read_csv(
        NMC_RESULTS_FILE
    )

    if len(nmc_results) != number_of_outer_paths:

        raise ValueError(
            "Saved NMC results do not contain "
            f"{number_of_outer_paths:,} scenarios."
        )

    with open(
        NMC_RUNTIME_FILE,
        "r"
    ) as file:

        nmc_runtime = float(
            file.read()
        )

    print(
        "Configuration: "
        f"{number_of_outer_paths:,} outer × "
        f"{number_of_inner_paths:,} inner"
    )

    print(
        f"Loaded {len(nmc_results):,} "
        f"saved NMC scenarios."
    )

    return (
        nmc_results,
        nmc_runtime
    )


# ============================================================
# 4. Check whether complete cache exists
# ============================================================

def nmc_cache_exists():

    required_files = [
        NMC_RESULTS_FILE,
        NMC_RUNTIME_FILE,
        OUTER_PATHS_FILE,
        DAY_10_RATES_FILE
    ]

    return all(
        os.path.exists(file)
        for file in required_files
    )


# ============================================================
# 5. Load or run Nested Monte Carlo
# ============================================================

def load_or_run_nmc():

    print(
        "Configuration: "
        f"{number_of_outer_paths:,} outer × "
        f"{number_of_inner_paths:,} inner"
    )

    if nmc_cache_exists():

        print(
            "Existing NMC results found."
        )

        print(
            "Skipping Nested Monte Carlo simulation."
        )

        return load_nmc_results()


    print(
        "No complete NMC cache found."
    )

    print(
        "Running Nested Monte Carlo simulation..."
    )

    nmc_results, nmc_runtime = (
        create_nmc_results()
    )

    save_nmc_results(
        nmc_results=nmc_results,
        nmc_runtime=nmc_runtime
    )

    print(
        f"Nested Monte Carlo completed in "
        f"{nmc_runtime:.2f} seconds."
    )

    return (
        nmc_results,
        nmc_runtime
    )


# ============================================================
# 6. Load saved outer states for sensitivity analysis
# ============================================================

def load_outer_states():

    if not os.path.exists(
        OUTER_PATHS_FILE
    ):

        raise FileNotFoundError(
            "Saved outer paths were not found. "
            "Run the main NMC benchmark first."
        )

    if not os.path.exists(
        DAY_10_RATES_FILE
    ):

        raise FileNotFoundError(
            "Saved Day-10 rates were not found. "
            "Run the main NMC benchmark first."
        )

    saved_outer_paths = np.load(
        OUTER_PATHS_FILE
    )

    saved_day_10_rates = np.load(
        DAY_10_RATES_FILE
    )

    return (
        saved_outer_paths,
        saved_day_10_rates
    )