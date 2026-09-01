# ============================================================
# V0 Pricing
# ============================================================
# Simulate full-year Hull-White paths from today,
# price the four caplets, discount them to today,
# and average across paths to obtain the initial cap value.
# ============================================================

import numpy as np

from market_data import (
    r0,
    a,
    sigma,
    strike,
    notional,
    curve_t,
    forward_curve,
    business_days_per_year,
    number_of_inner_paths,
    number_of_caplets
)

from hull_white import (
    simulate_outer_hull_white
)

from caplet_pricing import (
    full_caplet_boundaries,
    compound_sonia_business_days,
    calculate_caplet_cashflow,
    calculate_discount_factor
)


# ============================================================
# 1. Calculate initial cap value V0
# ============================================================

def calculate_v0():

    # --------------------------------------------------------
    # Full one-year simulation grid:
    # 253 observations -> 252 accrual intervals
    # --------------------------------------------------------

    today_t = (
        np.arange(business_days_per_year + 1)
        / business_days_per_year
    )

    today_forward = np.interp(
        today_t,
        curve_t,
        forward_curve
    )


    # --------------------------------------------------------
    # Simulate full-year Hull-White paths from today
    # --------------------------------------------------------

    today_paths = simulate_outer_hull_white(
        r0=r0,
        a=a,
        sigma=sigma,
        t=today_t,
        forward=today_forward,
        number_of_paths=number_of_inner_paths,
        seed=42
    )

    # 253 observations -> 252 accrual intervals
    interval_rates = today_paths[:-1, :]

    number_of_paths = interval_rates.shape[1]

    pathwise_cap_values = np.zeros(
        number_of_paths
    )

    caplet_prices = []


    # --------------------------------------------------------
    # Loop through the four quarterly caplets
    # --------------------------------------------------------

    for q in range(number_of_caplets):

        start = full_caplet_boundaries[q]
        end = full_caplet_boundaries[q + 1]


        # ----------------------------------------------------
        # Rates used to determine the caplet's SONIA rate
        # ----------------------------------------------------

        caplet_rates = interval_rates[
            start:end,
            :
        ]


        # ----------------------------------------------------
        # Compound SONIA over the 63-business-day period
        # ----------------------------------------------------

        compounded_rate, accrual_fraction = (
            compound_sonia_business_days(
                caplet_rates
            )
        )


        # ----------------------------------------------------
        # Calculate the caplet cashflow
        # ----------------------------------------------------

        caplet_cashflow = calculate_caplet_cashflow(
            compounded_rate=compounded_rate,
            strike=strike,
            notional=notional,
            accrual_fraction=accrual_fraction
        )


        # ----------------------------------------------------
        # Discount the caplet cashflow back to today
        # ----------------------------------------------------

        discount_factor = calculate_discount_factor(
            interval_rates[:end, :]
        )

        discounted_caplet_value = (
            caplet_cashflow
            * discount_factor
        )


        # ----------------------------------------------------
        # Add the discounted caplet value to each path
        # ----------------------------------------------------

        pathwise_cap_values += (
            discounted_caplet_value
        )


        # ----------------------------------------------------
        # Monte Carlo value of the individual caplet
        # ----------------------------------------------------

        caplet_prices.append(
            discounted_caplet_value.mean()
        )


    # --------------------------------------------------------
    # Average across paths to obtain the initial cap value
    # --------------------------------------------------------

    V0 = (
        pathwise_cap_values.mean()
    )

    return V0, caplet_prices, pathwise_cap_values