import numpy as np


# ============================================================
# 1. Caplet assumptions
# ============================================================

full_caplet_days = np.array([63, 63, 63, 63]) #A 252-business-day year is assumed throughout the simulation. The one-year cap is divided into four equal quarterly caplets of 63 business days each.

elapsed_business_days = 10

remaining_caplet_days = np.array([
    full_caplet_days[0] - elapsed_business_days,
    full_caplet_days[1],
    full_caplet_days[2],
    full_caplet_days[3]
])

full_caplet_boundaries = np.array([
    0, 63, 126, 189, 252
])

remaining_caplet_boundaries = np.array([
    0, 53, 116, 179, 242
])

# ============================================================
# 2. Compound SONIA over a caplet period
# ============================================================

def compound_sonia_business_days(
    caplet_rates: np.ndarray
):
    number_of_days = caplet_rates.shape[0]

    accumulation_factor = np.prod(
        1.0 + caplet_rates / 252.0,
        axis=0
    )

    accrual_fraction = (
        number_of_days / 252.0
    )

    compounded_rate = (
        accumulation_factor - 1.0
    ) / accrual_fraction

    return compounded_rate, accrual_fraction


# ============================================================
# 3. Calculate caplet cashflow
# ============================================================

def calculate_caplet_cashflow(
    compounded_rate: np.ndarray,
    strike: float,
    notional: float,
    accrual_fraction: float
) -> np.ndarray:

    caplet_cashflow = (
        notional
        * accrual_fraction
        * np.maximum(
            compounded_rate - strike,
            0.0
        )
    )

    return caplet_cashflow


# ============================================================
# 4. Calculate discount factor
# ============================================================

def calculate_discount_factor(
    rates_from_valuation_to_payment: np.ndarray
) -> np.ndarray:

    integrated_rate = np.sum(
        rates_from_valuation_to_payment / 252.0,
        axis=0
    )

    discount_factor = np.exp(
        -integrated_rate
    )

    return discount_factor