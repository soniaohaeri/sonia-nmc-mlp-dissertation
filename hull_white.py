import numpy as np
import pandas as pd

from market_data import (
    r0,
    a,
    sigma,
    curve_t,
    forward_curve,
    business_days_per_year,
    number_of_outer_paths,
    number_of_inner_paths,
    risk_horizon_days,
    maturity_business_day
)

# ============================================================
# 1. Outer Simulations
# ============================================================

def simulate_outer_hull_white(
    r0: float,
    a: float,
    sigma: float,
    t: np.ndarray,
    forward: np.ndarray,
    number_of_paths: int,
    seed: int = 42
) -> np.ndarray:

    number_of_times = t.shape[0]

    rates = np.ones(
        (number_of_times, number_of_paths)
    ) * r0

    alpha = (
        forward
        + sigma**2 / (2 * a**2)
        * (1 - np.exp(-a * t))**2
    )

    rng = np.random.default_rng(seed)

    Z = rng.standard_normal(
        size=(number_of_times - 1, number_of_paths)
    )

    for i in range(1, number_of_times):
        delta_t = t[i] - t[i - 1]

        decay = np.exp(-a * delta_t)

        expected_rate = (
            rates[i - 1, :] * decay
            + alpha[i]
            - alpha[i - 1] * decay
        )

        variance = (
            sigma**2 / (2 * a)
            * (1 - np.exp(-2 * a * delta_t))
        )

        rates[i, :] = (
            expected_rate
            + np.sqrt(variance) * Z[i - 1, :]
        )

    return rates


outer_t = (
    np.arange(risk_horizon_days + 1)
    / business_days_per_year
)

outer_forward = np.interp(
    outer_t,
    curve_t,
    forward_curve
)

outer_paths = simulate_outer_hull_white(
    r0=r0,
    a=a,
    sigma=sigma,
    t=outer_t,
    forward=outer_forward,
    number_of_paths=number_of_outer_paths,
    seed=42
)

day_10_rates = outer_paths[-1, :]

outer_results = pd.DataFrame({
    "Outer Scenario": np.arange(
        1,
        number_of_outer_paths + 1
    ),
    "Day 10 Short Rate (%)": day_10_rates * 100
})

# ============================================================
# 1. Inner Simulations
# ============================================================

def simulate_inner_hull_white(
    r_start: float,
    a: float,
    sigma: float,
    t: np.ndarray,
    forward: np.ndarray,
    number_of_paths: int,
    seed: int = 42
) -> np.ndarray:

    number_of_times = t.shape[0]

    rates = np.ones(
        (number_of_times, number_of_paths)
    ) * r_start

    alpha = (
        forward
        + sigma**2 / (2 * a**2)
        * (1 - np.exp(-a * t))**2
    )

    rng = np.random.default_rng(seed)

    Z = rng.standard_normal(
        size=(number_of_times - 1, number_of_paths)
    )

    for i in range(1, number_of_times):
        delta_t = t[i] - t[i - 1]

        decay = np.exp(-a * delta_t)

        expected_rate = (
            rates[i - 1, :] * decay
            + alpha[i]
            - alpha[i - 1] * decay
        )

        variance = (
            sigma**2 / (2 * a)
            * (1 - np.exp(-2 * a * delta_t))
        )

        rates[i, :] = (
            expected_rate
            + np.sqrt(variance) * Z[i - 1, :]
        )

    return rates


inner_t = (
    np.arange(
        risk_horizon_days,
        maturity_business_day + 1
    )
    / business_days_per_year
)

inner_forward = np.interp(
    inner_t,
    curve_t,
    forward_curve
)