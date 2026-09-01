import numpy as np
import pandas as pd


# ============================================================
# 1. Contract and model parameters
# ============================================================

notional = 1_000_000
strike = 0.04
number_of_caplets = 4

a = 0.04
sigma = 0.01

business_days_per_year = 252

number_of_outer_paths = 100_000
number_of_inner_paths = 1_000

risk_horizon_days = 10
maturity_business_day = 252


# ============================================================
# 2. Valuation dates
# ============================================================

valuation_date = pd.Timestamp("2026-08-03")

risk_date = valuation_date + pd.offsets.BDay(
    risk_horizon_days
)

t_risk = (
    risk_date - valuation_date
).days / 365.0


# ============================================================
# 3. Load Bloomberg GBP SONIA OIS curve
# ============================================================

curve = pd.read_excel(
    "GBP_OIS_Bloomberg.xlsx"
)

curve["Date"] = pd.to_datetime(
    curve["Date"],
    format="%d/%m/%Y"
)

curve["t"] = (
    curve["Date"] - valuation_date
).dt.days / 365.0


# ============================================================
# 4. Add time zero
# ============================================================

time_zero = pd.DataFrame({
    "Date": [valuation_date],
    "Zero Rate (%)": [np.nan],
    "Discount Factor": [1.0],
    "t": [0.0]
})

curve = pd.concat(
    [time_zero, curve],
    ignore_index=True
)

curve = (
    curve
    .sort_values("t")
    .reset_index(drop=True)
)


# ============================================================
# 5. Calculate initial forward curve
# ============================================================

curve_t = curve["t"].to_numpy(dtype=float)
discount_curve = curve["Discount Factor"].to_numpy(dtype=float)

forward_curve = -np.gradient(
    np.log(discount_curve),
    curve_t
)

curve["Forward Rate"] = forward_curve

r0 = forward_curve[0]