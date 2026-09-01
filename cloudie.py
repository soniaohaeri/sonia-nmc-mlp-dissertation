# ============================================================
# Cloudie Optimiser Ranking Framework
# ============================================================
# Compare surrogate optimisers using three equally weighted
# dimensions:
#
# 1. Predictive accuracy
# 2. Final training loss
# 3. Risk-measure accuracy
#
# A tolerance-aware midranking procedure prevents negligible
# numerical differences from producing artificial rankings.
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# 1. Tolerance-aware midranking
# ============================================================

def tolerance_rank(
    series,
    higher_is_better=False,
    tolerance=0.005
):

    values = series.astype(float)

    ordered = values.sort_values(
        ascending=not higher_is_better
    )

    groups = []
    current_group = []
    previous_value = None

    for idx, value in ordered.items():

        if previous_value is None:

            current_group = [idx]

        else:

            denominator = max(
                abs(previous_value),
                abs(value),
                1e-12
            )

            relative_difference = (
                abs(value - previous_value)
                / denominator
            )

            if relative_difference <= tolerance:

                current_group.append(idx)

            else:

                groups.append(
                    current_group
                )

                current_group = [idx]

        previous_value = value

    groups.append(
        current_group
    )


    # --------------------------------------------------------
    # Assign the average occupied rank to tied optimisers
    # --------------------------------------------------------

    ranks = pd.Series(
        index=values.index,
        dtype=float
    )

    position = 1

    for group in groups:

        occupied_positions = np.arange(
            position,
            position + len(group)
        )

        average_rank = (
            occupied_positions.mean()
        )

        for idx in group:

            ranks[idx] = average_rank

        position += len(group)

    return ranks


# ============================================================
# 2. Calculate Cloudie rank
# ============================================================

def calculate_cloudie_rank(
    results,
    tolerance=0.005
):

    cloudie = results.copy()


    # --------------------------------------------------------
    # Predictive accuracy
    # Lower MAE / RMSE / (1 - R²) are preferred
    # --------------------------------------------------------
    
    cloudie["MAE Rank"] = tolerance_rank(
        cloudie["MAE"],
        tolerance=tolerance
    )
    
    cloudie["RMSE Rank"] = tolerance_rank(
        cloudie["RMSE"],
        tolerance=tolerance
    )
    
    cloudie["1 - R2"] = (
        1 - cloudie["R2"]
    )
    
    cloudie["1 - R2 Rank"] = tolerance_rank(
        cloudie["1 - R2"],
        tolerance=tolerance
    )
    
    cloudie[
        "Predictive Accuracy Rank"
    ] = cloudie[
        [
            "MAE Rank",
            "RMSE Rank",
            "1 - R2 Rank"
        ]
    ].mean(
        axis=1
    )

    # --------------------------------------------------------
    # Final training loss
    # Lower loss is preferred
    # --------------------------------------------------------

    cloudie[
        "Training Loss Rank"
    ] = tolerance_rank(
        cloudie["Final Training Loss"],
        tolerance=tolerance
    )

    # --------------------------------------------------------
    # Risk-measure accuracy
    # Lower error relative to NMC is preferred
    # --------------------------------------------------------

    risk_columns = [
        "95% VaR Deviation (%)",
        "95% ES Deviation (%)",
        "99% VaR Deviation (%)",
        "99% ES Deviation (%)"
    ]

    risk_rank_columns = []

    for column in risk_columns:

        rank_column = (
            f"{column} Rank"
        )

        cloudie[
            rank_column
        ] = tolerance_rank(
            cloudie[column],
            tolerance=tolerance
        )

        risk_rank_columns.append(
            rank_column
        )

    cloudie[
        "Risk-Measure Accuracy Rank"
    ] = cloudie[
        risk_rank_columns
    ].mean(
        axis=1
    )


    # --------------------------------------------------------
    # Final Cloudie rank
    # Equal weighting across the three dimensions
    # --------------------------------------------------------

    cloudie[
        "Cloudie Rank"
    ] = cloudie[
        [
            "Predictive Accuracy Rank",
            "Training Loss Rank",
            "Risk-Measure Accuracy Rank"
        ]
    ].mean(
        axis=1
    )


    # --------------------------------------------------------
    # Return final ranking
    # --------------------------------------------------------

    output = cloudie[
        [
            "Optimiser",
            "Predictive Accuracy Rank",
            "Training Loss Rank",
            "Risk-Measure Accuracy Rank",
            "Cloudie Rank"
        ]
    ].copy()

    return output.sort_values(
        "Cloudie Rank"
    )


# ============================================================
# 3. Calculate mean risk-measure deviation
# ============================================================

def calculate_risk_deviation(
    risk_results,
    n_inner
):

    risk_measures = [
        "95% VaR (£)",
        "95% ES (£)",
        "99% VaR (£)",
        "99% ES (£)"
    ]


    # --------------------------------------------------------
    # Select the required NMC inner allocation
    # --------------------------------------------------------

    group = risk_results[
        risk_results["Inner Paths"] == n_inner
    ]

    nmc = group[
        group["Method"] == "Nested Monte Carlo"
    ].iloc[0]


    # --------------------------------------------------------
    # Compare each surrogate with the NMC benchmark
    # --------------------------------------------------------

    rows = []

    for optimiser in [
        "Adam",
        "Adagrad",
        "Shampoo"
    ]:

        surrogate = group[
            group["Method"]
            == f"MLP + {optimiser}"
        ].iloc[0]

        deviations = [
            abs(
                surrogate[measure] - nmc[measure]
            )
            / abs(
                nmc[measure]
            )
            * 100

            for measure in risk_measures
        ]

        rows.append({
            "Optimiser": optimiser,

            "Mean Risk-Measure Deviation (%)":
                np.mean(
                    deviations
                )
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Calculate deviation from 1,000-inner NMC benchmark
# ============================================================

def calculate_risk_deviation_1000(
    risk_results,
    n_inner
):

    risk_measures = [
        "95% VaR (£)",
        "95% ES (£)",
        "99% VaR (£)",
        "99% ES (£)"
    ]


    # Surrogates for current inner allocation

    group = risk_results[
        risk_results["Inner Paths"] == n_inner
    ]


    # Fixed 1,000-inner NMC benchmark

    nmc_1000 = risk_results[
        (risk_results["Inner Paths"] == 1000)
        &
        (
            risk_results["Method"]
            == "Nested Monte Carlo"
        )
    ].iloc[0]


    rows = []

    for optimiser in [
        "Adam",
        "Adagrad",
        "Shampoo"
    ]:

        surrogate = group[
            group["Method"]
            == f"MLP + {optimiser}"
        ].iloc[0]

        deviations = [
            abs(
                surrogate[measure]
                - nmc_1000[measure]
            )
            / abs(
                nmc_1000[measure]
            )
            * 100

            for measure in risk_measures
        ]

        rows.append({
            "Optimiser": optimiser,

            "Mean Risk-Measure Deviation "
            "from 1,000-inner NMC (%)":
                np.mean(deviations)
        })

    return pd.DataFrame(rows)
# ============================================================
# 4. Create Final Cloudie Summary
# ============================================================

def create_cloudie_summary(
    metrics,
    losses,
    risk_results,
    n_inner,
    tolerance=0.005
):

    optimisers = [
        "Adam",
        "Adagrad",
        "Shampoo"
    ]

    rows = []


    # --------------------------------------------------------
    # NMC Risk Benchmark
    # --------------------------------------------------------

    risk_group = risk_results[
        risk_results["Inner Paths"] == n_inner
    ].set_index("Method")

    nmc_risk = risk_group.loc[
        "Nested Monte Carlo"
    ]


    # --------------------------------------------------------
    # Assemble Cloudie Inputs
    # --------------------------------------------------------

    for optimiser in optimisers:

        optimiser_metrics = metrics[
            optimiser
        ]

        surrogate_risk = risk_group.loc[
            f"MLP + {optimiser}"
        ]

        rows.append({
            "Optimiser": optimiser,

            "MAE":
                optimiser_metrics["MAE"],

            "RMSE":
                optimiser_metrics["RMSE"],

            "R2":
                optimiser_metrics["R2"],

            "Final Training Loss":
                losses[optimiser][-1],

            "95% VaR Deviation (%)":
                abs(
                    surrogate_risk["95% VaR (£)"]
                    - nmc_risk["95% VaR (£)"]
                )
                / abs(
                    nmc_risk["95% VaR (£)"]
                )
                * 100,
            
            "95% ES Deviation (%)":
                abs(
                    surrogate_risk["95% ES (£)"]
                    - nmc_risk["95% ES (£)"]
                )
                / abs(
                    nmc_risk["95% ES (£)"]
                )
                * 100,
            
            "99% VaR Deviation (%)":
                abs(
                    surrogate_risk["99% VaR (£)"]
                    - nmc_risk["99% VaR (£)"]
                )
                / abs(
                    nmc_risk["99% VaR (£)"]
                )
                * 100,
            
            "99% ES Deviation (%)":
                abs(
                    surrogate_risk["99% ES (£)"]
                    - nmc_risk["99% ES (£)"]
                )
                / abs(
                    nmc_risk["99% ES (£)"]
                )
                * 100
        })


    cloudie_inputs = pd.DataFrame(
        rows
    )


    # --------------------------------------------------------
    # Calculate Cloudie Rank
    # --------------------------------------------------------

    ranks = calculate_cloudie_rank(
        results=cloudie_inputs,
        tolerance=tolerance
    )


    # --------------------------------------------------------
    # Mean Risk-Measure Deviation
    # Current NMC Inner Allocation
    # --------------------------------------------------------

    deviation = calculate_risk_deviation(
        risk_results=risk_results,
        n_inner=n_inner
    )


    # --------------------------------------------------------
    # Mean Risk-Measure Deviation
    # 1,000-Inner NMC Benchmark
    # --------------------------------------------------------

    deviation_1000 = calculate_risk_deviation_1000(
        risk_results=risk_results,
        n_inner=n_inner
    )


    # --------------------------------------------------------
    # Final Cloudie Table
    # --------------------------------------------------------

    cloudie_summary = (
        ranks
        .merge(
            deviation,
            on="Optimiser"
        )
        .merge(
            deviation_1000,
            on="Optimiser"
        )
    )

    return cloudie_summary