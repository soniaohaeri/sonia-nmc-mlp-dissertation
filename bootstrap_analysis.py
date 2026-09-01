# ============================================================
# Bootstrap Analysis
# ============================================================
# Construct bootstrap confidence intervals for surrogate
# predictive accuracy and estimated risk measures.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import bootstrap

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. Calculate predictive accuracy
# ============================================================

def calculate_accuracy(
    y_true,
    y_pred
):

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    return {
        "MAE": mean_absolute_error(
            y_true,
            y_pred
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        ),

        "R2": r2_score(
            y_true,
            y_pred
        )
    }


# ============================================================
# 2. Bootstrap predictive accuracy
# ============================================================

def bootstrap_accuracy(
    y_true,
    y_pred,
    n_resamples=1000,
    confidence_level=0.95,
    seed=42
):

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    indices = np.arange(
        len(y_true)
    )


    # --------------------------------------------------------
    # Bootstrap one accuracy statistic
    # --------------------------------------------------------

    def bootstrap_metric(
        metric
    ):

        def statistic(
            sample_indices
        ):

            sample_indices = (
                sample_indices.astype(int)
            )

            return metric(
                y_true[sample_indices],
                y_pred[sample_indices]
            )

        result = bootstrap(
            (indices,),
            statistic,
            vectorized=False,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method="percentile",
            batch=100,
            rng=np.random.default_rng(seed)
        )

        return (
            result.confidence_interval.low,
            result.confidence_interval.high
        )


    # --------------------------------------------------------
    # Bootstrap MAE, RMSE and R²
    # --------------------------------------------------------

    mae_ci = bootstrap_metric(
        mean_absolute_error
    )

    rmse_ci = bootstrap_metric(
        lambda y, p:
            np.sqrt(
                mean_squared_error(y, p)
            )
    )

    r2_ci = bootstrap_metric(
        r2_score
    )


    return {
        "MAE Lower": mae_ci[0],
        "MAE Upper": mae_ci[1],

        "RMSE Lower": rmse_ci[0],
        "RMSE Upper": rmse_ci[1],

        "R2 Lower": r2_ci[0],
        "R2 Upper": r2_ci[1]
    }


# ============================================================
# 3. Create predictive accuracy table
# ============================================================

def create_accuracy_table(
    y_true,
    predictions,
    n_resamples=1000,
    confidence_level=0.95
):

    results = []

    for optimiser, y_pred in predictions.items():

        metrics = calculate_accuracy(
            y_true,
            y_pred
        )

        ci = bootstrap_accuracy(
            y_true=y_true,
            y_pred=y_pred,
            n_resamples=n_resamples,
            confidence_level=confidence_level
        )

        results.append({

            "Optimiser": optimiser,

            "MAE": round(
                metrics["MAE"],
                2
            ),

            "RMSE": round(
                metrics["RMSE"],
                2
            ),

            "R2": round(
                metrics["R2"],
                4
            ),

            "MAE 95% CI":
                f"[{ci['MAE Lower']:.2f}, "
                f"{ci['MAE Upper']:.2f}]",

            "RMSE 95% CI":
                f"[{ci['RMSE Lower']:.2f}, "
                f"{ci['RMSE Upper']:.2f}]",

            "R2 95% CI":
                f"[{ci['R2 Lower']:.4f}, "
                f"{ci['R2 Upper']:.4f}]",

            "MAE CI Lower":
                ci["MAE Lower"],

            "MAE CI Upper":
                ci["MAE Upper"],

            "RMSE CI Lower":
                ci["RMSE Lower"],

            "RMSE CI Upper":
                ci["RMSE Upper"],

            "R2 CI Lower":
                ci["R2 Lower"],

            "R2 CI Upper":
                ci["R2 Upper"]
        })

    return pd.DataFrame(
        results
    )


# ============================================================
# 4. Calculate VaR and Expected Shortfall
# ============================================================

def calculate_var_es(
    losses,
    confidence_level
):

    losses = np.asarray(
        losses
    ).flatten()

    var = np.quantile(
        losses,
        confidence_level,
        method="linear"
    )

    es = losses[
        losses >= var
    ].mean()

    return var, es


# ============================================================
# 5. Bootstrap VaR and Expected Shortfall
# ============================================================

def bootstrap_var_es(
    losses,
    confidence_level,
    n_resamples=1000,
    bootstrap_confidence_level=0.95,
    seed=42
):

    losses = np.asarray(
        losses
    ).flatten()


    # --------------------------------------------------------
    # Bootstrap one risk measure
    # --------------------------------------------------------

    def bootstrap_risk_measure(
        statistic
    ):

        result = bootstrap(
            (losses,),
            statistic,
            vectorized=False,
            confidence_level=bootstrap_confidence_level,
            n_resamples=n_resamples,
            method="percentile",
            batch=100,
            rng=np.random.default_rng(seed)
        )

        return (
            result.confidence_interval.low,
            result.confidence_interval.high
        )


    # --------------------------------------------------------
    # Bootstrap VaR and Expected Shortfall
    # --------------------------------------------------------

    var_ci = bootstrap_risk_measure(
        lambda x:
            calculate_var_es(
                x,
                confidence_level
            )[0]
    )

    es_ci = bootstrap_risk_measure(
        lambda x:
            calculate_var_es(
                x,
                confidence_level
            )[1]
    )


    return {
        "VaR Lower": var_ci[0],
        "VaR Upper": var_ci[1],

        "ES Lower": es_ci[0],
        "ES Upper": es_ci[1]
    }


# ============================================================
# 6. Plot predictive accuracy confidence intervals
# ============================================================

def plot_accuracy_ci(
    optimiser_results
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    metrics = [
        ("MAE", "MAE (£)"),
        ("RMSE", "RMSE (£)"),
        ("R2", "R²")
    ]

    for ax, (metric, ylabel) in zip(
        axes,
        metrics
    ):

        values = optimiser_results[
            metric
        ].to_numpy()

        lower = optimiser_results[
            f"{metric} CI Lower"
        ].to_numpy()

        upper = optimiser_results[
            f"{metric} CI Upper"
        ].to_numpy()

        ax.errorbar(
            optimiser_results["Optimiser"],
            values,
            yerr=[
                values - lower,
                upper - values
            ],
            fmt="o",
            capsize=5
        )

        ax.set_title(
            metric
        )

        ax.set_xlabel(
            "Optimiser"
        )

        ax.set_ylabel(
            ylabel
        )

    fig.suptitle(
        "Optimiser Accuracy with 95% Bootstrap Confidence Intervals"
    )

    plt.tight_layout()
    plt.show()