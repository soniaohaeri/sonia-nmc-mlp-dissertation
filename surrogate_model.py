# ============================================================
# Surrogate Model with Adaptive Optimisers
# ============================================================
# Train a multilayer perceptron (MLP) surrogate to approximate
# the Nested Monte Carlo Day-10 cap valuation function.
# ============================================================

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. Surrogate MLP architecture
# ============================================================

class CapSurrogate(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# 2. Prepare training and test data
# ============================================================

def prepare_surrogate_data(
    nmc_results,
    test_size=0.20,
    seed=42
):

    # --------------------------------------------------------
    # Input:
    # Day-10 short-rate state from the outer simulation
    # --------------------------------------------------------

    X = nmc_results[
        ["Day 10 Short Rate"]
    ].to_numpy(dtype=np.float32)


    # --------------------------------------------------------
    # Target:
    # Day-10 cap value from Nested Monte Carlo
    # --------------------------------------------------------

    y = nmc_results[
        ["Day 10 Cap Value (£)"]
    ].to_numpy(dtype=np.float32)


    # --------------------------------------------------------
    # Split into training and test samples
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed
        )
    )


    # --------------------------------------------------------
    # Standardise using training data only
    # --------------------------------------------------------

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(
        X_train
    )

    X_test_scaled = scaler_X.transform(
        X_test
    )

    y_train_scaled = scaler_y.fit_transform(
        y_train
    )


    # --------------------------------------------------------
    # Convert scaled data to PyTorch tensors
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train_scaled,
        dtype=torch.float32
    )

    return (
        X_train_tensor,
        X_test_tensor,
        y_train_tensor,
        scaler_X,
        scaler_y,
        X_test,
        y_test
    )


# ============================================================
# 3. Train surrogate model
# ============================================================

def train_surrogate(
    model,
    optimiser,
    X_train,
    y_train,
    epochs=1000
):

    loss_function = nn.MSELoss()

    losses = []

    model.train()

    for epoch in range(epochs):

        optimiser.zero_grad()

        prediction = model(
            X_train
        )

        loss = loss_function(
            prediction,
            y_train
        )

        loss.backward()

        optimiser.step()

        losses.append(
            loss.item()
        )

    return model, losses


# ============================================================
# 4. Evaluate surrogate against NMC test values
# ============================================================

def evaluate_surrogate(
    model,
    X_test,
    y_test_original,
    scaler_y
):

    model.eval()

    with torch.no_grad():

        predictions_scaled = model(
            X_test
        ).numpy()


    # --------------------------------------------------------
    # Return predictions to original £ scale
    # --------------------------------------------------------

    predictions = scaler_y.inverse_transform(
        predictions_scaled
    )


    # --------------------------------------------------------
    # Calculate out-of-sample accuracy metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test_original,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test_original,
            predictions
        )
    )

    r2 = r2_score(
        y_test_original,
        predictions
    )

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return predictions, metrics