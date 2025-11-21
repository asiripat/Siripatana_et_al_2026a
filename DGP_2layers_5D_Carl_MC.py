#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-layer Deep Gaussian Process for 10 POD coefficients using GPyTorch (no GPflux).
- Python 3.12 compatible
- Same inputs as your GPflux script:
    sample_data/X_input_Carl_5D.csv
    sample_data/Y_pod_coefficients_Carl_5D.csv
    sample_data/combined_nodes_dat.csv  (test inputs)
- Trains 10 independent 2-layer DGPs (one per POD coefficient)
- Saves mean and central 90% interval (approx via Normal) for test set
- Plots train fit with 90% error bars
- Calibrates predictive std with a scalar to target ~90% empirical coverage on train
"""

import os
import math
import time
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt

# --------------------------
# Repro & device
# --------------------------
torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# I/O
# --------------------------
X_path = "sample_data/X_input_Carl180_5D.csv"
Y_path = "sample_data/Y_pod_coefficients_Carl180_5D.csv"
X_test_path = "sample_data/combined_nodes_dat_MC_Carl_5D.csv"

# Load (skip header row)
X_full = np.loadtxt(X_path, delimiter=",", skiprows=1)
Y_full = np.loadtxt(Y_path, delimiter=",", skiprows=1)

# Use first 125 samples, matching your script
N_TRAIN = 180
X_train = X_full[:N_TRAIN].astype(np.float64)
Y_train = Y_full[:N_TRAIN].astype(np.float64)  # shape (N, 10)

# Normalize inputs
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0, ddof=0)
X_std[X_std == 0.0] = 1.0
Xn = (X_train - X_mean) / X_std

# Normalize outputs
Y_mean = Y_train.mean(axis=0)           # (10,)
Y_std = Y_train.std(axis=0, ddof=0)     # (10,)
Y_std[Y_std == 0.0] = 1.0

# Test inputs
X_test = np.loadtxt(X_test_path, delimiter=",", skiprows=1).astype(np.float64)
X_testn = (X_test - X_mean) / X_std

# --------------------------
# DeepGP components (GPyTorch)
# --------------------------
from gpytorch.variational import (
    VariationalStrategy,
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy
)
from gpytorch.models.deep_gps import DeepGP
from gpytorch.distributions import MultitaskMultivariateNormal

class _GPHiddenLayer(gpytorch.models.ApproximateGP):
    def __init__(self, D_in: int, Q: int, M: int, Z: torch.Tensor):
        # Each latent GP has its own inducing points and distribution
        variational_distribution = CholeskyVariationalDistribution(M, batch_shape=torch.Size([Q]))
        variational_strategy = VariationalStrategy(
            self,
            Z.unsqueeze(0).expand(Q, -1, -1),   # shape (Q, M, D_in)
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([Q]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=D_in, batch_shape=torch.Size([Q])),
            batch_shape=torch.Size([Q])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)              # (Q, N)
        covar_x = self.covar_module(x)            # (Q, N, N)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class _GPOutputLayer(gpytorch.models.ApproximateGP):
    def __init__(self, Q: int, M: int, Z: torch.Tensor):
        variational_distribution = CholeskyVariationalDistribution(M)
        variational_strategy = VariationalStrategy(self, Z, variational_distribution, learn_inducing_locations=True)
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=Q)
        )

    def forward(self, x):
        # x: (N, Q)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class TwoLayerDeepGP(DeepGP):
    def __init__(self, X_init: torch.Tensor, Q: int = 3, M1: int = 64, M2: int = 32):
        super().__init__()
        N, D_in = X_init.shape

        # Layer 1 inducing points from inputs
        perm1 = torch.randperm(N)[:M1]
        Z1 = X_init[perm1].clone()
        self.layer1 = _GPHiddenLayer(D_in=D_in, Q=Q, M=M1, Z=Z1)

        # Init layer-2 inducing points in Q-dim space
        with torch.no_grad():
            W = torch.randn(D_in, Q, dtype=X_init.dtype, device=X_init.device)
            H_init = (X_init @ W)
            H_init = (H_init - H_init.mean(dim=0)) / (H_init.std(dim=0) + 1e-6)
        perm2 = torch.randperm(N)[:M2]
        Z2 = H_init[perm2].clone()    # (M2, Q)
        self.layer2 = _GPOutputLayer(Q=Q, M=M2, Z=Z2)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = torch.tensor(1e-3, dtype=X_init.dtype)

    def forward(self, x):
        """
        x: (N, D_in)
        layer1(x) -> MVN with batch shape Q and event shape N
        rsample() -> (Q, N), so transpose to (N, Q) for layer2
        """
        hidden = self.layer1(x)                       # batch Q MVN over N
        h = hidden.rsample()                          # (Q, N)
        h = h.transpose(-1, -2).contiguous()          # (N, Q)  <-- KEY FIX
        output = self.layer2(h)                       # MVN over N
        return output

    def predict(self, x, num_mc_samples: int = 16):
        self.eval(); self.likelihood.eval()
        with torch.no_grad():
            means, variances = [], []
            for _ in range(num_mc_samples):
                out = self(x)
                pred = self.likelihood(out)
                means.append(pred.mean.unsqueeze(-1))         # (N,1)
                variances.append(pred.variance.unsqueeze(-1)) # (N,1)
            mean = torch.mean(torch.cat(means, dim=-1), dim=-1).squeeze(-1)
            var  = torch.mean(torch.cat(variances, dim=-1), dim=-1).squeeze(-1)
        return mean, var

# --------------------------
# Training utils
# --------------------------
def train_one_pod(
    Xn_np: np.ndarray,
    y_np: np.ndarray,
    max_steps: int = 6000,
    lr: float = 0.01,
    Q_hidden: int = 3,
    M1: int = 64,
    M2: int = 32,
    batch_size: int = 128,
    print_every: int = 500,
    patience: int = 800,
):
    """
    Train a two-layer DeepGP for a single POD target (y_np shape [N,]).
    Returns: model, calibrated_scale (float), train_mean, train_std
    """
    X = torch.tensor(Xn_np, dtype=torch.float64, device=device)
    y = torch.tensor(y_np.reshape(-1, 1), dtype=torch.float64, device=device).squeeze(-1)

    model = TwoLayerDeepGP(X_init=X, Q=Q_hidden, M1=M1, M2=M2).to(device).double()
    likelihood = model.likelihood

    # ELBO with deep approximate MLL
    num_data = X.shape[0]
    elbo = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)
    mll = gpytorch.mlls.DeepApproximateMLL(elbo)

    # Optimizer & scheduler (cosine decay for stability)
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

    model.train(); likelihood.train()

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    # Mini-batches (shuffle each epoch-ish)
    idx = torch.arange(num_data, device=device)

    for step in range(1, max_steps + 1):
        perm = torch.randperm(num_data, device=device)
        for start in range(0, num_data, batch_size):
            sel = perm[start:start+batch_size]
            x_b = X[sel]
            y_b = y[sel]

            optimizer.zero_grad()
            output = model(x_b)
            loss = -mll(output, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        scheduler.step()

        if step % print_every == 0 or step == 1:
            with torch.no_grad():
                out_full = model(X)
                full_loss = -mll(out_full, y).item()
            print(f"  Step {step:5d} | ELBO: {-full_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.5f}")

            if full_loss < best_loss - 1e-4:
                best_loss = full_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += print_every
                if no_improve >= patience:
                    print(f"  Early stopping at step {step} (no improvement for {patience} steps)")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict on train
    mean_train, var_train = model.predict(X, num_mc_samples=24)
    std_train = torch.sqrt(var_train).cpu().numpy()
    mean_train = mean_train.cpu().numpy()
    y_true = y.cpu().numpy()

    # ---- Uncertainty calibration (simple & conservative) ----
    # Target 90% nominal: mean ± 1.645 * s * std should contain ~90% of y_true
    eps = 1e-9
    resid = np.abs(y_true - mean_train)
    ratio = resid / (std_train + eps)
    q90 = np.quantile(ratio, 0.90)
    s = max(1.0, float(q90 / 1.645))  # only scale up (conservative)
    # ---------------------------------------------------------

    return model, s, mean_train, std_train

# --------------------------
# Main training loop over 10 POD coefficients
# --------------------------
def main():
    n_out = Y_train.shape[1]
    assert n_out == 10, f"Expected 10 POD coefficients, found {n_out}"

    models = []
    scales = []
    rmses = []
    train_means_all = []
    train_stds_all = []

    # Use double precision for stability
    torch.set_default_dtype(torch.float64)

    for i in range(n_out):
        print(f"\n=== Training 2-layer DeepGP for POD Coefficient {i+1} ===")
        yi = ((Y_train[:, i] - Y_mean[i]) / Y_std[i]).astype(np.float64)

        # Warn if nearly constant after normalization
        std_norm = float(np.std(yi))
        print(f"Normalized std: {std_norm:.5f}")
        if std_norm < 1e-2:
            print("⚠️  This POD looks nearly constant; uncertainty may collapse.")

        model, scale, mean_tr_n, std_tr_n = train_one_pod(
            Xn_np=Xn,
            y_np=yi,
            max_steps=6000,
            lr=0.01,
            Q_hidden=3,
            M1=64,
            M2=32,
            batch_size=128,
            print_every=400,
            patience=1200,
        )
        models.append(model)
        scales.append(scale)

        # Unnormalize train predictions for diagnostics
        mean_tr = mean_tr_n * Y_std[i] + Y_mean[i]
        std_tr = std_tr_n * Y_std[i] * scale

        y_true = Y_train[:, i]
        rmse = float(np.sqrt(np.mean((mean_tr - y_true) ** 2)))
        rmses.append(rmse)

        # Plot train fit with 90% CI
        lower = mean_tr - 1.645 * std_tr
        upper = mean_tr + 1.645 * std_tr

        # plt.figure(figsize=(6, 4))
        # lower_err = mean_tr - lower
        # upper_err = upper - mean_tr
        # plt.errorbar(y_true, mean_tr, yerr=[lower_err, upper_err], fmt='o', alpha=0.6,
        #              label="Mean ± 90% CI (calibrated)")
        # mn, mx = float(np.min(y_true)), float(np.max(y_true))
        # plt.plot([mn, mx], [mn, mx], 'k--', lw=1, label="Ideal line")
        # plt.xlabel("True POD Coefficient")
        # plt.ylabel("Predicted")
        # plt.title(f"POD {i+1} — RMSE: {rmse:.4f} — scale s={scale:.3f}")
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        # plt.show()

        train_means_all.append(mean_tr)
        train_stds_all.append(std_tr)

    print("\n=== RMSE Summary (Train) ===")
    for i, rmse in enumerate(rmses, 1):
        print(f"POD {i:2d}: RMSE = {rmse:.5f} | var scale s = {scales[i-1]:.3f}")

    # --------------------------
    # Predict on TEST inputs
    # --------------------------
    Xtest_t = torch.tensor(X_testn, dtype=torch.float64, device=device)

    Y_mean_list = []
    Y_std_list = []
    Y_lo_list = []
    Y_hi_list = []

    for i, model in enumerate(models):
        model.eval(); model.likelihood.eval()
        with torch.no_grad():
            mean_t_n, var_t_n = model.predict(Xtest_t, num_mc_samples=32)
        mean_t_n = mean_t_n.cpu().numpy()
        std_t_n = np.sqrt(var_t_n.cpu().numpy())

        # Unnormalize & calibrate std
        mean_t = mean_t_n * Y_std[i] + Y_mean[i]
        std_t = std_t_n * Y_std[i] * scales[i]

        Y_mean_list.append(mean_t)
        Y_std_list.append(std_t)
        Y_lo_list.append(mean_t - 1.645 * std_t)  # 5th pct (approx)
        Y_hi_list.append(mean_t + 1.645 * std_t)  # 95th pct (approx)

    # Stack to (N_test, 10)
    Y_mean_arr = np.vstack(Y_mean_list).T
    Y_std_arr = np.vstack(Y_std_list).T
    Y_lower_5_arr = np.vstack(Y_lo_list).T
    Y_upper_95_arr = np.vstack(Y_hi_list).T

    # Save to CSVs
    header_line = ",".join([f"POD{i+1}" for i in range(10)])
    np.savetxt("Y_pred_test_Carl_5D_2layer_mean_60pts_gpytorch_MC.csv",
               Y_mean_arr, delimiter=",", header=header_line, comments='')
    np.savetxt("Y_pred_test_Carl_5D_2layer_std_60pts_gpytorch_MC.csv",
               Y_std_arr, delimiter=",", header=header_line, comments='')
    np.savetxt("Y_pred_test_Carl_5D_2layer_lower_5_60pts_gpytorch_MC.csv",
               Y_lower_5_arr, delimiter=",", header=header_line, comments='')
    np.savetxt("Y_pred_test_Carl_5D_2layer_upper_95_60pts_gpytorch_MC.csv",
               Y_upper_95_arr, delimiter=",", header=header_line, comments='')

    print("\n✅ Test predictions saved to:")
    print("   - Y_pred_test_Carl_5D_2layer_mean_125pts_gpytorch.csv")
    print("   - Y_pred_test_Carl_5D_2layer_std_125pts_gpytorch.csv")
    print("   - Y_pred_test_Carl_5D_2layer_lower_5_125pts_gpytorch.csv")
    print("   - Y_pred_test_Carl_5D_2layer_upper_95_125pts_gpytorch.csv")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n⏱️ Total training time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")

