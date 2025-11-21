import os
import numpy as np
import matplotlib.pyplot as plt
from dgpsi import dgp, kernel, combine, emulator

# --------------------------
# Parameters & file paths
# --------------------------
X_path = "sample_data/X_input_Carl180_5D.csv"
Y_path = "sample_data/Y_pod_coefficients_Carl180_5D.csv"
X_test_path = "sample_data/combined_nodes_dat_MC_Carl_5D.csv"

N_TRAIN = 180
N_POD = 10  # number of POD coefficients to use

# --------------------------
# Load inputs/outputs
# --------------------------
X_full = np.loadtxt(X_path, delimiter=",", skiprows=1)
Y_full = np.loadtxt(Y_path, delimiter=",", skiprows=1)

# Training data
X_train = X_full[:N_TRAIN].astype(float)   # (125, 5)
Y_train = Y_full[:N_TRAIN].astype(float)   # (125, 10)

# Normalize inputs (z-score)
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0, ddof=0)
X_std[X_std == 0.0] = 1.0
Xn = (X_train - X_mean) / X_std

# Normalize outputs (per coefficient)
Y_mean = Y_train.mean(axis=0)
Y_std = Y_train.std(axis=0, ddof=0)
Y_std[Y_std == 0.0] = 1.0
Yn = (Y_train - Y_mean) / Y_std

print(f"Inputs normalized shape: {Xn.shape}")
print(f"POD coeffs normalized shape: {Yn.shape}")

# --------------------------
# Define DeepGP architecture
# --------------------------
# Define DGP layers
layer1 = [kernel(length=np.array([1]), name='sexp'), kernel(length=np.array([1]), name='sexp'), kernel(length=np.array([1]), name='sexp'), kernel(length=np.array([1]), name='sexp'), kernel(length=np.array([1]), name='sexp')]
layer2 = [kernel(length=np.array([1]), name='sexp', connect=np.arange(5)), kernel(length=np.array([1]), name='sexp', connect=np.arange(5)), kernel(length=np.array([1]), name='sexp', connect=np.arange(5)), kernel(length=np.array([1]), name='sexp', connect=np.arange(5)), kernel(length=np.array([1]), name='sexp', connect=np.arange(5))]
#layer3 = [kernel(length=np.array([1]), name='sexp', connect=np.arange(3)), kernel(length=np.array([1]), name='sexp', connect=np.arange(3)), kernel(length=np.array([1]), name='sexp', connect=np.arange(3))]
#layer3 = [kernel(length=np.array([1]), name='sexp', connect=np.arange(5)), kernel(length=np.array([1]), name='sexp', connect=np.arange(5)), kernel(length=np.array([1]), name='sexp', connect=np.arange(5)), kernel(length=np.array([1]), name='sexp', connect=np.arange(5)), kernel(length=np.array([1]), name='sexp', connect=np.arange(5))]

layer4 = [kernel(length=np.array([1]), name='sexp', scale_est=True, connect=np.arange(5))]

#all_layer = combine(layer1, layer2, layer3, layer4)
layers = combine(layer1, layer2, layer4)

# --------------------------
# Train DGPs for each POD coefficient
# --------------------------
models, emulators = [], []
for j in range(N_POD):
    print(f"\n=== Training DeepGP for POD coefficient {j+1} ===")
    yj = Yn[:, j].reshape(-1, 1)

    model = dgp(Xn, yj, layers)
    model.train(N=100)  # training iterations
    emu = emulator(model.estimate(), N=30)

    models.append(model)
    emulators.append(emu)

    # Predict back on training for diagnostics
    mu_pred, var_pred = emu.predict(Xn)
    mu_pred = mu_pred.flatten()
    std_pred = np.sqrt(var_pred).flatten()

    # Unnormalize
    mu_orig = mu_pred * Y_std[j] + Y_mean[j]
    std_orig = std_pred * Y_std[j]
    y_true = Y_train[:, j]

    rmse = np.sqrt(np.mean((mu_orig - y_true)**2))

    # Plot true vs predicted with 90% CI
    lower = mu_orig - 1.645*std_orig
    upper = mu_orig + 1.645*std_orig
    lower_err = mu_orig - lower
    upper_err = upper - mu_orig

    # plt.figure(figsize=(6, 4))
    # plt.errorbar(y_true, mu_orig, yerr=[lower_err, upper_err],
    #              fmt='o', alpha=0.6, label="Mean ± 90% CI")
    # mn, mx = float(np.min(y_true)), float(np.max(y_true))
    # plt.plot([mn, mx], [mn, mx], 'k--', lw=1, label="Ideal")
    # plt.xlabel("True POD coefficient")
    # plt.ylabel("Predicted")
    # plt.title(f"POD {j+1} — RMSE: {rmse:.4f}")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()

# --------------------------
# Predict on TEST inputs
# --------------------------
X_test = np.loadtxt(X_test_path, delimiter=",", skiprows=1).astype(float)
X_testn = (X_test - X_mean) / X_std

Y_mean_list, Y_std_list, Y_lo_list, Y_hi_list = [], [], [], []
for j in range(N_POD):
    mu_pred, var_pred = emulators[j].predict(X_testn)
    mu_pred = mu_pred.flatten()
    std_pred = np.sqrt(var_pred).flatten()

    # Unnormalize
    mu_orig = mu_pred * Y_std[j] + Y_mean[j]
    std_orig = std_pred * Y_std[j]

    Y_mean_list.append(mu_orig)
    Y_std_list.append(std_orig)
    Y_lo_list.append(mu_orig - 1.645*std_orig)
    Y_hi_list.append(mu_orig + 1.645*std_orig)

# Stack to arrays (N_test, N_POD)
Y_mean_arr = np.vstack(Y_mean_list).T
Y_std_arr = np.vstack(Y_std_list).T
Y_lower_arr = np.vstack(Y_lo_list).T
Y_upper_arr = np.vstack(Y_hi_list).T

# --------------------------
# Save CSVs
# --------------------------
header_line = ",".join([f"POD{i+1}" for i in range(N_POD)])
np.savetxt("Y_pred_test_Carl30_5D_dgpsi_mean.csv", Y_mean_arr, delimiter=",", header=header_line, comments='')
np.savetxt("Y_pred_test_Carl30_5D_dgpsi_std.csv", Y_std_arr, delimiter=",", header=header_line, comments='')
np.savetxt("Y_pred_test_Carl30_5D_dgpsi_lower.csv", Y_lower_arr, delimiter=",", header=header_line, comments='')
np.savetxt("Y_pred_test_Carl30_5D_dgpsi_upper.csv", Y_upper_arr, delimiter=",", header=header_line, comments='')

print("\n✅ Saved test predictions with mean, std, lower, upper percentiles.")
