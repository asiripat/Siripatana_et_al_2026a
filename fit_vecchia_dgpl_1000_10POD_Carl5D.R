library(deepgp)

# ------------------------
# I/O
# ------------------------
X_path <- "sample_data/X_input_Carl180_5D.csv"
Y_path <- "sample_data/Y_pod_coefficients_Carl180_5D.csv"
X_test_path <- "sample_data/combined_nodes_dat_MC_Carl_5D.csv"

# Load CSVs
X_full <- as.matrix(read.csv(X_path, header = TRUE))
Y_full <- as.matrix(read.csv(Y_path, header = TRUE))
X_test <- as.matrix(read.csv(X_test_path, header = TRUE))

# ------------------------
# Train/test split
# ------------------------
N_TRAIN <- 180
N_TEST  <- 1000

X_train <- X_full[1:N_TRAIN, ]
Y_train <- Y_full[1:N_TRAIN, ]

# ------------------------
# Normalization
# ------------------------
X_mean <- colMeans(X_train)
X_sd   <- apply(X_train, 2, sd)
X_sd[X_sd == 0] <- 1

Xn_train <- sweep(sweep(X_train, 2, X_mean, "-"), 2, X_sd, "/")
Xn_test  <- sweep(sweep(X_test,  2, X_mean, "-"), 2, X_sd, "/")
Xn_test  <- Xn_test[1:N_TEST, , drop = FALSE]

Y_mean <- colMeans(Y_train)
Y_sd   <- apply(Y_train, 2, sd)
Y_sd[Y_sd == 0] <- 1
Yn_train <- sweep(sweep(Y_train, 2, Y_mean, "-"), 2, Y_sd, "/")

# ------------------------
# Loop over all POD modes
# ------------------------
num_modes <- ncol(Y_train)

for (target_index in 1:num_modes) {
  cat("\n=== Training Vecchia-DGP for POD", target_index, "of", num_modes, "===\n")
  
  # Select target POD coefficient
  yi <- as.numeric(Yn_train[, target_index])
  
  t1 <- Sys.time()
  
  # Fit Vecchia DGP
  fit <- fit_two_layer(
    Xn_train, yi,
    nmcmc = 2000,
    vecchia = TRUE,
    m = 60
  )
  
  t2 <- Sys.time()
  
  # Predict on test set
  pred <- predict(fit, Xn_test, lite = TRUE)
  
  t3 <- Sys.time()
  
  cat("Training time:", round(difftime(t2, t1, units="secs"), 2), "sec\n")
  cat("Prediction time:", round(difftime(t3, t2, units="secs"), 2), "sec\n")
  
  mean_vec <- drop(pred$mean) * Y_sd[target_index] + Y_mean[target_index]
  std_vec  <- drop(sqrt(pred$s2)) * Y_sd[target_index]
  
  # Save outputs
  output_mean <- data.frame(mean_vec)
  output_std  <- data.frame(std_vec)
  
  colnames(output_mean) <- paste0("POD", target_index, "_mean")
  colnames(output_std)  <- paste0("POD", target_index, "_std")
  
  write.csv(output_mean,
            paste0("Y_pred_test_POD_Carl_5D_Gramacy180sam_mode", target_index, "_", N_TEST, "_mean_m60.csv"),
            row.names = FALSE)
  
  write.csv(output_std,
            paste0("Y_pred_test_POD_Carl_5D_Gramacy_180sam_mode", target_index, "_", N_TEST, "_std_m60.csv"),
            row.names = FALSE)
  
  cat("✅ Predictions for POD", target_index, "saved to CSV (first", N_TEST, "test cases).\n")
}

cat("\n🎯 All POD modes processed successfully.\n")
