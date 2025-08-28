# Extract FactoMineR wine dataset for Python validation
# This script exports the wine dataset used in fviz_famd examples

# Install and load required packages
if (!require("FactoMineR")) {
  install.packages("FactoMineR")
  library(FactoMineR)
}

if (!require("factoextra")) {
  install.packages("factoextra")
  library(factoextra)
}

# Load the wine dataset
data(wine)

# Show dataset structure
cat("Wine dataset structure:\n")
str(wine)

cat("\nDataset dimensions:", dim(wine), "\n")
cat("Column names:\n")
print(colnames(wine))

# Extract the subset used in the fviz_famd example
# Columns: 1, 2, 16, 22, 29, 28, 30, 31
selected_columns <- c(1, 2, 16, 22, 29, 28, 30, 31)
wine_subset <- wine[, selected_columns]

cat("\nSelected columns for FAMD:\n")
print(colnames(wine_subset))

cat("\nSubset structure:\n")
str(wine_subset)

cat("\nFirst few rows:\n")
print(head(wine_subset))

# Save the full dataset and subset as CSV
write.csv(wine, "wine_full_dataset.csv", row.names = FALSE)
write.csv(wine_subset, "wine_famd_subset.csv", row.names = FALSE)

# Perform FAMD on the subset for reference
res.famd <- FAMD(wine_subset, graph = FALSE)

cat("\nFAMD Results (R FactoMineR):\n")
cat("Eigenvalues:", res.famd$eig[1:5, "eigenvalue"], "\n")
cat("Explained variance (%):", res.famd$eig[1:5, "percentage of variance"], "\n")
cat("Cumulative variance (%):", res.famd$eig[1:5, "cumulative percentage of variance"], "\n")

# Save FAMD results
famd_results <- data.frame(
  component = 1:nrow(res.famd$eig),
  eigenvalue = res.famd$eig[, "eigenvalue"],
  percentage_variance = res.famd$eig[, "percentage of variance"],
  cumulative_variance = res.famd$eig[, "cumulative percentage of variance"]
)

write.csv(famd_results, "wine_famd_results_R.csv", row.names = FALSE)

# Save individual coordinates
individuals_coords <- as.data.frame(res.famd$ind$coord)
write.csv(individuals_coords, "wine_individuals_coords_R.csv", row.names = TRUE)

# Save variable coordinates
variables_coords <- as.data.frame(res.famd$var$coord)
write.csv(variables_coords, "wine_variables_coords_R.csv", row.names = TRUE)

cat("\nFiles saved:\n")
cat("- wine_full_dataset.csv (full dataset)\n")
cat("- wine_famd_subset.csv (subset for FAMD)\n") 
cat("- wine_famd_results_R.csv (FAMD eigenvalues and variance)\n")
cat("- wine_individuals_coords_R.csv (individual coordinates)\n")
cat("- wine_variables_coords_R.csv (variable coordinates)\n")

cat("\nR FactoMineR reference results generated successfully!\n")
