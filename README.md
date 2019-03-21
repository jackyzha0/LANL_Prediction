# LANL Earthquake Prediction
https://www.kaggle.com/c/LANL-Earthquake-Prediction

### Goal
To predict when a fault will happen given acoustic data

### Things to try
 - Normalize to [0, 1]
 - LSTM
 - GPR (Gaussian Process Regression) coefficients
  - Diff between high / low
 - Standard statistical measures for frames of actual data & GPR vals
  - Quantiles (0.01, 0.05, 0.95, 0.99)
  - Mean
  - Standard Deviation
  - Absolute of the Median
  - Kurtosis
 - MFCCs
 - 2nd and 3rd Derivatives of all

### Frame Details
 - Sizes 50k, 10k, 5k, 1k
 - Stride
