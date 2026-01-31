# Marrel et al. (2010) Campbell2D pipeline (Python reproduction)

This pipeline reproduces the spatial Sobol' index pipeline described in:
Amandine Marrel, Bertrand Iooss, Michel Jullien, Beatrice Laurent, and Elena Volkova. Global
sensitivity analysis for models with spatially dependent outputs. Environmetrics, 22(3):383–397, 2010. doi: 10.1002/env.1071. URL https://doi.org/10.1002/env.1071.

## What is reproduced (Campbell2D study)

* Campbell2D function (Eq. (6)) on a 64x64 grid (nz=4096)
* Learning design: maximin LHS with n=200
* Wavelet decomposition (Daubechies; implemented as db4 via PyWavelets)
* Coefficient selection by decreasing empirical variance (Eq. (9))
* Method 3 in Step 2:
  - GP for k*=30 coefficients
  - Linear regression for k' = 500 coefficients
  - Remaining coefficients set to empirical mean
* Diagnostics: MSE and Q2
* Sobol maps from the functional metamodel using the nested MC procedure in Sec. 3.4
* Reference ("exact") Sobol maps computed using Saltelli MC on the true simulator (N=1e5 by default)

## How to run

Open `notebooks/01_run_pipeline.ipynb` and run all cells.

## Unit tests

`pytest -q` from the project root.

## Notes on matching the paper

Marrel et al. use a specialized GP fitting procedure (Marrel et al., 2008) with AICc-based trend selection and a generalized exponential correlation. Here we use scikit-learn GPs with a Matern kernel and a lightweight AIC subset selection for linear models. The tests therefore compare key reported aggregate diagnostics (Table 1 rMAE values) within tolerance.


## Notebook import notes

If you open the notebook from the `notebooks/` folder, it prepends the project root to `sys.path` so `import marrel_pipeline` works without relative imports.


## Additional notebooks

- `notebooks/02_run_frk_pipeline.ipynb`: FRK v2-style (Python analog) surrogate pipeline.
- `notebooks/03_compare_wavelet_vs_frk.ipynb`: comparison plots and rMAE table.
