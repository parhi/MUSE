# MUSE
MATLAB implementation of MUSE (Minimum Uncertainty and Sample Elimination) feature selection, with reproducible toy experiments and comparison against mRMR for binary classification.

MUSE iteratively:
1. Discretizes each feature into K equal-probability (quantile) bins (computed on the current surviving samples).
2. Computes an uncertainty score for each candidate feature using only the most certain bins covering > p probability mass:
   - For each bin, compute conditional entropy H(class | bin).
   - Sort bins by increasing entropy.
   - Take the smallest set of bins whose cumulative probability mass exceeds p.
   - Score: J(X_i) = sum over bins of P(B_k) * H(class | B_k). Lower is better.
3. Selects the feature with the minimum uncertainty score.
4. Performs sample elimination: bins with impurity < T (where impurity = min(P0,P1)) are considered “good”; samples in those bins are discarded.
5. Repeats until m features are selected or a stopping rule triggers.


---

## Files

- `MUSE.m`  
  Main function implementing MUSE. Includes the helper discretization routine as a local function (`equalprob_bins`).

- `MUSE_demo.m`  
  MATLAB demo script that generates a reproducible toy dataset, runs MUSE and mRMR feature selection, and evaluates classification performance using cross-validation.

- `MUSE_demo.pdf`  
  Published (PDF) version of `MUSE_demo.m`, containing the code, figures, and results in a notebook-style, human-readable format.
  
---

## Requirements

- MATLAB R2018b+ recommended (uses `discretize` and `quantile`).
- Statistics and Machine Learning Toolbox for running `MUSE_demo.m`

---

## Quick start

```matlab
% X: N x D double
% y: N x 1 labels (0/1, -1/+1, or any two numeric values)

m  = 10;    % select up to 10 features
K  = 20;    % number of bins for discretization
p  = 0.2;   % mass of "best bins" used in uncertainty score
T  = 0.1;   % impurity threshold for elimination
Ts = 0.01;  % stop if either class survival fraction < Ts (set 0 to disable)

[selected_fea, J_history, elim_frac, survivors_mask] = MUSE(X, y, m, K, p, T, Ts);

disp(selected_fea)
plot(J_history, '-o'); grid on; xlabel('Iteration'); ylabel('J(X)');
```

---

## Output interpretation

- `selected_fea`  
  Indices (columns of `X`) chosen by MUSE, in selection order.

- `J_history`  
  The uncertainty score of each selected feature (lower is “better”).

- `elim_frac`  
  Fraction of samples eliminated at each iteration (of the current surviving set).

- `survivors_mask`  
  `N x (k+1)` logical matrix tracking which samples remain after each iteration.
  - Column 1: all samples (initially true)
  - Column t+1: survivors after selecting feature t and eliminating “easy” samples
  - This is useful for debugging and for analyzing which observations your selected features separate early.

---

## Parameter tips (for EEG/PSD features)

- `K`: typically 10–30. Too large → sparse bins and unstable entropy estimates.
- `p`: 0.1–0.3 is common. Smaller p focuses more strongly on the “best” regions.
- `T`: 0.05–0.2. Smaller T eliminates only very pure bins; larger T eliminates more aggressively.
- `Ts`: 0.01–0.1. Use >0 if you want to avoid collapsing one class too quickly.

---

## Notes

- This implementation is for binary labels only.
- For large feature sets, parallelize the feature loop (`parfor`).

---

## Cite:

[1] Z. Zhang and K. K. Parhi, "MUSE: Minimum Uncertainty and Sample Elimination Based Binary Feature Selection," in IEEE Transactions on Knowledge and Data Engineering, vol. 31, no. 9, pp. 1750-1764, 1 Sept. 2019, doi: [10.1109/TKDE.2018.2865778.](https://doi.org/10.1109/TKDE.2018.2865778) 

[2] S. S. Balaji et al, "Patient-specific long-term seizure prediction via multi-model classification." Journal of neural engineering 22.6 (2025): 066011., doi: [10.1088/1741-2552/ae1875](https://doi.org/10.1088/1741-2552/ae1875)
