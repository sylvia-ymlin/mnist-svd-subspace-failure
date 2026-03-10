#Research Design &Implementation Plan

##Research Question

            Why does a rank -
        k SVD subspace classifier systematically confuse digit 3 and
    digit 8 in MNIST,
    and what geometric property of the representation causes this failure
    ?

    -- -

       ##Hypotheses

           Two competing mechanistic explanations,
    both testable :

    **Hypothesis A — Directional suppression **The gap structure in
            digit 3 corresponds to a low -
        energy direction in the data matrix.SVD truncation discards it because
            it carries small singular values,
    not because digit 3 is intrinsically harder to represent.

            * *Hypothesis B — Intrinsic dimensionality *
            *Digit 3 has higher intra -
        class variance than digit 8. The class manifold
          requires higher
        rank to represent adequately; the gap appears in the residual as a consequence of general under-representation, not directional suppression.

These are distinguishable by experiment. The results may support one, both, or neither.

---

## Experiment Structure

### Stage 1 — Establish the phenomenon (must-do)

**Experiment 1: Confusion analysis**

Build SVD rank-k classifiers for k ∈ {5, 10, 20, 40} and a CNN baseline.
Report full confusion matrices. Confirm 3↔8 is the dominant failure mode for SVD,
and that CNN does not exhibit the same failure.

Also build:
- SVD full-rank classifier (nearest centroid) — isolates the effect of rank truncation specifically
- Logistic regression — discriminative baseline

| Model | Accuracy | 3→8 errors | 8→3 errors |
|---|---|---|---|
| SVD rank-k | | | |
| SVD full-rank | | | |
| Logistic regression | | | |
| CNN | | | |

Key comparison: SVD rank-k vs SVD full-rank isolates truncation.
If full-rank SVD also confuses 3/8, the problem is not rank but the subspace model itself.

---

### Stage 2 — Spatial diagnosis (must-do)

**Experiment 2: Residual heatmap**

For each class i, compute the mean residual map across all training samples:

    R_k = mean over x in class_i of (x - U_k U_k^T x)

Plot R_k as a spatial heatmap for digit 3 and digit 8.

**Prediction under H_A**: R_k for digit 3 concentrates spatially at the gap region.
**Prediction under H_B**: R_k for digit 3 is diffuse, not localized.

This is the most interpretable figure in the project. It directly answers whether
SVD's failure is local (gap-specific) or global (general under-representation).

---

### Stage 3 — Distinguish H_A from H_B (must-do)

**Experiment 3: Centered spectrum comparison**

For each class, center the data: X̃_i = X_i - mean(X_i)
Compute SVD of X̃_i and plot the normalized cumulative energy curve:

    f_i(k) = sum(sigma_1..k)^2 / sum(sigma_all)^2

**If digit 3's curve decays slower** (flatter spectrum after centering):
→ supports H_B. Digit 3 has higher intrinsic dimensionality.

**If curves are similar**:
→ supports H_A. The residual concentration in Exp. 2 comes from directional suppression,
not from general rank insufficiency.

Note: centering is essential. Uncentered SVD conflates mean structure with variance structure;
the first singular vector correlates with the class mean and does not reflect intra-class variation.

---

### Stage 4 — Geometric diagnosis (optional, do if Stage 2-3 support H_A)

**Experiment 4a: Gap direction projection energy**

Define v as the unit vector supported on the gap pixels of digit 3 (binary mask, normalized).
Compute:

    rho = || U_k U_k^T v ||^2

This measures how much of the gap direction is captured by the rank-k subspace.

Compare rho for digit 3's subspace vs digit 8's subspace for an analogous local direction.
If rho_3 << rho_8, the gap direction is specifically poorly represented.

**Experiment 4b: Subspace angle matrix (optional)**

Compute principal angles between all pairs of digit subspaces using
scipy.linalg.subspace_angles. Plot as 10x10 heatmap.

Check: is distance(subspace_3, subspace_8) the smallest among all pairs?
This contextualizes the 3↔8 confusion within the full digit set.

---

### Stage 5 — Intervention (optional, do only if 4a confirms low rho)

**Experiment 5: Equal-dimension basis substitution**

Construct a modified subspace for digit 3:

    U_k^+ = orth({
  u_1, ..., u_ { k - 1 }} ∪ {v_perp})

where v_perp = (v - U_k U_k^T v) / ||v - U_k U_k^T v|| 
is the component of v orthogonal to the existing subspace.

This replaces the least energetic basis vector with the gap direction,
holding subspace dimension constant (dimension asymmetry controlled).

Re-run classification and measure change in 3↔8 confusion rate.

**Interpretation**:
- If confusion decreases: gap direction was causally responsible.
- If confusion does not change: the gap direction, while poorly represented,
  is not the bottleneck for classification.

Do not run the pixel-level mask removal experiment.
It changes label semantics and class geometry simultaneously — not a controlled intervention.

---

## Repository Structure

```
src/
    svd_classifier.py       # SVD rank-k and full-rank classifiers
    cnn_model.py
    baselines.py            # LR, kNN
    subspace_analysis.py    # principal angles, projection energy
    spectrum_analysis.py    # centered SVD, cumulative energy curves

experiments/
    01_confusion.py
    02_residual_heatmap.py
    03_centered_spectrum.py
    04_gap_direction.py     # optional
    05_substitution.py      # optional

figures/                    # all outputs saved here
app.py                      # Streamlit dashboard (optional)
```

---

## Execution Order and Decision Points

```
Run Exp. 1 (confusion)
    └── Confirms 3↔8 as primary failure mode?
        No → reconsider the digit pair
        Yes ↓

Run Exp. 2 (residual heatmap) + Exp. 3 (centered spectrum) in parallel

    Exp. 2: residual localized at gap?    Exp. 3: digit 3 spectrum flatter?
    Yes + No  →  H_A supported            →  run Exp. 4a, 5
    No  + Yes →  H_B supported            →  conclude: rank insufficient
    Yes + Yes →  both contribute          →  run Exp. 4a, report dual cause
    No  + No  →  neither hypothesis fits  →  re-examine assumptions
```

---

## Narrative (to be confirmed by data)

The project tells one of the following stories depending on results:

**If H_A**: SVD truncation discards the gap direction because it carries low singular value energy within the digit 3 data matrix. The gap is a consistent but low-contrast local feature — it is discriminative but not high-variance. CNN learns local nonlinear features that preserve this structure regardless of its energy rank.

**If H_B**: Digit 3's shape variability is intrinsically higher-dimensional. Any fixed rank-k budget is insufficient for adequate representation, and the gap region absorbs residual error. The fix is more capacity, not different directions.

**If both**: Two compounding factors — rank insufficiency and directional suppression — explain the confusion. Their individual contributions are quantified by Exp. 3 and Exp. 4a respectively.

---

## What this project does NOT claim

- It does not claim SVD systematically fails on topological features in general.
  That would require validation across multiple digit pairs and datasets.
- It does not claim CNN is better because it captures topology.
  CNN is better because it learns discriminative local features;
topology is a description of why those features are informative,
    not an explanation of the model's mechanism. -
            The conclusions are scoped to this digit pair and
        this model class.