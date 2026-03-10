# MNIST Linear vs Nonlinear: Why does SVD confuse 3 and 8?

## Motivation
SVD-based subspace classifiers are interpretable, efficient, and require no gradient-based backpropagation for training. However, they exhibit systematic failure modes that are not well understood when acting on spatial image data. This project investigates one such failure mode: why rank-k SVD classifiers consistently confuse digit 3 and digit 8 in the MNIST dataset, while retaining competency elsewhere. We evaluate mechanisms related to either directional suppression or intrinsic dimensionality shortcomings.

## Method

**Models Evaluated:**
- **SVD rank-k**: Truncates subspace representation to top $k$ principal vectors.
- **SVD full-rank**: Uses the nearest centroid approach on the un-truncated projection space.
- **Logistic Regression & CNN**: Discriminative baselines ensuring this failure mode is linear-centric.

**Experimental Design:**
- **Stage 1 (Confusion)**: Establishes that the 3↔8 error is uniquely exaggerated under SVD rank-k constraints.
- **Stage 2 (Spatial Diagnosis)**: Computes and plots the mean spatial residuals between raw data and subspace approximations to see exactly where reconstruction fails.
- **Stage 3 (Centered Spectrum)**: Compares normalized cumulative energy curves. *Decision: Centering prevents SVD from conflating the global mean with variance.*
- **Stage 4a (Projection Energy)**: Tests if the specific feature gap direction in digit 3 is uniquely suppressed by computing geometric projection capabilities on those distinct gap vectors.
- **Stage 5 (Basis Substitution)**: Tests causality. Substitutes the lowest energy component with the gap vector while retaining equal dimensions and comparing classifier prediction improvements. *Decision: equal-dimension substitution enforces that simple "adding capacity" is not the resolution factor.*

## Results

1. **Stage 1 (Confusion Analysis):** The CNN baseline and full-rank SVD do not confuse 3 and 8 at the scale that rank-truncated SVD does (especially when $k \in \{10, 20\}$). This indicates the problem is purely one of rank truncation, rather than inherently non-separable geometries.
2. **Stage 2 (Spatial Diagnosis):** Spatial analysis of the errors reveals that the residual map for digit 3 concentrates heavily around the "gap" regions in the shape. The model fails locally, not globally.
3. **Stage 3 (Centered Spectrum Overview):** The intrinsic dimensionality spectrum for 3 and 8 is relatively analogous (e.g. at $k=20$, $68.20\%$ energy encoded for 3 and $65.30\%$ for 8). The failure is not due to digit 3 being generally under-represented or "harder to memorize"; there is no requirement for broadly more rank. Hypothesis B is ruled out.
4. **Stage 4a (Gap Direction Projection):** $\rho_3$ grows more slowly with rank than $\rho_8$ and remains consistently lower, indicating the gap direction is systematically deprioritized — partially represented but not sufficiently so for reliable discrimination.
5. **Stage 5 (Basis Substitution & Principal Angles):** When substituting the lowest-energy basis vector for the unrepresented gap direction, the 3↔8 errors actually *increased* (e.g., from 31 to 61 errors at $k=5$). By measuring the Principal Angles between the original and substituted subspaces, we found this intervention rotated the manifold by 18° at $k=5$ (and 9° at $k=10$). A rotation of 18° in a 784-dimensional space represents a substantial reorientation of the subspace, sufficient to shift projection residuals for the majority of digit 3 samples. This indicates that while the gap is a critical missing feature, forcibly orienting the linear subspace to include it massively disrupts the existing delicate variance balance SVD uses to separate 3 from 8 overall.

## Conclusion

The 3↔8 failure mode in rank-k SVD classifiers is a case of **directional suppression** (Hypothesis A). The gap features dictating a 3 instead of an 8 carry high discriminative capacity but low general intra-class variance compared to basic bulk stroke features. Standard SVD systematically trims this vector, prioritizing broadly prevalent energy directions.

Crucially, **why does the CNN succeed where SVD (even modified SVD) fails?** SVD is structurally forced to balance the entire class manifold linearly; as Stage 5 showed, manually prioritizing a low-variance but highly discriminative feature (the gap) rotates the subspace geometry, reducing the total projection energy captured from digit 3 samples and narrowing the residual margin that separates them from digit 8. A CNN, by contrast, learns a non-linear decision boundary. It can act as a local feature detector—becoming hyper-sensitive to the localized gap region without needing to rigidly tether that sensitivity to the global variance maximization that restrains linear subspace models.

**Note on Constraints:** This project does not holistically declare that SVD universally fails on topological characteristics or other feature pairs. It systematically limits scope to tracing the exact reason this particular digit pair breaks under SVD bounds.
