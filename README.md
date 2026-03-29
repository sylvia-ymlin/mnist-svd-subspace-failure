# Scientific Computing, 2023 | Project 1: SVD-Based Digit Classification

Each MNIST handwritten digit image is a 28×28 pixel grid, represented as a 784-dimensional vector. Digit classes do not fill this high-dimensional space uniformly. They concentrate near low-dimensional subspaces, and this geometric structure governs both classification success and failure.

This project investigates the following questions:

1. How does an SVD-based projection classifier work geometrically?
2. Do different digit classes differ in intrinsic dimensionality?
3. How does subspace rank $k$ affect classification accuracy?
4. Are classification errors structured or random?
5. Can the geometry between digit subspaces predict confusion patterns?


## 1. SVD Classifier

For each digit class $i$, an orthonormal basis $U_i$ is obtained via singular value decomposition (SVD) of the training data matrix $X_i \in \mathbb{R}^{784 \times 400}$:

$$X_i = U_i \Sigma_i V_i^T$$

The classifier projects a test image $d \in \mathbb{R}^{784}$ onto the subspace spanned by the first $k$ columns of $U_i$ (denoted $U_{k,i}$):

$$\hat{d}_i = U_{k,i} U_{k,i}^T d$$

The reconstruction residual, or the distance from $d$ to class $i$'s subspace, is:

$$r_i = \left\| d - U_{k,i} U_{k,i}^T d \right\|_2$$

The test image is assigned to the class with the smallest residual:

$$\hat{y} = \arg\min_{i} \; r_i$$

Geometric Interpretation: The classifier assigns each test image to the subspace it lies closest to. A small residual indicates that the image is well approximated by that class's basis. Classification fails when two subspaces are nearly aligned.


## 2. Digit Subspaces: Structure

Decomposing the training data ($X \in \mathbb{R}^{784 \times 400}$) via SVD reveals how variance distributes across directions, with faster decay indicating lower intrinsic dimensionality. 

As shown in Figure 1 and the table below, structural complexity varies by class. Digit "1" is the simplest ($k_{90\%} = 8$), owing to its consistent shape. In contrast, digit "5" is the most complex ($k_{90\%} = 33$), reflecting higher within-class variation. Optimal rank typically sits near this 90% variance threshold, capturing dominant structure while ignoring noise (§3).

<div align="center">
<img src="figures/singular_value_decay.png" width="400">

**Figure 1.** Singular value decay per digit class (log scale).
</div>

<div align="center">

| Digit | k (90%) | k (95%) |       | Digit | k (90%) | k (95%) |
| :---: | :--------: | :--------: | :---: | :---: | :--------: | :--------: |
|   1   |     8      |     16     |       |   3   |     25     |     46     |
|   0   |     14     |     30     |       |   4   |     26     |     49     |
|   6   |     18     |     34     |       |   2   |     27     |     47     |
|   7   |     18     |     36     |       |   8   |     28     |     51     |
|   9   |     19     |     37     |       |   5   |     33     |     57     |

</div>

The basis vectors $u_j$ (columns of $U$) provide an orthonormal basis for the class subspace. Figure 2 displays the first three components ($u_1, u_2, u_3$) per digit. Without pre-centering, $u_1$ captures the mean digit shape, while subsequent vectors represent dominant within-class variations like stroke slant and thickness.

<div align="center">
<img src="figures/basis_components_grid.png" width="600">

**Figure 2.** First three basis vectors $u_1, u_2, u_3$ per digit class.
</div>


## 3. Rank Selection

Subspace rank $k$ defines the number of basis vectors per class. Figure 3 evaluates testing accuracy on 10,000 images across $k \in [1, 50]$. Accuracy peaks at 95.36% ($k = 22$) before declining, as high-rank directions capture sample-specific noise rather than stable class structure. This result is consistent with Figure 1: the discriminative signal for each digit is concentrated in its low-rank directions.

<div align="center">
<img src="figures/accuracy_vs_rank.png" width="400">

**Figure 3.** Classification accuracy vs. subspace rank $k$.
</div>

Figure 4 compares original test images with their rank-22 reconstructions ($\hat{d} = U_{22} U_{22}^T d$). The reproduction of key visual features confirms that $k = 22$ captures the dominant class structure by finding the best approximation of $d$ within the 22-dimensional subspace.

<div align="center">
<img src="figures/reconstruction_comparison.png" width="600">

**Figure 4.** Original test images (top) and rank-22 reconstructions (bottom) for one example per digit class.
</div>



## 4. Confusion Matrix

Figure 5 shows the normalized confusion matrix at $k = 22$. The three largest off-diagonal entries are:

* 5→3: 3.6% confusion rate
* 8→1: 2.62% confusion rate
* 7↔9: ~2.3–2.4% in both directions

<div align="center">
<img src="figures/confusion_matrix.png" width="400">

**Figure 5.** Normalized confusion matrix at $k = 22$.
</div>

Errors are structured rather than random, with specific digit pairs consistently confused. For instance, the 8→1 confusion occurs despite visual dissimilarity, suggesting a non-pixel-level geometric cause (§5).


## 5. Subspace Geometry Between Digit Classes

Principal angles $\theta$ between subspaces $U_i$ and $U_j$ quantify their alignment:

$$\theta_\ell = \arccos\bigl(\sigma_\ell(U_{k,i}^T U_{k,j})\bigr), \quad \ell = 1, \ldots, k$$

where $\sigma_\ell$ are singular values of the cross-Gram matrix. Ordered $0 \leq \theta_1 \leq \ldots \leq 90°$, a small $\theta_1$ indicates shared directions, leading to higher confusion rates.

Figure 6 shows mean principal angles between digit pairs. Pairs with small angles - (8, 1), (5, 3), (7, 9) - match the high confusion rates (8→1, 5→3, and 7↔9) shown in Figure 5.

<div align="center">
<img src="figures/mean_principal_angles_heatmap.png" width="400">

**Figure 6.** Mean principal angle (degrees) between all pairs of digit subspaces at $k = 22$.
</div>

The correlation ($\rho = -0.67$) shown in Figure 7 indicates that classification errors are governed by subspace proximity. This suggests that the classifier's failure modes are predictable from the geometric alignment of digit classes in the training data.

<div align="center">
<img src="figures/angle_vs_confusion_mean.png" width="400">

**Figure 7.** Mean principal angle vs. confusion rate for all 45 digit pairs.
</div>

## 6. Centered SVD (PCA): Decoupling Mean from Variation

In the baseline classifier (§1), I computed the SVD directly on the raw image matrix. The resulting subspace $U_i$ captures the dominant directions of the data, which inherently includes the class mean. To isolate the unique structural variations of each digit, I implemented Centered SVD, which is equivalent to Principal Component Analysis (PCA). By subtracting the class mean $\mu_i$ before performing SVD, the resulting subspace $U_i^{PCA}$ represents only the within-class variation, effectively decoupling the static "average" shape from the dynamic variations. The two methods are compared at their respective optimal ranks.

Figure 8 shows that PCA reaches 95.67% accuracy at $k = 23$, an improvement of 0.31 percentage points over the uncentered SVD baseline.

<div align="center">
<img src="figures/pca_accuracy_vs_rank.png" width="400">

**Figure 8.** Accuracy vs. rank $k$ for uncentered SVD and centered SVD (PCA).
</div>

Figure 9 shows the confusion matrices side by side: the 8→1 confusion rate drops from 2.62% to 1.93% after centering, consistent with the hypothesis that part of the confusion is driven by shared mean image shape.

<div align="center">
<img src="figures/pca_confusion_comparison.png" width="700">

**Figure 9.** Confusion matrices for uncentered SVD (left) and centered SVD / PCA (right) at their respective optimal ranks.
</div>

Furthermore, centering increases the minimum principal angle for the 8–1 pair from 6.21° to 13.78°, effectively pulling the two subspaces apart by removing the shared mean component. This confirms that much of the initial confusion was due to the shared static baseline of the digit shapes rather than their unique structural variations. While 13.78° remains the smallest minimum angle among all 45 digit pairs after centering, the reduction in overlap qualitatively demonstrates that PCA provides more robust separation of digit classes by isolating unique structural features.



## 7. Conclusion

This project demonstrates that an SVD-based projection classifier is highly effective for MNIST, achieving up to 95.67% accuracy through PCA. Returning to the questions proposed before:

1. **How does an SVD-based projection classifier work geometrically?**
   
   *It classifies test images by projecting them onto class-specific low-dimensional subspaces and finding the minimum reconstruction residual. Its success depends entirely on whether digit classes occupy distinct, non-overlapping geometric regions.*

2. **Do different digit classes differ in intrinsic dimensionality?**
   
   *Yes. Simple, structurally consistent digits (like "1") saturate their variance around $k \approx 10$, while geometrically complex digits with higher within-class variation (like "5" and "8") require $k \geq 20$.*

3. **How does subspace rank $k$ affect classification accuracy?**
   
   *Accuracy peaks at $k=22$. Below this, the model underfits core structural modes. Above $k=22$, subspace directions overfit to sample-specific noise rather than stable class structure, causing performance degradation.*

4. **Are classification errors structured or random?**
   
   *Errors are highly structured. Specific pairs like 5→3 and 8→1 dominate the failure cases, revealing that misclassifications stem from inherent geometric proximity rather than random pixel-level noise.*

5. **Can the geometry between digit subspaces predict confusion patterns?**
   
   *Yes. Confusion rates correlate strongly ($\rho = -0.67$) with mean principal angles between subspaces. Furthermore, decoupling the structural variations via PCA (Centered SVD) demonstrates that removing shared mean shapes separates heavily confused pairs (e.g., increasing the 8–1 minimum angle from 6.21° to 13.78°).*


## Quick Start

### 1. Environment Setup

Create and activate a new Conda environment (`python=3.10` recommended):

```bash
conda create -n mnist-svd python=3.10
conda activate mnist-svd
```

Install the project in editable mode (this automatically installs all dependencies from `pyproject.toml`):

```bash
pip install -e .
```

### 2. Prepare Data

Download the MNIST data and save the arrays to the `data/` directory:

```text
data/TrainDigits.npy   # shape (784, 60000)
data/TrainLabels.npy   # shape (60000,)
data/TestDigits.npy    # shape (784, 10000)
data/TestLabels.npy    # shape (10000,)
```

### 3. Run Analysis

Execute the scripts in order. All generated visualizations will be saved to the `figures/` directory.

```bash
python src/data_preparation.py      # L2-normalize images
python src/svd_basis.py             # singular values, basis images, reconstruction
python src/classifier.py            # accuracy vs rank, confusion matrix
python src/subspace_geometry.py     # principal angle heatmap, correlation
python src/centered_svd.py          # PCA comparison
```
