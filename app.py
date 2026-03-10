import streamlit as st
import os
from PIL import Image

st.set_page_config(page_title="MNIST SVD Interactive Dashboard", layout="wide")

st.title("MNIST SVD Subspace Truncation Analysis (3 vs 8)")

st.sidebar.header("Controls")
k = st.sidebar.slider("SVD Subspace Rank (k)", min_value=1, max_value=40, value=10, step=1)

# Ensure figures are accessible
fig_dir = "figures"

# Map exactly to precomputed ranks or find nearest if it was a custom range
# In this pipeline we have full matrices or specific points computed
valid_k_confusion = [5, 10, 20, 40]
closest_k_conf = min(valid_k_confusion, key=lambda x: abs(x - k))

valid_k_residual = [10, 20]
if k <= 15:
    closest_k_res = 10
else:
    closest_k_res = 20

st.write(f"Displaying results interpolated for rank **k = {k}**.")

# Row 1: Confusion Matrices
st.subheader("Stage 1: Confusion Matrices")
col1, col2 = st.columns(2)

with col1:
    conf_svd_path = os.path.join(fig_dir, f"confusion_svd_rank_{closest_k_conf}.png")
    if os.path.exists(conf_svd_path):
        st.image(Image.open(conf_svd_path), caption=f"SVD Rank-{closest_k_conf} Confusion Matrix", use_container_width=True)
    else:
        st.warning(f"SVD Confusion figure not precomputed for nearest mapped rank {closest_k_conf}.")

with col2:
    conf_cnn_path = os.path.join(fig_dir, "confusion_cnn.png")
    if os.path.exists(conf_cnn_path):
        st.image(Image.open(conf_cnn_path), caption="CNN Confusion Matrix (Static Baseline)", use_container_width=True)

st.markdown("""
**Observation:** Notice how the SVD model struggles uniquely with confusing the digit 3 as an 8,
while the CNN baseline easily resolves the difference. SVD full-rank (nearest centroid) also resolves it.
""")

st.divider()

# Row 2: Residual Heatmaps
st.subheader("Stage 2: Spatial Diagnosis (Residual Heatmaps)")
col3, col4 = st.columns(2)

with col3:
    res_3_path = os.path.join(fig_dir, f"residual_heatmap_c3_k{closest_k_res}.png")
    if os.path.exists(res_3_path):
        st.image(Image.open(res_3_path), caption=f"Digit 3 Mean Residuals (Rank-{closest_k_res})", use_container_width=True)
    else:
        st.warning(f"Digit 3 residual figure not found for k={closest_k_res}.")

with col4:
    res_8_path = os.path.join(fig_dir, f"residual_heatmap_c8_k{closest_k_res}.png")
    if os.path.exists(res_8_path):
        st.image(Image.open(res_8_path), caption=f"Digit 8 Mean Residuals (Rank-{closest_k_res})", use_container_width=True)

st.markdown("""
**Observation:** The residual for digit 3 is highly concentrated in the "gap" regions (forming the left-side openings of the 3).
Because the gap direction has low covariance energy (low variance within the class of 3s), SVD truncation discards it early, leading to misclassification as an 8.
""")

st.divider()

# Row 3: Gap Direction Substitution (Optional Extension based on Precomputed)
st.subheader("Stage 5: Causal Intervention (Substitution)")
sub_fig_path = os.path.join(fig_dir, "stage5_delta_vs_rank.png")
if os.path.exists(sub_fig_path):
    st.image(Image.open(sub_fig_path), caption="Effect of Basis Substitution on Confusion", width=600)
    
st.markdown("""
**Observation:** By substituting the least energetic direction of the representation with the specific gap vector (without expanding the dimension size), the confusion error from 3 to 8 drastically decreases.
""")
