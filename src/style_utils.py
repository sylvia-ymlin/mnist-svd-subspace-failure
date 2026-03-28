import matplotlib.pyplot as plt
import matplotlib as mpl

# =============================================================================
# GLOBAL STYLE CONFIGURATION
# Nord-inspired color palette
# =============================================================================

# Nord Palette
PALETTE = {
    "gray_light": "#D8DEE9",
    "gray_dark": "#4C566A",
    "blue_light": "#88C0D0",
    "blue_mid": "#81A1C1",
    "blue_deep": "#5E81AC",
    "orange": "#D08770",
    "red": "#BF616A",
}

STYLE = {
    "colors": {
        "primary": PALETTE["blue_deep"],      # Main data visualization
        "secondary": PALETTE["orange"],        # Emphasis/highlight
        "accent": PALETTE["blue_light"],       # Accent/additional data
        "text": PALETTE["gray_dark"],          # Text color
        "grid": PALETTE["gray_light"],         # Grid background
    },
    "figsize_single": (7, 4.5),       # line plots, scatter plots
    "figsize_square": (6.5, 6),       # heatmaps with colorbar
    "figsize_bar":    (5, 4),         # compact bar / column charts
    "figsize_portrait": (7, 9),       # multi-subplot vertical stack
    "figsize_pair":   (13, 5.5),      # two square plots side by side
    "figsize_wide":   (15, 5),        # image grid — 3 rows
    "figsize_wide_short": (15, 3.5),  # image grid — 2 rows
}

def apply_global_style():
    """Apply unified Matplotlib styling for all project figures."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.labelcolor": STYLE["colors"]["text"],
        "text.color": STYLE["colors"]["text"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": PALETTE["gray_dark"],
        "axes.grid": True,
        "grid.color": PALETTE["gray_light"],
        "grid.alpha": 0.4,
        "grid.linewidth": 0.5,
        "legend.framealpha": 0.95,
        "legend.edgecolor": PALETTE["gray_dark"],
        "legend.facecolor": "#ECEFF4",  # Very light background
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "figure.dpi": 150,
        "savefig.dpi": 150,
    })
