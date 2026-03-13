from __future__ import annotations

from pathlib import Path


# Nord-inspired color palette used across all figures.
NBody_Palette = {
    "gray_light": "#D8DEE9",
    "gray_dark":  "#4C566A",
    "blue_light": "#88C0D0",
    "blue_mid":   "#81A1C1",
    "blue_deep":  "#5E81AC",
    "orange":     "#D08770",
    "red":        "#BF616A"
}

def set_plot_theme() -> None:
    import matplotlib as mpl
    import seaborn as sns
    from cycler import cycler

    sns.set_theme(style="white", context="paper", font="DejaVu Sans")
    
    # Set default color cycle using NBody Palette
    default_cycler = cycler(color=[
        NBody_Palette["blue_deep"],
        NBody_Palette["orange"],
        NBody_Palette["red"],
        NBody_Palette["blue_light"],
        NBody_Palette["gray_dark"]
    ])

    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelweight": "regular",
            "axes.prop_cycle": default_cycler,
            "axes.labelcolor": NBody_Palette["gray_dark"],
            "xtick.color": NBody_Palette["gray_dark"],
            "ytick.color": NBody_Palette["gray_dark"],
            "text.color": NBody_Palette["gray_dark"],
            "legend.frameon": False,
        }
    )


def save_figure(fig, filename: str, out_dirs: list[str | Path] | None = None) -> list[Path]:
    """Save a figure to one or more output directories. Defaults to figures/."""
    if out_dirs is None:
        out_dirs = [Path("figures")]

    written: list[Path] = []
    for d in out_dirs:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        path = d / filename
        fig.savefig(path)
        written.append(path)
    return written

