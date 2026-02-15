from __future__ import annotations

from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set seaborn style
sns.set_theme(style="whitegrid")


def create_star_plot(
    output_dir: str | Path = "./output_plots",
    filename: str = "star_plot.png",
    show: bool = True,
) -> None:
    """Create a radar/star plot comparing single vs multiframe model performance."""
    
    data = {
        "single": {
            "material understanding": 0.32,
            "mechanics": 0.38,
            "spatial reasoning": 0.33,
            "viewpoint": 0.30
        },
        "multiframe": {
            "material understanding": 0.34,
            "mechanics": 0.32,
            "persistence": 0.28,
            "spatial reasoning": 0.38,
            "temporal": 0.23,
            "viewpoint": 0.35
        }
    }
    
    # Get all unique categories (sorted for consistency)
    all_categories = sorted(set(
        list(data["single"].keys()) + list(data["multiframe"].keys())
    ))
    
    # Number of variables
    num_vars = len(all_categories)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle
    
    # Initialize the plot with seaborn styling
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Use seaborn color palette
    colors = sns.color_palette("husl", 2)
    color_map = {
        'single': colors[0],
        'multiframe': colors[1]
    }
    
    # Plot data for each group
    for idx, (group_name, values_dict) in enumerate(data.items()):
        # Get values in the same order as categories
        # Use NaN for missing categories (temporal for single)
        values = [values_dict.get(cat, np.nan) for cat in all_categories]
        values_plot = values + values[:1]  # Complete the circle
        
        # Plot line (will break at NaN points)
        ax.plot(angles, values_plot, 'o-', linewidth=2.5, label=group_name.capitalize(), 
                color=color_map[group_name], markersize=10, alpha=0.9)
        
        # Fill area (handles NaN by not filling those sections)
        ax.fill(angles, values_plot, alpha=0.2, color=color_map[group_name])
    
    # Fix axis to go clockwise and start at 12 o'clock
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis labels with better formatting
    ax.set_xticks(angles[:-1])
    formatted_labels = [cat.replace(' ', '\n').title() for cat in all_categories]
    ax.set_xticklabels(formatted_labels, size=12, fontweight='bold')
    
    # Set y-axis limits and labels
    ax.set_ylim(0, 0.5)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels(['0.1', '0.2', '0.3', '0.4', '0.5'], size=10)
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=1)
    
    # Customize radial gridlines
    ax.spines['polar'].set_linewidth(1.5)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=13, 
               frameon=True, shadow=True, fancybox=True)
    
    plt.title("Model Performance by Category", size=16, fontweight='bold', pad=30)
    
    # Save figure
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f_out = out_dir / filename
    fig.savefig(f_out, dpi=300, bbox_inches="tight")
    print(f"Star plot saved to: {f_out}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    create_star_plot(
        output_dir="./output_plots",
        filename="star_plot.png",
        show=True
    )
