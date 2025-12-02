import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Read CSV data
vram = []
cost = []
providers = []
gpus = []

with open('output.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cost_val = float(row['cost_per_hour'])
        # Filter out 0 costs
        if cost_val > 0:
            vram.append(float(row['vram']))
            cost.append(cost_val)
            providers.append(row['provider'])
            gpus.append(row['gpu'])

# Create color map for providers
provider_colors = {
    'Vast.AI': '#FF6B6B',
    'RunPod': '#4ECDC4',
    'Lambda Labs': '#FFE66D'
}

colors = [provider_colors[p] for p in providers]

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

# Create scatter plot
scatter = ax.scatter(vram, cost, c=colors, s=150, alpha=0.7, edgecolors='black', linewidth=1.5)

# Fit linear regression
vram_arr = np.array(vram)
cost_arr = np.array(cost)
slope, intercept, r_value, p_value, std_err = stats.linregress(vram_arr, cost_arr)

# Plot regression line
x_line = np.array([vram_arr.min(), vram_arr.max()])
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, 'k--', alpha=0.5, linewidth=2, label=f'Linear fit (R²={r_value**2:.3f})')

# Add cost per GB label on the line
mid_x = (x_line[0] + x_line[1]) / 2
mid_y = slope * mid_x + intercept
ax.annotate(f'${slope:.4f}/GB',
            xy=(mid_x, mid_y),
            xytext=(10, -15),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='black'))

# Labels and title
ax.set_xlabel('VRAM per GPU (GB)', fontsize=12, fontweight='bold')
ax.set_ylabel('Cost per Hour (USD)', fontsize=12, fontweight='bold')
ax.set_title('GPU Rental Pricing Comparison', fontsize=14, fontweight='bold', pad=20)

# Add grid for better readability
ax.grid(True, alpha=0.3, linestyle='--')

# Create legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=provider_colors[p], edgecolor='black', label=p)
                   for p in sorted(provider_colors.keys())]
legend_elements.append(ax.get_lines()[0])
ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)

# Add annotations for interesting points
# Find min cost, max vram, best value (lowest cost/vram ratio)
min_cost_idx = np.argmin(cost)
max_vram_idx = np.argmax(vram)
with np.errstate(divide='ignore', invalid='ignore'):
    value_ratio = np.where(vram_arr > 0, cost_arr / vram_arr, np.inf)
best_value_idx = np.argmin(value_ratio)

for idx, label in [(min_cost_idx, 'Cheapest'), (best_value_idx, 'Best Value')]:
    label_color = provider_colors[providers[idx]]
    ax.annotate(gpus[idx],
                xy=(vram[idx], cost[idx]),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=label_color, alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Label the MI300X
mi300x_idx = next((i for i, g in enumerate(gpus) if 'MI300X' in g), None)
if mi300x_idx is not None:
    label_color = provider_colors[providers[mi300x_idx]]
    ax.annotate('MI300X\n$1.99/192GB',
                xy=(vram[mi300x_idx], cost[mi300x_idx]),
                xytext=(20, -30),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=label_color, alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Tight layout
plt.tight_layout()

# Save and show
plt.savefig('gpu_pricing.png', dpi=150, bbox_inches='tight')
print("Plot saved to gpu_pricing.png")
plt.show()
