import matplotlib.pyplot as plt
import numpy as np


import sys
sys.path.insert(1, '../source/')
import plot_utils as putils

# ---- Journal‐style settings ----
# plt.rcParams.update({
#     'font.family':       'serif',
#     'font.size':         12,
#     'mathtext.fontset':  'cm',
#     'axes.linewidth':    1.2,
#     'xtick.direction':   'in',
#     'ytick.direction':   'in',
#     'xtick.top':         False,
#     'ytick.right':       False,
#     'xtick.major.size':  6,
#     'xtick.major.width': 1.2,
#     'ytick.major.size':  6,
#     'ytick.major.width': 1.2,
#     'figure.dpi':        300,
# })

# Data
N = [4, 6, 8]
c_avg = [81.24, 81.40, 80.87]
c_std = [0.31, 0.38, 2.85]
q_avg = [80.75, 80.74, 81.62]
q_std = [0.38, 0.36, 0.39]
q_non_avg = [70.99, 77.00, 78.66]
q_non_std = [1.10, 0.55, 10.27]

# # Set MATLAB-like style
plt.style.use('classic')

putils.setPlot(fontsize=20, labelsize=20, lw=2)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Define bar width and positions
bar_width = 0.2
x = np.array(N)
x1 = x - bar_width  # Position for c bars
x2 = x   # Position for q bars
x3 = x + bar_width  # Position for q bars

# Define a nice color palette (MATLAB-inspired)
# colors = ['#0072BD', '#D95319']

# Modern color palette (inspired by contemporary design)
colors = ['#4C78A8', '#F58518', '#6d933c']  # Teal-blue and vibrant orange

# Plot bars with error bars
ax.bar(x1, c_avg, bar_width, yerr=c_std, color=colors[0], 
       capsize=5, label='Classical (contrastive)', alpha=0.9)
ax.bar(x2, q_avg, bar_width, yerr=q_std, color=colors[1], 
       capsize=5, label='IQFMs (contrastive)', alpha=0.9)
ax.bar(x3, q_non_avg, bar_width, yerr=q_non_std, color=colors[2], 
       capsize=5, label='IQFMs (non-contrastive)', alpha=0.9)
ax.set_xticklabels([16, 64, 256])

# Customize the plot
ax.set_xlabel('Layer width ($M$)')
ax.set_ylabel('Accuracy (%)')
#ax.set_xscale('log', base=2)

#ax.set_title('Average Values with Standard Deviation vs N', fontsize=14, pad=10)
ax.grid(True, linestyle='--', alpha=0.7, axis='y')  # Grid only on y-axis for clarity

# Set x-ticks to match data points
ax.set_xticks(N)
ax.set_ylim([60, 90])

# Add legend
ax.legend(fontsize=14, loc='best')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Display the plot
plt.show()
for ftype in ['png', 'pdf', 'svg']:
    plt.savefig(f'fig9.{ftype}', format=ftype, bbox_inches = 'tight', dpi=300)