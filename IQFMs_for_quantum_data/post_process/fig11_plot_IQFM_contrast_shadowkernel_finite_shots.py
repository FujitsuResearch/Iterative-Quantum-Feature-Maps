# Purpose: Plot IQFM contrastive vs non-contrastive results
# and QCNN results for Task A and Task B (Figure 5 in the paper)

import matplotlib.pyplot as plt

# ---- Journal‐style settings ----
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         12,
    'mathtext.fontset':  'cm',
    'axes.linewidth':    1.2,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'xtick.top':         False,
    'ytick.right':       False,
    'xtick.major.size':  6,
    'xtick.major.width': 1.2,
    'ytick.major.size':  6,
    'ytick.major.width': 1.2,
    'figure.dpi':        300,
})
mksz = 6
lw = 1.5
capsz = 0


# ---- Data ----
n_shots = [10, 100, 500, 1000]

# Task A IQFM
avg_A_contrastive = [56.06, 84.08, 91.8, 93.28]
std_A_contrastive = [3.09, 2.74, 1.17, 1.37]

# Task A shadow kernel data
avg_A_shadow  = [51.81, 89.46, 94.13, 94.75]
std_A_shadow  = [1.95, 1.29, 0.58, 0.45]


# ---- Colors ----
orange = '#ea926e' # Contrastive
blue   = '#175676'  # iqfm Infinite Shot
green  = '#6d933c'  # shadow

# ---- Common y‐limits across all panels ----
ymin_A, ymax_A = 45, 100

# ---- Create 1x2 subplots ----
fig, axA = plt.subplots(1, 1, figsize=(4, 4))


# ---- (a) Task A IQFM and Shadow Kernel ----
axA.errorbar(n_shots, avg_A_contrastive, yerr=std_A_contrastive,
                marker='o', markersize=mksz, linewidth=lw, linestyle='-',
                color=orange, capsize=capsz, label='IQFMs')
axA.errorbar(n_shots, avg_A_shadow, yerr=std_A_shadow,
                marker='o', markersize=mksz, linewidth=lw, linestyle='--',
                color=green, capsize=capsz, label='Shadow Kernel')
# ---- Add a horizontal line for 96.82 with std dev ----
# Center line
axA.axhline(y=96.82, color=blue, linestyle='-.', linewidth=lw, label='IQFMs (Infinite Shot)')

axA.set_xticks(n_shots)
axA.set_xticklabels(n_shots, fontsize=10)
axA.set_xlabel('Number of shots')
axA.set_ylabel('Accuracy (%)')
axA.set_title('Task A')
axA.set_yticks(range(ymin_A, ymax_A + 1, 10))  # Set y-ticks
axA.set_yticklabels(range(ymin_A, ymax_A + 1, 10), fontsize=10)  # Set font size for y-tick labels
axA.set_ylim(ymin_A, ymax_A)
axA.legend(frameon=False, loc='lower right')
axA.grid(True, linestyle='--', alpha=0.3)


plt.tight_layout()
# Save figure
fig_file = 'fig11'
for ftype in ['png', 'svg', 'pdf']:
    plt.savefig('{}.{}'.format(fig_file, ftype), bbox_inches = 'tight', dpi=300)
plt.show()
plt.clf()
plt.close()