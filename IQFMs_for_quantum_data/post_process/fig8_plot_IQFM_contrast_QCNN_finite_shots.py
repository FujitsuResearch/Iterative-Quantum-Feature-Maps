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

# Task A QCNN data
avg_A_qcnn  = [64.52, 77.81, 78.14, 77.86]
std_A_qcnn  = [2.08, 1.57, 1.56, 1.59]

# Task B IQFM
avg_B_contrastive = [59.67, 90.05, 93.45, 94.3]
std_B_contrastive = [5.6, 1.8, 1.1, 1.17]

# Task B QCNN data
avg_B_qcnn  = [55.85, 68.33, 69.99, 69.96]
std_B_qcnn  = [5.13, 5.21, 6.57, 6.43]


# ---- Colors ----
orange = '#ea926e' # Contrastive
blue   = '#175676'  # Non-Contrastive
green  = '#6d933c'  # QCNN

# ---- Common y‐limits across all panels ----
ymin_A, ymax_A = 50, 100
ymin_B, ymax_B = 45, 100

# ---- Create 1x2 subplots ----
fig, (axA, axB) = plt.subplots(1, 2, figsize=(8, 4))


# ---- (a) Task A IQFM and QCNN ----
axA.errorbar(n_shots, avg_A_contrastive, yerr=std_A_contrastive,
                marker='o', markersize=mksz, linewidth=lw, linestyle='-',
                color=orange, capsize=capsz, label='IQFMs')
axA.errorbar(n_shots, avg_A_qcnn, yerr=std_A_qcnn,
                marker='o', markersize=mksz, linewidth=lw, linestyle='--',
                color=green, capsize=capsz, label='QCNN')
axA.set_xticks(n_shots)
axA.set_xticklabels(n_shots, fontsize=10)
axA.set_xlabel('Number of shots')
axA.set_ylabel('Accuracy (%)')
axA.set_title('(a) Task A')
axA.set_yticks(range(ymin_A, ymax_A + 1, 10))  # Set y-ticks
axA.set_yticklabels(range(ymin_A, ymax_A + 1, 10), fontsize=10)  # Set font size for y-tick labels
axA.set_ylim(ymin_A, ymax_A)
axA.legend(frameon=False, loc='lower right')
axA.grid(True, linestyle='--', alpha=0.3)

# ---- (b) Task B IQFM and QCNN ----
axB.errorbar(n_shots, avg_B_contrastive, yerr=std_B_contrastive,
                marker='o', markersize=mksz, linewidth=lw, linestyle='-',
                color=orange, capsize=capsz, label='IQFMs')
axB.errorbar(n_shots, avg_B_qcnn, yerr=std_B_qcnn,
                marker='o', markersize=mksz, linewidth=lw, linestyle='--',
                color=green, capsize=capsz, label='QCNN')
axB.set_xticks(n_shots)
axB.set_xticklabels(n_shots, fontsize=10)
axB.set_xlabel('Number of shots')
axB.set_title('(b) Task B')
axB.set_yticks(range(ymin_B, ymax_B + 1, 10))  # Set y-ticks
axB.set_yticklabels(range(ymin_B, ymax_B + 1, 10), fontsize=10)  # Set font size for y-tick labels
axB.set_ylim(ymin_B, ymax_B)
axB.legend(frameon=False, loc='lower right')
axB.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
# Save figure
fig_file = 'fig8'
for ftype in ['png', 'svg', 'pdf']:
    plt.savefig('{}.{}'.format(fig_file, ftype), bbox_inches = 'tight', dpi=300)
plt.show()
plt.clf()
plt.close()