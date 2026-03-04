import matplotlib.pyplot as plt

import pickle

import numpy as np

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
avg_A_contrastive = [59.96, 79.62, 89.28, 91.57]
std_A_contrastive = [5.07, 2.17, 1.97, 0.87]

# Task A exactQCNN data
avg_A_exactQCNN  = [71.81, 83.93, 85, 84.85]
std_A_exactQCNN  = [1.42, 0.85, 0.3, 0.45]


# ---- Colors ----
orange = '#ea926e'
blue   = '#175676'  
green  = '#6d933c'

# ---- Common y‐limits across all panels ----
ymin_A, ymax_A = 50, 100
ymin_B, ymax_B = 0, 1

# ---- Create 1x2 subplots ----
fig, (axA, axB) = plt.subplots(1, 2, figsize=(8, 4))

axA.errorbar(n_shots, avg_A_contrastive, yerr=std_A_contrastive,
                marker='o', markersize=mksz, linewidth=lw, linestyle='-',
                color=orange, capsize=capsz, label='IQFMs')
axA.errorbar(n_shots, avg_A_exactQCNN, yerr=std_A_exactQCNN,
                marker='o', markersize=mksz, linewidth=lw, linestyle='--',
                color=green, capsize=capsz, label='exact QCNN')

axA.set_xticks(n_shots)
axA.set_xticklabels(n_shots, fontsize=10)
axA.set_xlabel('Number of shots')
axA.set_ylabel('Accuracy (%)')
axA.set_title('(a) Accuracy (Task A)')
axA.set_yticks(range(ymin_A, ymax_A + 1, 10))  # Set y-ticks
axA.set_yticklabels(range(ymin_A, ymax_A + 1, 10), fontsize=10)  # Set font size for y-tick labels
axA.set_ylim(ymin_A, ymax_A)
axA.legend(frameon=True, loc='lower right', fontsize=10)
axA.grid(True, linestyle='--', alpha=0.3)

load_path = "exact_qcnn_results_per_shot.pkl"

shot_list = [10, 1000]
shot_color = [green, blue]
shot_line = ['-', '-']

with open(load_path, "rb") as f:
    results_per_shot = pickle.load(f)

    for i, n_shots in enumerate(shot_list):
        res = results_per_shot[n_shots]
        h2_list = res["h2_list"]
        mean_vals = res["mean_outputs"]
        std_vals = res["std_outputs"]

        sort_idx = np.argsort(h2_list)
        h2_sorted = h2_list[sort_idx]
        mean_sorted = mean_vals[sort_idx]
        std_sorted = std_vals[sort_idx]

        eb = axB.errorbar(
            h2_sorted,
            mean_sorted,
            yerr=std_sorted,
            label=f"n_shots={n_shots}",
            color=shot_color[i],
            fmt="o",
            capsize=capsz,
            markersize=3,
            linewidth=lw,
            linestyle=shot_line[i]
        )

        for cap in eb[1]:
            cap.set_alpha(0.3)
        for bar in eb[2]:
            bar.set_alpha(0.3)

xticks = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
axB.set_xticks(xticks)
axB.set_xticklabels([str(x) for x in xticks], fontsize=10)
axB.set_xlabel(r'$h_2$')
axB.set_ylabel(r'$\langle X \rangle$')
axB.set_title('(b) Exact QCNN output (Task A)')
yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
axB.set_yticks(yticks)  # Set y-ticks
axB.set_yticklabels([str(y) for y in yticks], fontsize=10)  # Set font size for y-tick labels
axB.set_ylim(ymin_B, ymax_B)
axB.legend(frameon=True, loc='upper right', fontsize=10)
axB.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
# Save figure
fig_file = 'fig10'
for ftype in ['png', 'svg', 'pdf']:
    plt.savefig('{}.{}'.format(fig_file, ftype), bbox_inches = 'tight', dpi=300)
plt.show()
plt.clf()
plt.close()