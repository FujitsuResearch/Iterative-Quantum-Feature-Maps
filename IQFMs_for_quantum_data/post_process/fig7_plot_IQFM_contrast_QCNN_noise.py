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
noise_level = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

# Task A IQFM
avg_A_contrastive     = [96.82, 92.45, 86.75, 80.19, 74.70, 70.45, 64.88]
std_A_contrastive     = [0.66, 4.38, 8.62, 11.99, 13.40, 14.96, 13.82]

# Task B IQFM
avg_B_contrastive     = [94.75, 93.78, 89.32, 82.15, 74.90, 73.55, 70.89]
std_B_contrastive     = [1.11, 2.06, 6.94, 10.47, 13.86, 13.36, 13.06]

# QCNN data
avg_A_qcnn  = [95.24, 95.24, 88.28, 68.18, 55.97, 51.40, 50.42]
std_A_qcnn  = [0.42, 0.75, 4.78, 14.33, 10.33, 3.80, 3.01]
avg_B_qcnn  = [86.27, 77.33, 61.18, 53.81, 51.17, 46.51, 40.95]
std_B_qcnn  = [4.09, 11.36, 13.74, 12.24, 14.56, 14.26, 13.77]

# ---- Colors ----
orange = '#ea926e' # Contrastive
blue   = '#175676'  # Non-Contrastive
green  = '#6d933c'  # QCNN

# ---- Common y‐limits across all panels ----
ymin_A, ymax_A = 40, 100
ymin_B, ymax_B = 20, 100


# ---- Create 2x2 subplots ----
fig, ((axA, axB), (axA_ret, axB_ret)) = plt.subplots(2, 2, figsize=(8, 8))

# ---- (a) Task A IQFM and QCNN ----
axA.errorbar(noise_level, avg_A_contrastive, yerr=std_A_contrastive,
                marker='o', markersize=mksz, linewidth=lw, linestyle='-',
                color=orange, capsize=capsz, label='IQFMs')
axA.errorbar(noise_level, avg_A_qcnn, yerr=std_A_qcnn,
                marker='o', markersize=mksz, linewidth=lw, linestyle='--',
                color=green, capsize=capsz, label='QCNN')
axA.set_xticks(noise_level)
axA.set_xticklabels(noise_level)
# axA.set_xlabel('Noise Level')
axA.set_ylabel('Accuracy (%)')
axA.set_title('(a) Accuracy vs noise (Task A)')
axA.set_ylim(ymin_A, ymax_A)
axA.legend(frameon=False, loc='lower left')
axA.grid(True, linestyle='--', alpha=0.3)

# ---- (b) Task B IQFM and QCNN ----
axB.errorbar(noise_level, avg_B_contrastive, yerr=std_B_contrastive,
                marker='o', markersize=mksz, linewidth=lw, linestyle='-',
                color=orange, capsize=capsz, label='IQFMs')
axB.errorbar(noise_level, avg_B_qcnn, yerr=std_B_qcnn,
                marker='o', markersize=mksz, linewidth=lw, linestyle='--',
                color=green, capsize=capsz, label='QCNN')
axB.set_xticks(noise_level)
axB.set_xticklabels(noise_level)
# axB.set_xlabel('Noise Level')
axB.set_title('(b) Accuracy vs noise (Task B)')
axB.set_ylim(ymin_B, ymax_B)
axB.legend(frameon=False, loc='lower left')
axB.grid(True, linestyle='--', alpha=0.3)


# Task A IQFM
avg_A_contrastive_ret     = [96.82/96.82, 92.45/96.82, 86.75/96.82, 80.19/96.82, 74.70/96.82, 70.45/96.82, 64.88/96.82]

# Task B IQFM
avg_B_contrastive_ret     = [94.75/94.75, 93.78/94.75, 89.32/94.75, 82.15/94.75, 74.90/94.75, 73.55/94.75, 70.89/94.75]

# QCNN data
avg_A_qcnn_ret  = [95.24/95.24, 95.24/95.24, 88.28/95.24, 68.18/95.24, 55.97/95.24, 51.40/95.24, 50.42/95.24]
avg_B_qcnn_ret  = [86.27/86.27, 77.33/86.27, 61.18/86.27, 53.81/86.27, 51.17/86.27, 46.51/86.27, 40.95/86.27]

# ---- Colors ----
orange = '#ea926e' # Contrastive
blue   = '#175676'  # Non-Contrastive
green  = '#6d933c'  # QCNN

# ---- Common y‐limits across all panels ----
ymin_A_ret, ymax_A_ret = 0, 1.1
ymin_B_ret, ymax_B_ret = 0, 1.1

# ---- (c) Task A IQFM and QCNN_Retention----
axA_ret.errorbar(noise_level, avg_A_contrastive_ret,
                marker='o', markersize=mksz, linewidth=lw, linestyle='-',
                color=orange, capsize=capsz, label='IQFMs')
axA_ret.errorbar(noise_level, avg_A_qcnn_ret,
                marker='o', markersize=mksz, linewidth=lw, linestyle='--',
                color=green, capsize=capsz, label='QCNN')
axA_ret.set_xticks(noise_level)
axA_ret.set_xticklabels(noise_level)
axA_ret.set_xlabel('Noise Level')
axA_ret.set_ylabel('Retention')
axA_ret.set_title('(c) Retention vs noise (Task A)')
axA_ret.set_ylim(ymin_A_ret, ymax_A_ret)
axA_ret.legend(frameon=False, loc='lower left')
axA_ret.grid(True, linestyle='--', alpha=0.3)

# ---- (d) Task B IQFM and QCNN ----
axB_ret.errorbar(noise_level, avg_B_contrastive_ret,
                marker='o', markersize=mksz, linewidth=lw, linestyle='-',
                color=orange, capsize=capsz, label='IQFMs')
axB_ret.errorbar(noise_level, avg_B_qcnn_ret,
                marker='o', markersize=mksz, linewidth=lw, linestyle='--',
                color=green, capsize=capsz, label='QCNN')
axB_ret.set_xticks(noise_level)
axB_ret.set_xticklabels(noise_level)
axB_ret.set_xlabel('Noise Level')
axB_ret.set_title('(d) Retention vs noise (Task B)')
axB_ret.set_ylim(ymin_B_ret, ymax_B_ret)
axB_ret.legend(frameon=False, loc='lower left')
axB_ret.grid(True, linestyle='--', alpha=0.3)


plt.tight_layout()
# Save figure
fig_file = 'fig7'
for ftype in ['png', 'svg', 'pdf']:
    plt.savefig('{}.{}'.format(fig_file, ftype), bbox_inches = 'tight', dpi=300)
plt.show()
plt.clf()
plt.close()