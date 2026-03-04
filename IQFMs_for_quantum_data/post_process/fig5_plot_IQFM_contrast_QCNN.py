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
depths = [1, 2, 3, 4, 5, 10]
# Task A IQFM
avg_A_contrastive     = [94.64,	95.97, 96.55, 96.78, 96.82, 96.71]
std_A_contrastive     = [1.55, 0.97, 0.74, 0.73, 0.66, 0.63]
avg_A_non_contrastive = [94.95, 95.17, 95.26, 95.25, 95.25, 95.34]
std_A_non_contrastive = [0.85, 0.65, 0.51, 0.43, 0.42, 0.32]

# Task B IQFM
avg_B_contrastive     = [93.67, 94.16, 94.69, 94.84, 94.75, 94.85]
std_B_contrastive     = [1.52, 1.29, 1.29, 1.01, 1.11, 0.8]
avg_B_non_contrastive = [94.28, 93.75, 93.71, 93.62, 93.54, 93.51]
std_B_non_contrastive = [1.2, 1.03, 1, 0.74, 0.71, 0.59]

# QCNN depths & data
var_depth = [1, 2, 4, 8, 16, 32]
avg_A_qcnn  = [87.97, 93.98, 95.24, 94.8, 94.45, 94.38]
std_A_qcnn  = [2.89, 1.53, 0.42, 0.29, 0.14, 0.14]
avg_B_qcnn  = [74.91, 81.46, 80.69, 84.26, 85.76, 86.27]
std_B_qcnn  = [ 6.36,  6.74,  7.44,  6.03,  4.46,  4.09]

# ---- Colors ----
orange = '#ea926e' # Contrastive
blue   = '#175676'  # Non-Contrastive
green  = '#6d933c'  # QCNN

# ---- Common y‐limits across all panels ----
ymin_A, ymax_A = 84, 98
ymin_B, ymax_B = 68, 98

# ---- Create 2x2 subplots ----
fig, ((axA_iq, axA_qc), (axB_iq, axB_qc)) = plt.subplots(2, 2, figsize=(8, 8))

# ---- (a) Task A IQFM ----
axA_iq.errorbar(depths, avg_A_contrastive, yerr=std_A_contrastive,
                marker='o', markersize=mksz, linewidth=lw, linestyle='-',
                color=orange, capsize=capsz, label='Contrastive')
axA_iq.errorbar(depths, avg_A_non_contrastive, yerr=std_A_non_contrastive,
                marker='o', markersize=mksz, linewidth=lw, linestyle='--',
                color=blue, capsize=capsz, label='Non‑Contrastive')
axA_iq.set_xticks(depths)
axA_iq.set_ylabel('Accuracy (%)')
axA_iq.set_title('(a) IQFMs (Task A)')
axA_iq.set_ylim(ymin_A, ymax_A)
axA_iq.legend(frameon=False, loc='lower right')
axA_iq.grid(True, linestyle='--', alpha=0.3)

# ---- (b) Task A QCNN ----
axA_qc.errorbar(var_depth, avg_A_qcnn, yerr=std_A_qcnn,
                marker='o', markersize=mksz, linewidth=lw, linestyle='-',
                color=green, capsize=capsz, label='QCNN')
axA_qc.set_xscale('log', base=2)
axA_qc.set_xticks(var_depth)
axA_qc.set_xticklabels(var_depth)
axA_qc.set_title('(b) QCNN (Task A)')
axA_qc.set_ylim(ymin_A, ymax_A)
axA_qc.legend(frameon=False, loc='lower right')
axA_qc.grid(True, linestyle='--', alpha=0.3)

# ---- (c) Task B IQFM ----
axB_iq.errorbar(depths, avg_B_contrastive, yerr=std_B_contrastive,
                marker='o', markersize=mksz, linewidth=lw, linestyle='-',
                color=orange, capsize=capsz, label='Contrastive')
axB_iq.errorbar(depths, avg_B_non_contrastive, yerr=std_B_non_contrastive,
                marker='o', markersize=mksz, linewidth=lw, linestyle='--',
                color=blue, capsize=capsz, label='Non‑Contrastive')
axB_iq.set_xticks(depths)
axB_iq.set_xlabel('Layer Depth')
axB_iq.set_ylabel('Accuracy (%)')
axB_iq.set_title('(c) IQFMs (Task B)')
axB_iq.set_ylim(ymin_B, ymax_B)
axB_iq.legend(frameon=False, loc='lower right')
axB_iq.grid(True, linestyle='--', alpha=0.3)

# ---- (d) Task B QCNN ----
axB_qc.errorbar(var_depth, avg_B_qcnn, yerr=std_B_qcnn,
                marker='o', markersize=mksz, linewidth=lw, linestyle='-',
                color=green, capsize=capsz, label='QCNN')
axB_qc.set_xscale('log', base=2)
axB_qc.set_xticks(var_depth)
axB_qc.set_xticklabels(var_depth)
axB_qc.set_xlabel('Var Depth')
axB_qc.set_title('(d) QCNN (Task B)')
axB_qc.set_ylim(ymin_B, ymax_B)
axB_qc.legend(frameon=False, loc='lower right')
axB_qc.grid(True, linestyle='--', alpha=0.3)


plt.tight_layout()
# Save figure
fig_file = 'fig5'
for ftype in ['png', 'svg', 'pdf']:
    plt.savefig('{}.{}'.format(fig_file, ftype), bbox_inches = 'tight', dpi=300)
plt.show()
plt.clf()
plt.close()