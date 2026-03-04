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
noise_level = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Task A IQFM_2step
avg_A_2step = [96.82, 92.45, 86.75, 80.19, 74.70, 70.45, 64.88]
std_A_2step = [0.66, 4.38, 8.62, 11.99, 13.40, 14.96, 13.82]

# Task B IQFM_2step
avg_B_2step = [94.75, 93.78, 89.32, 82.15, 74.90, 73.55, 70.89]
std_B_2step = [1.11, 2.06, 6.94, 10.47, 13.86, 13.36, 13.06]

# Task A IQFM_1step
avg_A_1step = [95.54, 93.40, 92.68, 89.28, 85.45, 82.78, 81.89]
std_A_1step = [0.84, 1.74, 2.96, 4.83, 7.40, 10.42, 10.42]

# Task B IQFM_1step
avg_B_1step = [94.03, 90.38, 81.64, 71.97, 66.21, 63.14, 60.79]
std_B_1step = [1.40, 4.26, 10.18, 12.53, 10.72, 13.34, 11.66]

# Task A QCNN
avg_A_qcnn = [95.24, 95.24, 88.28, 68.18, 55.97, 51.40, 50.42]
std_A_qcnn = [0.42, 0.75, 4.78, 14.33, 10.33, 3.80, 3.01]

# Task B QCNN
avg_B_qcnn = [86.27, 77.33, 61.18, 53.81, 51.17, 46.51, 40.95]
std_B_qcnn = [4.09, 11.36, 13.74, 12.24, 14.56, 14.26, 13.77]

# ---- Colors ----
orange = '#ea926e' # IQFM_2step
blue   = '#175676'  # IQFM_1step
green  = '#6d933c'  # QCNN

# ---- Common y‐limits across all panels ----
ymin_A, ymax_A = 20, 100
ymin_B, ymax_B = 20, 100

# ---- Create 1x2 subplots ----
fig, (axA, axB) = plt.subplots(1, 2, figsize=(8, 4))


# ---- (a)
axA.errorbar(noise_level, avg_A_2step, yerr=std_A_2step,
                marker='o', markersize=mksz, linewidth=lw, linestyle='-',
                color=orange, capsize=capsz, label='IQFMs_2step')
axA.errorbar(noise_level, avg_A_1step, yerr=std_A_1step,
                marker='o', markersize=mksz, linewidth=lw, linestyle='--',
                color=blue, capsize=capsz, label='IQFMs_1step')
axA.errorbar(noise_level, avg_A_qcnn, yerr=std_A_qcnn,
                marker='o', markersize=mksz, linewidth=lw, linestyle='-.',
                color=green, capsize=capsz, label='QCNN')
axA.set_xticks(noise_level)
axA.set_xticklabels(noise_level)
axA.set_xlabel('Noise Level')
axA.set_ylabel('Accuracy (%)')
axA.set_title('(a) Task A')
axA.set_ylim(ymin_A, ymax_A)
axA.legend(frameon=False, loc='lower left', handlelength=3)
axA.grid(True, linestyle='--', alpha=0.3)

# ---- (b)
axB.errorbar(noise_level, avg_B_2step, yerr=std_B_2step,
                marker='o', markersize=mksz, linewidth=lw, linestyle='-',
                color=orange, capsize=capsz, label='IQFMs_2step')
axB.errorbar(noise_level, avg_B_1step, yerr=std_B_1step,
                marker='o', markersize=mksz, linewidth=lw, linestyle='--',
                color=blue, capsize=capsz, label='IQFMs_1step')
axB.errorbar(noise_level, avg_B_qcnn, yerr=std_B_qcnn,
                marker='o', markersize=mksz, linewidth=lw, linestyle='-.',
                color=green, capsize=capsz, label='QCNN')
axB.set_xticks(noise_level)
axB.set_xticklabels(noise_level)
axB.set_xlabel('Noise Level')
axB.set_title('(b) Task B')
axB.set_ylim(ymin_B, ymax_B)
axB.legend(frameon=False, loc='lower left', handlelength=3)
axB.grid(True, linestyle='--', alpha=0.3)


plt.tight_layout()
# Save figure
fig_file = 'fig13'
for ftype in ['png', 'svg', 'pdf']:
    plt.savefig('{}.{}'.format(fig_file, ftype), bbox_inches = 'tight', dpi=300)
plt.show()
plt.clf()
plt.close()