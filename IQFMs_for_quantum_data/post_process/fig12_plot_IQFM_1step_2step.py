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
layer_depth = [1, 2, 3, 4, 5]

# Task A IQFM_2step
avg_A_2step = [94.64, 95.97, 96.55, 96.78, 96.82]
std_A_2step = [1.55, 0.97, 0.74, 0.73, 0.66]

# Task B IQFM_2step
avg_B_2step = [93.67, 94.16, 94.69, 94.84, 94.75]
std_B_2step = [1.52, 1.29, 1.29, 1.01, 1.11]

# Task A IQFM_1step
avg_A_1step = [93.04, 94.97, 95.13, 95.49, 95.54]
std_A_1step = [2.87, 1.35, 1.22, 0.78, 0.84]

# Task B IQFM_1step
avg_B_1step = [90.80, 92.63, 93.37, 94.07, 94.03]
std_B_1step = [2.84, 2.02, 1.40, 1.49, 1.40]

# ---- Colors ----
orange = '#ea926e' # IQFM_2step
blue   = '#175676'  # IQFM_1step
green  = '#6d933c'  # QCNN

# ---- Common y‐limits across all panels ----
ymin_A, ymax_A = 84, 98
ymin_B, ymax_B = 84, 98

# ---- Create 1x2 subplots ----
fig, (axA, axB) = plt.subplots(1, 2, figsize=(8, 4))

# ---- (a) 
axA.errorbar(layer_depth, avg_A_2step, yerr=std_A_2step,
                marker='o', markersize=mksz, linewidth=lw, linestyle='-',
                color=orange, capsize=capsz, label='IQFMs_2step')
axA.errorbar(layer_depth, avg_A_1step, yerr=std_A_1step,
                marker='o', markersize=mksz, linewidth=lw, linestyle='--',
                color=blue, capsize=capsz, label='IQFMs_1step')
axA.set_xticks(layer_depth)
axA.set_xticklabels(layer_depth)
axA.set_xlabel('Layer Depth')
axA.set_ylabel('Accuracy (%)')
axA.set_title('(a) Task A')
axA.set_ylim(ymin_A, ymax_A)
axA.legend(frameon=False, loc='lower left')
axA.grid(True, linestyle='--', alpha=0.3)

# ---- (b)
axB.errorbar(layer_depth, avg_B_2step, yerr=std_B_2step,
                marker='o', markersize=mksz, linewidth=lw, linestyle='-',
                color=orange, capsize=capsz, label='IQFMs_2step')
axB.errorbar(layer_depth, avg_B_1step, yerr=std_B_1step,
                marker='o', markersize=mksz, linewidth=lw, linestyle='--',
                color=blue, capsize=capsz, label='IQFMs_1step')
axB.set_xticks(layer_depth)
axB.set_xticklabels(layer_depth)
axB.set_xlabel('Layer Depth')
axB.set_title('(b) Task B')
axB.set_ylim(ymin_B, ymax_B)
axB.legend(frameon=False, loc='lower left')
axB.grid(True, linestyle='--', alpha=0.3)


plt.tight_layout()
# Save figure
fig_file = 'fig12'
for ftype in ['png', 'svg', 'pdf']:
    plt.savefig('{}.{}'.format(fig_file, ftype), bbox_inches = 'tight', dpi=300)
plt.show()
plt.clf()
plt.close()