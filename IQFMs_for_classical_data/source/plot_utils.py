import matplotlib
import matplotlib.pyplot as plt
import numpy as np

BLUE= [x/255.0 for x in [0, 114, 178]]
VERMILLION= [x/255.0 for x in [213, 94, 0]]
GREEN= [x/255.0 for x in [0, 158, 115]]
BROWN = [x/255.0 for x in [72, 55, 55]]

typical = [
    BLUE,
    VERMILLION,
    GREEN,
    BROWN
]


BLUE_m = [x/255.0 for x in [76, 114, 176]]
VERMILLION_m = [x/255.0 for x in [221, 132, 82]]
YELLOW_m = [x/255.0 for x in [204, 185, 116]]
PURPLE_m = [x/255.0 for x in [129, 114, 179]]
RED_m = [x/255.0 for x in [196, 78, 82]]
GREEN_m = [x/255.0 for x in [112, 173, 71]]

modern = [
    BLUE_m,
    VERMILLION_m,
    RED_m,
    GREEN_m,
    PURPLE_m,
    YELLOW_m
]

typical_2 = [
    (152 / 256, 73 / 256, 66 / 256),
    (71 / 256, 150 / 256, 157 / 256),
    "black"
]

cycle = [
'#e41a1c',
'#377eb8',
'#4daf4a',
'#984ea3',
'#ff7f00',
'#ffd92f'
]

d_colors = [
'#777777',
'#2166ac',
'#fee090',
'#fdbb84',
'#fc8d59',
'#e34a33',
'#b30000',
'#00706c'
]

seq_colors = [
'#777777',
'#a1dab4',
'#41b6c4',
'#2c7fb8',
'#253494'
]
def setPlot(fontsize=24, labelsize=24, lw=2):
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    if fontsize >  0:
        plt.rcParams["font.size"] = fontsize
    if labelsize > 0: 
        plt.rcParams['xtick.labelsize'] = labelsize
        plt.rcParams['ytick.labelsize'] = labelsize
    plt.rc('axes', linewidth=lw)

def plotContour(fig, ax, data, title, fontsize, vmin, vmax, cmap):
    ax.set_title(title, fontsize=fontsize)
    t, s = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
    if vmin == None:
        vmin = np.min(data)
    if vmax == None:
        vmax = np.max(data)

    mp = ax.contourf(s, t, np.transpose(data), 15, cmap=cmap, levels=np.linspace(vmin, vmax, 60), extend="both", zorder=-20)
    #fig.colorbar(mp, ax=ax)
    ax.set_rasterization_zorder(-10)
    #ax.set_xlabel(r"Time", fontsize=fontsize)
    return mp

def set_axes_tick1(axs, xlabel=None, ylabel=None, legend=True, w=1, xmin=0, xmax=0, ymin=0, ymax=0, \
        tick_length_unit=4, tick_direction='in', tick_minor=True, alpha=0.8):
    for ax in axs:
        if legend:
            ax.legend()
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
            
        ax.grid(lw = 0.5, ls = 'dotted')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params('both', length=tick_length_unit*3, width=w, which='major', direction=tick_direction, grid_alpha=alpha)
        if tick_minor:
            ax.minorticks_on()
            ax.tick_params('both', length=tick_length_unit*2, width=w, which='minor', direction=tick_direction, grid_alpha=alpha)
        
        if xmax > xmin:
            ax.set_xlim([xmin, xmax])
        if ymax > ymin:
            ax.set_ylim([ymin, ymax])

def set_axes_facecolor(axs, fcolor=(0.95, 0.95, 0.95)):
    for ax in axs.ravel():
        ax.set_facecolor(fcolor)