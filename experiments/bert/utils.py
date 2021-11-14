import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D # Imported for legends
import pdb 
import numpy as np

def visualize_confusion(intervals, label_ids, nor_val_s, save_dir=None):

    true_interval = np.zeros_like(intervals)
    true_interval[:,0] = intervals[:,0] - intervals[:,1] 
    true_interval[:,1] = intervals[:,0] + intervals[:,1] 

    nor_val_s = nor_val_s.detach().cpu().numpy()
    label_ids = label_ids.detach().cpu().numpy()
    
    num_intervals = len(intervals)
    viridis = plt.cm.get_cmap('viridis', num_intervals)
    colors = np.array([viridis(idx / num_intervals) for idx in range(len(intervals))])

    # Prepare the input data in correct format for LineCollection 
    lines = [[(i[0], j), (i[1], j)] for i, j in zip(intervals, range(len(intervals)))]

    lc = LineCollection(lines, colors= colors, linewidths=2)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.margins(0.1)
    plt.yticks([], [])

    # Adding the legends
    def make_proxy(col, scalar_mappable, **kwargs):
        color = col 
        return Line2D([0, 1], [0, 1], color=color, **kwargs)
    proxies = [make_proxy(c, lc, linewidth=2) for c in colors]
    ax.legend(proxies, range(5))

    # Adding annotations
    for i, x in enumerate(intervals):
        plt.text(x[0], i+0.1, x[0], color=colors[i])
        plt.text(x[1], i+0.1, x[1], color=colors[i])
        query_c = 'r' if label_ids[i] else 'g'
        plt.scatter(nor_val_s[i], i, s=20, c=query_c)

    if save_dir:
        plt.savefig(f'{save_dir}.png')
    plt.clf()