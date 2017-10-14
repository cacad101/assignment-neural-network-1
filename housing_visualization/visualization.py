"""
Visualization data:
- Title
- x and y Label
- Legend
"""

import matplotlib.pyplot as plt

def plot_graph(title, x_label, y_label, x_vals, y_vals, data_labels, save_fig="", show_graph=True):
    """ Plot graphs using matplotlib """
    plt.figure()

    for i in range(len(data_labels)):
        plt.plot(x_vals[i], y_vals[i], label=data_labels[i])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    if save_fig != "":
        plt.savefig(save_fig)

    if show_graph:
        plt.show()
