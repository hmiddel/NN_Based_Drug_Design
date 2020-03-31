import random

import matplotlib.pyplot as plt
import numpy as np


def plots(x_values, y_values, label=None, save=False):
    mini = min(min(x_values), min(y_values))
    maxi = max(max(x_values), max(y_values))
    margin = (maxi - mini) / 10
    fig = plt.figure(figsize=(5, 12))

    ax1 = fig.add_subplot(311)
    ax1.set_title('real')
    ax1.hist(x_values)
    ax1.set_xlim(mini, maxi)

    ax2 = fig.add_subplot(312)
    ax2.set_title('predicted')
    ax2.hist(y_values)
    ax2.set_xlim(mini, maxi)

    ax3 = fig.add_subplot(313)
    ax3.scatter(x_values, y_values)
    ax3.set_ylim(mini - margin, maxi + margin)
    ax3.set_xlim(mini - margin, maxi + margin)
    plt.show()
    if save:
        plt.savefig('data/' + str(label) + '.png')


if __name__ == "__main__":
    x = random.sample(range(10000), 100)
    y = np.array(random.sample(range(200), 100)) - 1000
    plots(x, y)
