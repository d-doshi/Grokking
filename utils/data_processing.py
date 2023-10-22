import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


######################### IPR etc. #########################
def calculate_ipr(array, r):
    return np.power(array / np.sqrt((array ** 2).sum()), 2*r).sum()

def calculate_gini(array):
    return np.abs(np.expand_dims(array, 0) - np.expand_dims(array, 1)).mean() / array.mean() / 2

def ipr_test(array, r, chi_ipr):
    ipr_local = calculate_ipr(array, r)
    if ipr_local >= chi_ipr:
        return True
    else:
        return False


######################### Plot vertical lines at given freq #########################
def lines(k, p, col='grey'):
    plt.axvline(x=k%97, alpha=0.5, color=col)
    plt.axvline(x=97-k%97, alpha=0.5, color=col)

def line(k, p, col='grey'):
    kk = np.minimum(k%p, p - k%p)
    plt.axvline(x=kk, alpha=0.5, color=col)


######################### Custom contourplots #########################
def plot_linearmap(cdict):
    newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
    rgba = newcmp(np.linspace(0, 1, 256))
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    col = ['r', 'g', 'b']
    for xx in [0.25, 0.5, 0.75]:
        ax.axvline(xx, color='0.7', linestyle='--')
    for i in range(3):
        ax.plot(np.arange(256)/256, rgba[:, i], color=col[i])
    ax.set_xlabel('index')
    ax.set_ylabel('RGB')
    plt.show()