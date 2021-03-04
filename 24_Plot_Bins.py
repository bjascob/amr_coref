#!/usr/bin/python3
import numpy
import matplotlib.pyplot as plt


# From CorefFeaturizer.get_one_hot_bin "curve_a"
def val_to_bin(val, max_val, num_bins):
    a = (max_val/num_bins)
    y = 1 - a/(a + val)
    bin = int(round(num_bins*y))
    bin = numpy.clip(bin, 0, num_bins-1)
    # Extra logic to fix narrow graphs at first
    if bin > val:
        bin = int(numpy.ceil(val))
    return bin


# Test CorefFeaturizer bin'ing function
if __name__ == '__main__':
    # Definition for val_to_bin
    num_bins   = 40
    max_val    = 1000
    # Ploting variables
    num_prnt_x = 10
    max_plot_x = int(max_val * 1.1)
    max_plot_y = num_bins

    # Print to screen
    last_bin = -1
    pctr = 0
    for x in range(0, max_val-1):
        bin = val_to_bin(x, max_val, num_bins)
        if bin != last_bin:
            print('Val=%-3d Bin=%-2d  : ' % (x, bin), end='')
            last_bin = bin
            if (pctr+1) % 6 == 0: print()
            pctr += 1
    print()

    # Plot graph
    xvals = list(range(0, max_plot_x))
    yvals = [val_to_bin(x, max_val, num_bins) for x in xvals]
    plt.plot(xvals, yvals)
    ax = plt.gca()
    ax.set_xlim(0, max_plot_x)
    ax.set_ylim(0, max_plot_y)
    ax.set_xticks(range(0, max_plot_x, max_plot_x//num_prnt_x))
    ax.set_yticks(range(0, max_plot_y, max_plot_y//num_bins))
    plt.xlabel('feature value')
    plt.ylabel('bin number')
    plt.grid(True)
    plt.show()
