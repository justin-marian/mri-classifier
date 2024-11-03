""" models/plots.py """

import os
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """ Generating different types of plots for data analysis. """
    
    def __init__(self, title="Plot", xlabel="X-axis", ylabel="Y-axis", save_dir="../images"):
        """ Initialize plot settings with default title, axis labels, and save directory. """
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        sns.set_theme(style="whitegrid")
        mpl.rcParams.update({
            "font.family": "serif",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "grid.color": "#777777"
        })

    def plot_combined_bar_charts(self, train_counts, valid_counts, test_counts, save_name=None):
        """ Plot combined horizontal bar charts for training, validation, and testing distributions with mean and median lines. """
        data_sets = [("Training", train_counts), ("Validation", valid_counts), ("Testing", test_counts)]
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))

        for idx, (title, counts) in enumerate(data_sets):
            labels = list(counts.keys())
            values = np.array(list(counts.values()))
            total = values.sum()
            mean_value = np.mean(values)
            median_value = np.median(values)

            ax_bar = axes[idx]
            colors = sns.color_palette("Spectral", len(values))
            ax_bar.barh(labels, values, color=colors)
            
            ax_bar.axvline(mean_value, color='red', linestyle='--', label='Mean')
            ax_bar.axvline(median_value, color='blue', linestyle='--', label='Median')
            ax_bar.set_title(f"{title} - Class Distribution", fontsize=12, fontweight="bold")

            for i, v in enumerate(values):
                percentage = f"{(v / total * 100):.1f}%"
                ax_bar.text(v + 0.1, i, f"{v} ({percentage})", va="center", fontsize=9, color="black")

            mean_line_legend = plt.Line2D([0], [0], color="red", lw=1.5, linestyle='--', label="Mean (Red)")
            median_line_legend = plt.Line2D([0], [0], color="blue", lw=1.5, linestyle='--', label="Median (Blue)")
            ax_bar.legend(handles=[mean_line_legend, median_line_legend], loc="upper right")

        fig.text(0.5, 0.04, self.xlabel, ha='center', fontsize=12)
        fig.text(0.04, 0.5, self.ylabel, va='center', rotation='vertical', fontsize=12)

        plt.tight_layout(pad=2.0, rect=[0.05, 0.05, 1, 0.95])

        if save_name:
            plt.savefig(os.path.join(self.save_dir, f"{save_name}_bar_charts.svg"), format="svg", bbox_inches="tight")
            print(f"Bar chart plot saved as {save_name}_bar_charts.svg in {self.save_dir}")
        else:
            plt.show()

    def plot_combined_histograms(self, train_counts, valid_counts, test_counts, save_name=None):
        """ Plot combined normalized histograms for training, validation, and testing distributions with peak annotations. """
        data_sets = [("Training", train_counts), ("Validation", valid_counts), ("Testing", test_counts)]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, (title, counts) in enumerate(data_sets):
            values = np.array(list(counts.values()))
            norm_values = values / values.sum() * 100.0

            ax_hist = axes[idx]
            counts_per_bin, bins, _ = ax_hist.hist(norm_values, bins=len(set(norm_values)), color=sns.color_palette("muted")[1], edgecolor="black")
            ax_hist.set_title(f"{title} - Frequency of Counts", fontsize=10, fontweight="bold")

            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            for count, bin_edge in zip(counts_per_bin, bins):
                ax_hist.text(bin_edge + (bins[1] - bins[0]) / 2, count + 0.1, f"{int(count)}", ha='center', fontsize=8)
            
            ax_hist.plot(bin_centers, counts_per_bin, linestyle='-', marker='o', color='red')

        fig.text(0.5, 0.04, "Percentage of Images per Class", ha='center', fontsize=12)
        fig.text(0.04, 0.5, "Frequency", va='center', rotation='vertical', fontsize=12)

        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

        if save_name:
            plt.savefig(os.path.join(self.save_dir, f"{save_name}_histograms.svg"), format="svg", bbox_inches="tight")
            print(f"Histogram plot saved as {save_name}_histograms.svg in {self.save_dir}")
        else:
            plt.show()

    def set_labels(self, title, xlabel, ylabel):
        """ Update the title and axis labels for the plots. """
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
