import os

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
class Visualization:
    def __init__(self, df):
        self.df = df

    def plot_metrics(self, save=False):
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # Plot CER
        sns.barplot(data=self.df, x='Filename', y='CER', ax=axs[0])
        axs[0].set_title('Character Error Rate')
        axs[0].tick_params(axis='x', rotation=90)

        # Plot WER
        sns.barplot(data=self.df, x='Filename', y='WER', ax=axs[1])
        axs[1].set_title('Word Error Rate')
        axs[1].tick_params(axis='x', rotation=90)

        # Plot Levenshtein Distance
        sns.barplot(data=self.df, x='Filename', y='Levenshtein Distance', ax=axs[2])
        axs[2].set_title('Levenshtein Distance')
        axs[2].tick_params(axis='x', rotation=90)

        plt.tight_layout()

        if save:
            if not os.path.exists("resources/plots"):
                os.makedirs("resources/plots")
            plt.savefig(f"resources/plots/metrics_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png")
        plt.show()

