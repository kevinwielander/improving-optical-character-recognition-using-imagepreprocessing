import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from utils.config import PLOTS_PATH
import logging
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

class Visualization:
    def __init__(self, df):
        logger.info('Initialized Logging')
        self.df = df

    def plot_metrics(self, save=False):
        logger.info('Plotting Metrics')
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
            logger.info('Saving Metrics')
            if not os.path.exists(PLOTS_PATH):
                os.makedirs(PLOTS_PATH)
            plt.savefig(PLOTS_PATH+"/metrics_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png")
        #plt.show()

    def plot_histogram(self, save=False):
        logger.info('Plotting Histogram')
        # Filter out the rows with statistics
        df = self.df[self.df['Improvement in Percent'].apply(lambda x: isinstance(x, (int, float)))]

        # Create a histogram of the improvements
        plt.hist(df['Improvement in Percent'], bins=20, edgecolor='black')

        # Set the title and labels
        plt.title('Histogram of Improvements')
        plt.xlabel('Improvement in Percent')
        plt.ylabel('Frequency')

        if save:
            logger.info('Saving Histogram Metrics')
            if not os.path.exists(PLOTS_PATH):
                os.makedirs(PLOTS_PATH)
            plt.savefig(PLOTS_PATH+f"/histogram_improvement{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png")

        # Show the plot
        #plt.show()

