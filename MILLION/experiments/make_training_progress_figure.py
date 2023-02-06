import pandas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import numpy as np
from os import listdir

def smooth_values(x, factor=2):
    x = np.concatenate(([x[0]], x, [x[-1]]))
    x = x[:-2] + x[1:-1] + x[2:]
    x = x/3
    if factor > 1:
        return smooth_values(x, factor - 1)
    return x


def plot_with_std(x, y, label, color):
    std = np.std(y, axis=0)
    mean = np.mean(y, axis=0)
    plt.plot(x, mean, label=label, color=color)
    plt.fill_between(x, mean - 0.5 * std, mean + 0.5 * std, color=color, alpha=0.2)

if __name__ == '__main__':
    path = Path(__file__).parent / '..' / 'logs' / 'ml10'
    demonstrations = listdir(path / 'demonstrations')
    rewards = listdir(path / 'dense_rewards')
    language_instructions = listdir(path / 'language_instructions')
    language_instructions.remove('run_3')

    df_1 = [pandas.read_csv(path / 'demonstrations' / run / 'progress.csv')[::2] for run in demonstrations]
    df_2 = [pandas.read_csv(path / 'dense_rewards' / run / 'progress.csv')[::2] for run in rewards]
    df_3 = [pandas.read_csv(path / 'language_instructions' / run / 'progress.csv')[::2] for run in language_instructions]
    df_2[1] = df_2[1][:-1]

    y_1 = np.stack([df['Traj_Infos/training_episode_success'] for df in df_1])
    y_2 = np.stack([df['Traj_Infos/training_episode_success'] for df in df_2])
    y_3 = np.stack([df['Traj_Infos/training_episode_success'] for df in df_3])

    x = df_1[0]['Diagnostics/CumSteps']
    with PdfPages(r'./training_tasks.pdf') as export_pdf:
        plot_with_std(x, y_1, 'Demonstrations', 'red')
        plot_with_std(x, y_2, 'Dense Rewards', 'blue')
        plot_with_std(x, y_3, 'Language Instructions', 'green')
        plt.xlabel('Environment Steps', fontsize=14)
        plt.ylabel('Average Trial Success', fontsize=14)
        plt.legend()
        plt.grid(True)
        # plt.show()
        export_pdf.savefig()
        plt.close()

    y_1 = np.stack([df['Traj_Infos/testing_episode_success'] for df in df_1])
    y_2 = np.stack([df['Traj_Infos/testing_episode_success'] for df in df_2])
    y_3 = np.stack([df['Traj_Infos/testing_episode_success'] for df in df_3])

    with PdfPages(r'./test_tasks.pdf') as export_pdf:
        plot_with_std(x, y_1, 'Demonstrations', 'red')
        plot_with_std(x, y_2, 'Dense Rewards', 'blue')
        plot_with_std(x, y_3, 'Language Instructions', 'green')
        plt.xlabel('Environment Steps', fontsize=14)
        plt.ylabel('Average Trial Success', fontsize=14)
        plt.legend()
        plt.grid(True)
        # plt.show()
        export_pdf.savefig()
        plt.close()
