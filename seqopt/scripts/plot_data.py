import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_files', type=argparse.FileType('r'), nargs='+', help="Data files for same environment")
    parser.add_argument('--smooth', type=int, default=300, help="No. of models to smooth rewards over")
    args = parser.parse_args()

    # TODO(someshdaga):     Ensure that the same environment is used in each data file

    def plot_data(data, xaxis='Step', value="AverageEpRet", condition="Condition1", smooth=1, **kwargs):
        if smooth > 1:
            """
            smooth data with moving window average.
            that is,
                smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
            where the "smooth" param is width of that window (2k+1)
            """
            y = np.ones(smooth)
            for datum in data:
                x = np.asarray(datum[value])
                z = np.ones(len(x))
                smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
                # print(f"Smoothed size: {np.shape(smoothed_x)}, x size: {np.shape(x)}")
                datum[value] = smoothed_x

        if isinstance(data, list):
            data = pd.concat(data, ignore_index=True)
        sns.set(style="darkgrid", font_scale=1.5)
        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)

        """
        Changes the colorscheme and the default legend style, though.
        """
        # plt.legend(loc='best').set_draggable(True)
        # plt.legend(loc='upper center', ncol=3, handlelength=1,
        #           borderaxespad=0., prop={'size': 13})

        """
        For the version of the legend used in the Spinning Up benchmarking page, 
        swap L38 with:
        plt.legend(loc='upper center', ncol=6, handlelength=1,
                   mode="expand", borderaxespad=0., prop={'size': 13})
        """
        plt.legend(loc='upper center', ncol=6, handlelength=1,
                   mode="expand", borderaxespad=0., prop={'size': 13})

        print(f"x-axis: {np.asarray(data[xaxis])}")
        xscale = np.max(np.asarray(data[xaxis])) > 5e3
        if xscale:
            # Just some formatting niceness: x-axis scale in scientific notation if max x is large
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        plt.tight_layout(pad=0.5)

    # Create figure
    plt.figure()

    # Create a pandas dataframe combining all the data files
    datasets = []
    for data_file in args.data_files:
        data = np.load(data_file.name)
        exp_name = f"{data['algorithm'].item()}"

        # Get number of rewards per timestep
        num_rewards_per_step = data['rewards'].shape[-1]

        # Reshape the rewards into a 1D array
        rewards = data['rewards'].reshape(-1)

        # Repeat the time indices to correlate with the expanded rewards
        timesteps = np.repeat(data['step_nums'], num_rewards_per_step)

        # Assign the experiment name
        exp_names = len(timesteps) * [exp_name]

        ts_rews = np.asarray(list(zip(timesteps, rewards, exp_names)))

        ts_df = pd.DataFrame(ts_rews, columns=['Step', 'AverageEpRet', 'Experiment'])
        ts_df['Step'] = ts_df['Step'].astype(float).astype(int)
        ts_df['AverageEpRet'] = ts_df['AverageEpRet'].astype(float)
        datasets += [ts_df]

    # Plot all graphs
    plot_data(datasets, condition='Experiment', smooth=args.smooth)

    # Display the figure
    plt.show()

