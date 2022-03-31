import numpy as np
import torch
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os
import pandas as pd
import json

sns.set(style='whitegrid')

LOGS = 'logs'

save_path = "./plots"

def load_log(path):
    with open(path, 'r') as f:
        data = f.read().split('\n')
    epoch = []
    value = []
    for d in data:
        if len(d) == 0:
            continue
        j = json.loads(d)

        epoch.append(j['epoch'])
        value.append(j['acc'])

    return epoch, value


def make_plot(ylabel, fname):
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)

    plt.legend(loc='best', ncol=2)
    #plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname + '.png'))
    plt.close()


def plot_training_curve(y_label, model, train_eval):
    for i, file in enumerate(train_eval):
        epochs, value = [], []
        path = os.path.join(LOGS, model, file)
        try:
            seed_epochs, seed_value = load_log(path)
            epochs.extend(seed_epochs)
            value.extend(seed_value)
        except:
            print('Failed to load', path)
        if len(value) == 0:
            continue
        epochs = np.array(epochs)
        value = np.array(value)
        df = pd.DataFrame.from_dict(dict(
            epochs=epochs,
            value=value
        ))
        g = sns.lineplot(
            x='epochs',
            y='value',
            data=df,
            label=file.split('.')[0],
            ci='sd'
        )
    make_plot(y_label, model+"_acc")


if __name__ == '__main__':

    model = "small_cnn_mse"
    save_path = os.path.join(save_path, model)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_eval = ["eval.log"]

    plot_training_curve("loss", model, train_eval)