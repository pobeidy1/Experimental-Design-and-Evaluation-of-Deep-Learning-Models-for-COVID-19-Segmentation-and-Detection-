import os
import numpy as np
import matplotlib.pyplot as plt

figs_dir = os.path.join(os.getcwd(), 'figures')
try:
    os.makedirs(figs_dir, exist_ok=False)
    print('Directory successfully created')
except OSError as error:
    print('Directory already exist')


def training_stats(stats_log, model, hideplot=False):
    """
    :param stats_log:  a dictionary with the metrics from the training and validation stage
    :param model: a string with the name of the trained model
    :param hideplot: a boolean to display or not the stats plots
    :return: the plots
    """

    # Plot the mean loss per epoch
    fig1 = plt.figure(figsize=(8, 4))
    x = np.arange(1, len(stats_log['train_loss']) + 1)
    plt.plot(x, stats_log['train_loss'], label='train')
    inter = len(x)//3 if len(x) % 2 == 1 else len(x)/2
    plt.xticks(range(1, len(x) + 1, inter))
    title = 'Loss trend ' + model
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    fname = f'{model}_loss.png'
    plt.savefig(os.path.join(figs_dir, fname), bbox_inches='tight',
                format='png', dpi=500)
    if hideplot:
        plt.close(fig1)
    else:
        plt.rcParams.update({'font.size': 10})
        plt.show(fig1)

    # Plot Average precision per epoch @ IoU=0.50:0.95
    fig2 = plt.figure(figsize=(8, 4))
    plt.plot(x, stats_log['mAPb'], label='AP boxes')
    plt.plot(x, stats_log['mAPseg'], label='AP seg')
    plt.xticks(range(1, len(x) + 1, inter))
    title = 'Average Precision ' + model
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Average Precision')
    plt.legend(loc='lower right')
    fname = f'{model}_apiou_all.png'
    plt.savefig(os.path.join(figs_dir, fname), bbox_inches='tight',
                format='png', dpi=500)

    if hideplot:
        plt.close(fig2)
    else:
        plt.rcParams.update({'font.size': 10})
        plt.show(fig2)

    # Plot Average precision @ IoU=0.50 per epoch
    fig3 = plt.figure(figsize=(8, 4))
    plt.plot(x, stats_log['mAPb50'], label='AP boxes, IoU=0.5')
    plt.plot(x, stats_log['mAP50seg'], label='AP seg, IoU=0.5')
    plt.xticks(range(1, len(x) + 1, inter))
    title = 'Average Precision ' + model
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Average Precision')
    plt.legend(loc='lower right')
    fname = f'{model}_apiou50.png'
    plt.savefig(os.path.join(figs_dir, fname), bbox_inches='tight',
                format='png', dpi=500)
    if hideplot:
        plt.close(fig3)
    else:
        plt.rcParams.update({'font.size': 10})
        plt.show(fig3)

    # Plot Average precision @ IoU=0.75
    fig4 = plt.figure(figsize=(8, 4))
    plt.plot(x, stats_log['mAPb75'], label='AP boxes, IoU=0.75')
    plt.plot(x, stats_log['mAP75seg'], label='AP seg, IoU=0.75')
    plt.xticks(range(1, len(x) + 1, inter))
    title = 'Average Precision ' + model
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Average Precision')
    plt.legend(loc='lower right')
    fname = f'{model}_apiou75.png'
    plt.savefig(os.path.join(figs_dir, fname), bbox_inches='tight',
                format='png', dpi=500)
    if hideplot:
        plt.close(fig4)
    else:
        plt.rcParams.update({'font.size': 10})
        plt.show(fig4)

    return fig1, fig2, fig3, fig4