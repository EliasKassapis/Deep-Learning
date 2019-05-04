from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import matplotlib.pyplot as plt


def plot_graphs(*args, title=None, ylabel=None, xlabel=None, step_size=None, m=None, path = None):
    y = args[0::3]
    legends = args[1::3]
    colors = args[2::3]

    if title != None:
        plt.title(title)

    if xlabel != None:
        plt.xlabel(xlabel)

    if ylabel != None:
        plt.ylabel(ylabel)


    x = np.arange(step_size,(len(y[0])+1)*step_size,step_size)

    for i, current_y in enumerate(y):
        plt.plot(x, current_y, label=legends[i], color=colors[i])

    # plt.legend()
    fig1 = plt.gcf()
    plt.grid(True, lw = 0.75, ls = '--', c = '.75')
    plt.show()
    plt.draw()
    fig1.savefig(path + m + ' - ' + ylabel + '.png')



def get_graphs(model):
    if model == 1:
        path = './results/numpy results/'
        m = 'Numpy MLP'
        step_size = 100

    elif model ==2:
        path = './Results/'
        m = 'Generative LSTM'
        step_size = 100


    train_loss = np.load(path + 'epoch_20_loss.npy')
    train_acc = np.load(path + 'epoch_20_accuracy.npy')


    print(train_loss)

    print('Final train accuracy: ', max(list(train_acc)), '\nFinal train loss: ', train_loss[-1])

    plot_graphs(train_loss, 'Loss', 'darkblue',
                # train_acc, 'Accuracy', 'darkblue',
                title= m + ': Loss vs. training time elapsed',
                ylabel='Loss',
                xlabel='No. of steps',
                step_size=step_size,
                m=m,
                path=path
                )

    plot_graphs(train_acc, 'Train', 'darkblue',
                title= m + ': Accuracy vs. training time elapsed',
                ylabel='Accuracy',
                xlabel='No. of steps',
                step_size=step_size,
                m=m,
                path=path
                )

get_graphs(2)


texts = np.load('./Results/epoch_20_texts.npy')

#Get text for every interval of 5*100 steps
print('\nText generated\n---------------')
for s in range(0,24,5):
    print('Step ', str(s*100), ': ', texts[s])
