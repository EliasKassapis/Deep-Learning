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
        path = './Results/RNN/'
        m = 'RNN'
        step_size = 100

        #Get stats for sequence length of 10
        train_loss = np.load(path + '10_RNN_loss.npy')
        train_acc = np.load(path + '10_RNN_accuracy.npy')

    elif model ==2:
        path = './Results/LSTM/'
        m = 'LSTM'
        step_size = 100

        #Get stats for sequence length of 10
        train_loss = np.load(path + '10_LSTM_loss.npy')
        train_acc = np.load(path + '10_LSTM_accuracy.npy')


    print('Final train accuracy: ', max(list(train_acc)), '\nFinal train loss: ', train_loss[-1])

    plot_graphs(train_loss, 'Loss', 'darkorange',
                # train_acc, 'Accuracy', 'darkblue',
                title= m + ': Loss vs. training time elapsed',
                ylabel='Loss',
                xlabel='No. of steps',
                step_size=step_size,
                m=m,
                path=path
                )

    plot_graphs(train_acc, 'Train', 'darkorange',
                title= m + ': Accuracy vs. training time elapsed',
                ylabel='Accuracy',
                xlabel='No. of steps',
                step_size=step_size,
                m=m,
                path=path
                )

get_graphs(2)


#Initialize lists of loss and accuracies for length comparisons
RNN_losses = []
RNN_accs = []
LSTM_losses = []
LSTM_accs = []

for length in [5,10,15,20,25,30,35,40,45,50]:
    for model in ['RNN','LSTM']:
        train_loss = np.load('./Results/' + model + '/' + str(length) + '_' + model + '_loss.npy')
        train_acc = np.load('./Results/' + model + '/' + str(length) + '_' + model + '_accuracy.npy')

        if model == 'RNN':
            RNN_losses.append(train_loss[-1])
            RNN_accs.append(train_acc[-1])
        elif model == 'LSTM':
            LSTM_losses.append(train_loss[-1])
            LSTM_accs.append(train_acc[-1])


# plot_graphs(RNN_losses, 'RNN', 'darkorange',
#             LSTM_losses, 'LSTM', 'darkblue',
#             title= 'Loss vs. Palindrome Length',
#             ylabel='Loss',
#             xlabel='Palindrome length',
#             step_size=5,
#             m='RNN vs. LSTM',
#             # m='RNN',
#             path= './Results/'
#             )
#
# plot_graphs(RNN_accs, 'RNN', 'darkorange',
#             LSTM_accs, 'LSTM', 'darkblue',
#             title= 'RNN Accuracy vs. Palindrome Length',
#             ylabel='Accuracy',
#             xlabel='Palindrome length',
#             step_size=5,
#             m='RNN vs. LSTM',
#             # m='RNN',
#             path= './Results/'
#             )
