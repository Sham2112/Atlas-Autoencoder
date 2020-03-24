import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import stats


def plotdata(data, logy=False):  # taken from https://github.com/erwulff/lth_thesis_project
    plt.figure()
    plt.hist(data['m'], bins=100)
    plt.xlabel('m')
    plt.ylabel('No of jets')
    if logy:
        plt.yscale('log')

    plt.figure()
    plt.hist(data['pt'], bins=100)
    plt.xlabel('pt')
    plt.ylabel('No of jets')
    if logy:
        plt.yscale('log')

    plt.figure()
    plt.hist(data['phi'], bins=100)
    plt.xlabel('phi')
    plt.ylabel('No of jets')
    if logy:
        plt.yscale('log')

    plt.figure()
    plt.hist(data['eta'], bins=100)
    plt.xlabel('eta')
    plt.ylabel('No of jets')
    if logy:
        plt.yscale('log')

class AE_3d_200(nn.Module):
    def __init__(self):
        super(AE_3d_200, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(4,200),
            nn.ReLU(True),
            nn.Linear(200,100),
            nn.ReLU(True),
            nn.Linear(100,50),
            nn.ReLU(True),
            nn.Linear(50,3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3,50),
            nn.ReLU(True),
            nn.Linear(50,100),
            nn.ReLU(True),
            nn.Linear(100,200),
            nn.ReLU(True),
            nn.Linear(200,4)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AE_3d_small(nn.Module):
    def __init__(self):
        super(AE_3d_small, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(4,75),
            nn.ReLU(True),
            nn.Linear(75,30),
            nn.ReLU(True),
            nn.Linear(30,3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3,30),
            nn.ReLU(True),
            nn.Linear(30,75),
            nn.ReLU(True),
            nn.Linear(75,4)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AE_3d_big(nn.Module):
    def __init__(self):
        super(AE_3d_big, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(4,100),
            nn.ReLU(True),
            nn.Linear(100,250),
            nn.ReLU(True),
            nn.Linear(250,600),
            nn.ReLU(True),
            nn.Linear(600,250),
            nn.ReLU(True),
            nn.Linear(250,100),
            nn.ReLU(True),
            nn.Linear(100,3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3,100),
            nn.ReLU(True),
            nn.Linear(100, 250),
            nn.ReLU(True),
            nn.Linear(250, 600),
            nn.ReLU(True),
            nn.Linear(600,250),
            nn.ReLU(True),
            nn.Linear(250,100),
            nn.ReLU(True),
            nn.Linear(100,4)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AE_3d_LeakyReLU(nn.Module):
    def __init__(self):
        super(AE_3d_LeakyReLU, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(4,200),
            nn.LeakyReLU(True),
            nn.Linear(200,100),
            nn.LeakyReLU(True),
            nn.Linear(100,50),
            nn.LeakyReLU(True),
            nn.Linear(50,3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3,50),
            nn.LeakyReLU(True),
            nn.Linear(50,100),
            nn.LeakyReLU(True),
            nn.Linear(100,200),
            nn.LeakyReLU(True),
            nn.Linear(200,4)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AE_3d_tanh(nn.Module):
    def __init__(self):
        super(AE_3d_tanh, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(4,200),
            nn.Tanh(),
            nn.Linear(200,100),
            nn.Tanh(),
            nn.Linear(100,50),
            nn.Tanh(),
            nn.Linear(50,3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3,50),
            nn.Tanh(),
            nn.Linear(50,100),
            nn.Tanh(),
            nn.Linear(100,200),
            nn.Tanh(),
            nn.Linear(200,4)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AE_3d_dropout(nn.Module):
    def __init__(self):
        super(AE_3d_dropout, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(4,200),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(200,100),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(100,50),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(50,3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3,50),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(50,100),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(100,200),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(200,4)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AE_3d_bn(nn.Module):
    def __init__(self):
        super(AE_3d_bn, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(4,200),
            nn.BatchNorm1d(200),
            nn.ReLU(True),
            nn.Linear(200,100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100,50),
            nn.BatchNorm1d(50),
            nn.ReLU(True),
            nn.Linear(50,3),
            nn.BatchNorm1d(3),
        )

        self.decoder = nn.Sequential(
            nn.Linear(3,50),
            nn.BatchNorm1d(50),
            nn.ReLU(True),
            nn.Linear(50,100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100,200),
            nn.BatchNorm1d(200),
            nn.ReLU(True),
            nn.Linear(200,4),
            nn.BatchNorm1d(4),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AE_3d_bn_dropout(nn.Module):
    def __init__(self):
        super(AE_3d_bn_dropout, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(4,200),
            nn.BatchNorm1d(200),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(200,100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(100,50),
            nn.BatchNorm1d(50),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(50,3),
            nn.BatchNorm1d(3),
        )

        self.decoder = nn.Sequential(
            nn.Linear(3,50),
            nn.BatchNorm1d(50),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(50,100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(100,200),
            nn.BatchNorm1d(200),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(200,4),
            nn.BatchNorm1d(4),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def fit(n_epochs, criterion, optimizer, model, trainloader, validloader, device):
    '''
    A function to train a given model that will return the model with the least validation loss, training losses and validatin losses
    '''

    valid_loss_min = np.Inf  # track change in validation loss
    train_losses = []
    valid_losses = []

    for e in range(n_epochs):
        running_train_loss = 0.
        model.train()
        for x, y in trainloader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        running_valid_loss = 0
		
        with torch.no_grad():
            for x, y in validloader:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                running_valid_loss += criterion(y_hat, y).item()

        train_loss = running_train_loss / len(trainloader)
        valid_loss = running_valid_loss / len(validloader)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)


        print("Epoch: {}, Training loss: {}, Validation loss: {}".format(e + 1, train_loss, valid_loss))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({} --> {}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss

    model.load_state_dict(torch.load('model.pt'))

    return model, train_losses, valid_losses


def std_error(x, axis=None, ddof=0):
    return np.nanstd(x, axis=axis, ddof=ddof) / np.sqrt(2 * len(x))

def plot_results(model, test_x, train_mean, train_std ,savefig=False, range = (-0.5, 0.5)):
    '''
    -Plots histograms of different variables before being compressed and after being uncompresses for comparison
    -Also plots the residuals for each variable
    -Returns Residuals of each variable
    -Taken from - https://github.com/Skelpdar/HEPAutoencoders
    '''
    plt.close('all')
    unit_list = ['[GeV]', '[rad]', '[rad]', '[GeV]']
    variable_list = [r'$m$', r'$p_T$', r'$\phi$', r'$\eta$']
    line_style = ['--', '-']
    colors = ['orange', 'c']
    markers = ['*', 's']

    model.to('cpu')

    # Histograms
    idxs = (0, 100000)  # Choose events to compare
    data = (torch.tensor(test_x[idxs[0]:idxs[1]].values).float())
    pred = model(data).detach().numpy()
    pred = np.multiply(pred, train_std.values)
    pred = np.add(pred, train_mean.values)
    data = np.multiply(data, train_std.values)
    data = np.add(data, train_mean.values)

    alph = 0.8
    n_bins = 50
    for kk in np.arange(4):
        plt.figure(kk + 4)
        n_hist_data, bin_edges, _ = plt.hist(data[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
        n_hist_pred, _, _ = plt.hist(pred[:, kk], color=colors[0], label='Output', alpha=alph, bins=bin_edges)
        plt.suptitle(train_x.columns[kk])
        plt.xlabel(variable_list[kk] + ' ' + unit_list[kk])
        plt.ylabel('Number of events')
        # plt.yscale('log')
        plt.legend()
        fig_name = 'trainforever_hist_%s' % traindata.columns[kk]
        #plt.savefig(curr_save_folder + fig_name)
    
    residual_strings = [r'$(p_{T,out} - p_{T,in}) / p_{T,in}$',
                        r'$(\eta_{out} - \eta_{in}) / \eta_{in}$',
                        r'$(\phi_{out} - \phi_{in}) / \phi_{in}$',
                        r'$(E_{out} - E_{in}) / E_{in}$']
    
    
    residuals = (pred - data.detach().numpy()) / data.detach().numpy()
    for kk in np.arange(4):
        plt.figure()
        n_hist_pred, bin_edges, _ = plt.hist(
            residuals[:, kk], label='Residuals', linestyle=line_style[0], alpha=alph, bins=100, range=range)
        plt.suptitle('Residuals of %s' % traindata.columns[kk])
        plt.xlabel(residual_strings[kk])  
        plt.ylabel('Number of jets')
        std = np.std(residuals[:, kk])
        std_err = std_error(residuals[:, kk])
        mean = np.nanmean(residuals[:, kk])
        sem = stats.sem(residuals[:, kk], nan_policy='omit')
        ax = plt.gca()
        plt.text(.75, .8, 'Mean = %f$\pm$%f\n$\sigma$ = %f$\pm$%f' % (mean, sem, std, std_err), bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10},
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=18)
        fig_name = 'trainforever_residual_%s' % traindata.columns[kk]

        if savefig:    
            plt.savefig(curr_save_folder + fig_name)
