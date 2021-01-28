import random
import torch
from torch import utils
from torch import nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from gnn_nlp.DivGraphPointer import DivGraphDecoder, DivGraphEncoder, DivGraphNet
from gnn_nlp.config import TRAIN_SIZE

torch.manual_seed(0)

def init_parameters(embed_size, convs, hidden, rnns, opt, tb):
    diven = DivGraphEncoder(embed_size, embed_size, convs)
    divdec = DivGraphDecoder(embed_size, hidden, rnns)
    divnet = DivGraphNet(diven, divdec, tb)

    optimiser = opt(divnet.parameters(), lr=0.01)
    return divnet, optimiser

def split_data(data):

    train_len = int(len(data)*TRAIN_SIZE)
    train_ids = random.sample(range(len(data)), train_len)
    valid_ids = set(range(len(data)))-set(train_ids)
    train_list = [data[i] for i in train_ids]
    valid_list = [data[i] for i in valid_ids]

    return train_list, valid_list

def pad_nodes(data_list):
    max_size = max([len(data_list[i].words) for i in range(len(data_list))])
    for data in data_list:
        if len(data.x) < max_size:
            pad = torch.tensor([[-100] * data.x.size(1)] * (max_size - len(data.x)))
            data.x = torch.cat((data.x, pad), 0)
    return data_list

def pad_labels(data_list):
    pass

def prepare_data(data_list):

    data_list = pad_nodes(data_list)
    array = torch.stack([i.x for i in data_list])
    return array



def train_divnet(n_epochs, data_list, embed_size,
                 hidden, n_rnn_layers, n_conv_layers,
                 # TODO: attention weights already pass through Softmax in the model,
                 #  maybe use just NLLLoss instead?
                 criterion=nn.CrossEntropyLoss,
                 optimiser=optim.Adam,
                 tensorboard=True):
    '''

    :param n_epochs: int, number of epochs to train
    :param data_obj: A Python list of torch_geometric.data.Data objects
    :param embed_size: int, size of initial node representation
    :param hidden: int, size of a hidden layers for RNN in decoder
    :param n_rnn_layers: int, number of layers in RNN in decoder
    :param n_conv_layers: int, number of convolutional layers in encoder
    :param optimiser: non-initialised optimiser, e.g. torch.optim.Adam
    :param loss: torch.nn.CrossEntropyLoss or other
    :param tensorboard: bool, True if model parameters should be written to Tensorboard turing the training
    :return:
    '''

    #torch.autograd.set_detect_anomaly(True)
    divnet, optimiser = init_parameters(embed_size,
                                        n_conv_layers,
                                        hidden,
                                        n_rnn_layers,
                                        optimiser,
                                        tensorboard,)
    criterion = criterion()
    train_list, valid_list = split_data(data_list)


    train_loader = DataLoader(train_list, batch_size=1)
    valid_loader = DataLoader(valid_list, batch_size=1)

    train_losses = {i: [] for i in range(n_epochs)}
    valid_losses = {i: [] for i in range(n_epochs)}

    train_kps = []
    valid_kps = []
    #import pdb;pdb.set_trace()
    for epoch in range(n_epochs):
        optimiser.zero_grad()
        divnet.train()
        # TODO: if batch_size > 1, could the model learn to treat a batch as one single sequence?
        #  could it make the training less effective or not?
        for batch in train_loader:
            preds, attent_w = divnet(batch)
            # TODO: reshape preds, pad batch.labels if necessary
            loss = criterion(attent_w, batch.labels)
            loss.backward()
            optimiser.step()
            train_losses[epoch].append(loss.item())
            train_kps.append((batch, preds))

        with torch.no_grad():
            divnet.eval()
            for valid_batch in valid_loader:
                valid_kp, valid_loss = divnet(valid_batch, valid_batch.labels)
                valid_losses[epoch].append(valid_loss.item())
                valid_kps.append((valid_batch, valid_kp))

    return train_kps, train_losses, valid_kps, valid_losses


if __name__ == '__main__':

    import torch

    data = torch.load('../data/small_kpdfDataList.pt')
    train_kps, train_losses, valid_kps, valid_losses = train_divnet(n_epochs=3,
                                                                    data_list=data[:5],
                                                                    embed_size=100,
                                                                    hidden=300,
                                                                    n_rnn_layers=1,
                                                                    n_conv_layers=1,)