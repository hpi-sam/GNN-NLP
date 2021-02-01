import random
import torch
from torch import utils
from torch import nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from gnn_nlp.DivGraphPointer import DivGraphDecoder, DivGraphEncoder, DivGraphNet
from gnn_nlp.config import TRAIN_SIZE
from torch.utils.tensorboard import SummaryWriter


#torch.manual_seed(0)

def init_parameters(embed_size, convs, hidden, rnns, opt, loss, tb, lr):
    diven = DivGraphEncoder(embed_size, embed_size, convs)
    divdec = DivGraphDecoder(embed_size, hidden, rnns)
    divnet = DivGraphNet(diven, divdec, loss, tb)
    optimiser = opt(divnet.parameters(), lr=lr)
    return divnet, optimiser

def split_data(data):

    train_len = int(len(data)*TRAIN_SIZE)
    train_ids = random.sample(range(len(data)), train_len)
    valid_ids = set(range(len(data)))-set(train_ids)
    train_list = [data[i] for i in train_ids]
    valid_list = [data[i] for i in valid_ids]

    return train_list, valid_list


def split_batches(size, data):
    batches = []
    ids = list(range(0, len(data), size))
    for i in range(len(ids)):
        if i != len(ids) - 1:
            batches.append(data[ids[i]:ids[i + 1]])
        else:
            batches.append(data[ids[i]:])
    return batches



def train_divnet(n_epochs, data_list, embed_size,
                 hidden, n_rnn_layers, n_conv_layers,
                 # TODO: attention weights already pass through Softmax in the model,
                 #  maybe use just NLLLoss instead?
                 batch_size,
                 criterion=nn.CrossEntropyLoss,
                 optimiser=optim.Adam,
                 tensorboard=True,
                 tb_text='',
                 lr=2e-4,
                 tb_folder='runs'):
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
                                        criterion,
                                        tensorboard,
                                        lr=lr,
                                        )
    train_list, valid_list = split_data(data_list)

    train_loader = split_batches(batch_size, train_list)
    valid_loader = split_batches(batch_size, valid_list)

    train_losses = {i: [] for i in range(n_epochs)}
    valid_losses = {i: [] for i in range(n_epochs)}

    train_kps = []
    valid_kps = []
    tb_writer = SummaryWriter(log_dir=tb_folder)
    if tb_text:
        tb_writer.add_text('divnet', tb_text)
    for epoch in range(n_epochs):
        print('Epoch', epoch)
        divnet.train()
        # TODO: if batch_size > 1, could the model learn to treat a batch as one single sequence?
        #  could it make the training less effective or not?
        for batch in train_loader:
            optimiser.zero_grad()
            preds, loss = divnet(batch)
            #import pdb;pdb.set_trace()
            loss.backward()
            optimiser.step()
            train_losses[epoch].append(loss.item())
            train_kps.append((batch, preds))

        #param_dict = dict(divnet.named_parameters())
        #for name in param_dict.keys():
        #    tb_writer.add_histogram(name.upper(),
        #                            param_dict[name],
        #                            epoch)
        #    if epoch != 0:
        #        tb_writer.add_histogram(name.upper() + '  GRAD',
        #                                param_dict[name].grad,
        #                                epoch)


        with torch.no_grad():
            divnet.eval()
            for valid_batch in valid_loader:
                valid_kp, valid_loss = divnet(valid_batch)
                valid_losses[epoch].append(valid_loss.item())
                valid_kps.append((valid_batch, valid_kp))
    tb_writer.close()
    return train_kps, train_losses, valid_kps, valid_losses


if __name__ == '__main__':

    import torch

    data = torch.load('../data/kp5DataList29-01-21.pt')
    train_kps, train_losses, valid_kps, valid_losses = train_divnet(n_epochs=3,
                                                                    data_list=data[:5],
                                                                    embed_size=100,
                                                                    hidden=300,
                                                                    n_rnn_layers=1,
                                                                    n_conv_layers=1,
                                                                    batch_size=2)