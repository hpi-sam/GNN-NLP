import os
import time
from datetime import datetime
import re
import string
from itertools import chain
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils


from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Dataset
import torch_geometric.transforms as T
from torch_geometric import utils
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GCNWeightedConv(MessagePassing):
    '''
    This implementation of GCN layer follows the PyTorchGeometric guidelines,
    but might not be as accurate.
    GCNWeightedConvM follows the paper more precisely, which is what needed for now
    in order to improve the understanding of the model.
    '''
    def __init__(self, in_channels, out_channels):
        super(GCNWeightedConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.left_lin = torch.nn.Linear(in_channels, out_channels)
        self.right_lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, left_deg, right_deg):

        self.x = x
        linear_x = self.lin(x)
        left_x = self.left_lin(x)
        right_x = self.right_lin(x)

        row, col = edge_index
        left_deg = left_deg.pow(-0.5)
        right_deg = right_deg.pow(-0.5)
        left_norm = left_deg[row] * left_deg[col]
        right_norm = right_deg[row] * right_deg[col]

        left = self.propagate(edge_index, x=left_x, norm=left_norm)
        right = self.propagate(edge_index, x=right_x, norm=right_norm)

        return left + right + linear_x

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class GCNWeightedConvM(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNWeightedConvM, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, X, A_left_tilda, A_right_tilda):
        '''
        This implementation overwrites the propagate() method.
        This way it is les efficient, but more clear and comparable to the paper.
        Later, when paper is understood better, it could be re-written in a more efficient way.

        :param X:
        :param A_left_tilda:
        :param A_right_tilda:
        :return:
        '''
        D_right = torch.diag_embed(A_right_tilda.sum(dim=1).pow(-0.5))
        D_left = torch.diag_embed(A_left_tilda.sum(dim=1).pow(-0.5))

        A_left_hat = D_left @ A_left_tilda
        A_right_hat = D_right @ A_right_tilda

        f = self.propagate(X, A_left_hat, A_right_hat)
        g = torch.sigmoid(f)

        activation = (f * g)
        if X.size() == activation.size():
            return X + activation
        return activation

    def propagate(self, X, A_left, A_right):
        left = self.lin(A_left.float() @ X.float())
        right = self.lin(A_right.float() @ X.float())
        linear = self.lin(X)

        return left + right + linear


class DivGraphNet(torch.nn.Module):
    '''
    Implementation of DivGraphPointer from https://arxiv.org/abs/1905.07689
    '''

    def __init__(self, encoder, decoder):
        super(DivGraphNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.y_W = nn.Linear(self.decoder.embed_size, self.decoder.embed_size)
        self.h_W = nn.Linear(self.decoder.embed_size, self.decoder.hidden_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch, labels):

        keyphrases = []
        # Encoder
        encoder_out = self.encoder(batch)
        context_vector = encoder_out.mean(axis=0)
        loss = 0
        nodes = torch.cat((encoder_out, torch.ones(1, encoder_out.size(1))), 0)
        coverage = torch.zeros(len(nodes), 1)
        # Decoder
        for l in range(len(labels)):

            if l == 0:
                # First input token is always <BOS> (BegginingOfString)
                input_token = torch.zeros((1, 1, self.decoder.embed_size))
                hidden = torch.tanh(self.h_W(context_vector)).unsqueeze(0).unsqueeze(0)
            else:
                ids = list(chain.from_iterable(keyphrases))
                kp = nodes[ids, :].mean(axis=0)

                cy = (context_vector * kp)
                input_token = self.y_W(cy).unsqueeze(0).unsqueeze(0)
                hidden = torch.tanh(self.h_W(cy)).unsqueeze(0).unsqueeze(0)

            keyphrases.append([])
            # Passing each word through decoder in order to predict (point) the next one
            for i in range(len(labels[l])):

                att_w, hidden, word_id = self.decoder(input_token,
                                                      hidden,
                                                      nodes,
                                                      coverage)

                if word_id == (len(nodes) - 1):
                    keyphrases[-1].append(word_id)
                    loss += self.loss(att_w.unsqueeze(0),
                                      torch.tensor(labels[l][i]))
                    break
                input_token = nodes[word_id].unsqueeze(0).unsqueeze(0)
                keyphrases[-1].append(word_id)
                coverage = coverage.clone()
                coverage[word_id] += 1
                # sum the loss up like this
                # or return predictions and compute the loss for the
                # whole batch in the training loop?
                loss += self.loss(att_w.unsqueeze(0),
                                  torch.tensor(labels[l][i]))

        return keyphrases, loss


class DivGraphEncoder(nn.Module):
    '''DivGraphEncoder'''
    def __init__(self, input_dim, hidden_dim, num_convs):
        super(DivGraphEncoder, self).__init__()

        self.num_convs = num_convs
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.convs.append(GCNWeightedConvM(input_dim, hidden_dim))
        for i in range(num_convs - 1):
            self.convs.append(GCNWeightedConvM(hidden_dim, hidden_dim))

    def forward(self, data_obj):
        '''Passing the data with its matrices through the conv layer GCNWeightedConvM'''
        x, a_left, a_right = data_obj.x, data_obj.a_left, data_obj.a_right

        for i in range(len(self.convs)):
            x = self.convs[i](x, a_left, a_right)
            # embedding = x
            # x = F.relu(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)
            # if not i == self.num_layers - 1:
            #    x = self.lns[i](x)
        return x


class DivGraphDecoder(torch.nn.Module):

    def __init__(self,
                 embed_size,
                 hidden_size,
                 rnn_layers,
                 bi=False,
                 batch_first=True):
        super(DivGraphDecoder, self).__init__()
        self.rnn = torch.nn.GRU(
            embed_size,
            hidden_size,
            rnn_layers,
            bidirectional=bi,
            batch_first=batch_first,
        )
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.att_v = nn.Linear(hidden_size, 1, bias=False)
        self.att_W_hidden = nn.Linear(hidden_size, hidden_size, bias=False)
        self.att_W_input = nn.Linear(embed_size, hidden_size, bias=False)
        self.att_W_coverage = nn.Linear(1, hidden_size)
        self.bi = 2 if bi else 1
        self.rnn_layers = rnn_layers

    def forward(self, word_input, word_hidden, nodes, coverage):
        '''
        DivGraphDecoder

        :param word_input: a vector representation of one word (node) of shape (batch, 1, embed)
        :param word_hidden: a context vector of shape (num_directions*num_layers, 1, hidden)
        :param nodes: representation of all the nodes as came out of the encoder of shape (n_words, embed)
        :param coverage: an array of size (1, n_words), shows how much each word has appeared in keywords so far
        :return:
        '''
        # GRU
        rnn_word, rnn_hidden = self.rnn(word_input, word_hidden)
        rnn_word = rnn_word.squeeze(0)
        hidden = rnn_hidden.squeeze(0)
        # Attention
        hidden = self.att_W_hidden(hidden).expand(len(nodes), self.hidden_size)
        term = hidden + self.att_W_input(nodes) + self.att_W_coverage(coverage)
        att_coef = self.att_v(torch.tanh(term)).squeeze(1)
        norm_att = F.softmax(att_coef, dim=0)
        next_word_id = norm_att.argmax().item()

        return norm_att, rnn_hidden, next_word_id