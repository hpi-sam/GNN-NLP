import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.nn import MessagePassing


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
        self.lin_left = torch.nn.Linear(in_channels, out_channels)
        self.lin_right = torch.nn.Linear(in_channels, out_channels)
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
        # TODO: add noise to A---tilda matrices, experiment what differences the amount of
        #  noise makes (e.g. add more noise to A_left, less A_right or viceversa) ?
        D_right = torch.diag_embed(A_right_tilda.sum(dim=1).pow(-0.5))
        D_left = torch.diag_embed(A_left_tilda.sum(dim=1).pow(-0.5))

        A_left_hat = D_left @ A_left_tilda
        A_right_hat = D_right @ A_right_tilda

        f = self.propagate(X, A_left_hat, A_right_hat)
        # TODO: try Relu instead of sigmoid
        g = torch.sigmoid(f)

        activation = (f * g)
        if X.size() == activation.size():
            return X + activation

        return activation

    def propagate(self, X, A_left, A_right):
        left = self.lin_left(A_left.float() @ X.float())
        right = self.lin_right(A_right.float() @ X.float())
        linear = self.lin(X)

        return left + right + linear


class DivGraphNet(torch.nn.Module):
    '''
    Implementation of DivGraphPointer from https://arxiv.org/abs/1905.07689
    '''

    def __init__(self, encoder, decoder, loss, tensorboard=True, max_kp=5):
        super(DivGraphNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss()
        self.hsize = self.decoder.hidden_size
        self.esize = self.decoder.embed_size
        self.y_W = nn.Linear(self.esize, self.esize)
        self.h_W = nn.Linear(self.esize, self.hsize)
        self.lrelu = nn.ReLU()
        if tensorboard:
            self.tb = SummaryWriter(log_dir='runs/ATT_VvsCY/')
        self.max_kp = max_kp

    def get_labels(self, batch, kp, word):
        labels = [x.labels for x in batch]
        target = []
        for i in range(len(labels)):
            if (kp < len(labels[i])):
                if (word < len(labels[i][kp])):
                    target.append(labels[i][kp][word])
                else:
                    target.append(batch[i].x.size(0)-1) # TODO: append -100 instead?
                    #target.append(-100)
            else:
                target.append(-100)
        return torch.tensor(target)


    def forward(self, batch, test=False):

        batch_size = len(batch)
        keyphrases = []
        used_words = {i: set() for i in range(batch_size)}
        last_ids = torch.tensor([[len(i.words)-1] for i in batch])
        # Encoder
        nodes = self.encoder(batch)
        # nodes shape: (8, 100)

        context_vectors = nodes.mean(axis=1) # TODO: mean through rows or columns ?
        # TODO: instead check the following: 1) how is EOS encoded? 2) does EOS always need to be [1, ...., 1]?
        #nodes = torch.cat((encoder_out, torch.ones(1, encoder_out.size(1))), 0)
        coverages = torch.zeros(batch_size, nodes.size(1), 1)
        #att_ws = []
        # Decoder
        loss = 0
        for l in range(self.max_kp):
            #print('##############################################################################')
            #print('Generating keyphrase', l)
            # Choosing keyphrases in this loop
            if l == 0:
                # First input token is always <BOS> (BegginingOfString)
                input_tokens = torch.zeros((batch_size, 1, self.decoder.embed_size)) # (4, 1, 100)
                # TODO: this reshaping only works if number of rnn layers multiplied
                #  with number of directions equals 1. To experiment with more layers
                #   or directions, 2d hidden state should be created carefully. In the
                #  paper it's not described how to do that

                hiddens = (self.h_W(context_vectors)
                          .expand(batch_size, self.hsize).unsqueeze(0)) # (num_layers*num_dirs, batch, 100)
            else:
                #nodes[torch.arange(batch_size).unsqueeze(-1), keyphrases[-1]].mean(axis=1)

                #ids = [list(set(chain.from_iterable(i))) for i in keyphrases]
                kp_mean = (torch.stack([nodes[i][list(used_words[i]), :].mean(axis=0)
                           for i in range(batch_size)]))
                cy = (context_vectors * kp_mean) # TODO: concatenate them instead?
                # TODO: shouldnt an input token be the one RNN outputted?
                '''Added LReLu'''
                input_tokens = self.y_W(cy).reshape((batch_size, 1, nodes.size(2))) # shape (batch, 1, embed)
                hiddens = self.h_W(cy).unsqueeze(0)
            n_words = 0
            eos = torch.tensor([[False]]*batch_size)
            # Passing each word through decoder in order to predict (point) the next one
            while True:
                #print('Picking words for the phrase', l)
                #print('Number of words', n_words)
                # Choosing words of keyphrases in this loop
                att_w, hidden, word_id, term = self.decoder(input_tokens,
                                                          hiddens,
                                                          nodes,
                                                          coverages)
                self.tb.add_histogram('ATT_V TERM', term, l)
                if l != 0:
                    self.tb.add_histogram('cY', cy, l)

                #param_dict = dict(self.named_parameters())
                #self.tb.add_histogram('ATTENTION',
                #                      att_w,
                #                      n_words)

                for i in range(batch_size):
                    used_words[i].add(word_id[i].item())
                if n_words == 0:
                    kp = word_id
                else:
                    kp = torch.cat((kp, word_id), dim=1)

                # TODO: LOSS:
                #  - for each word? each keyphrase? sum them up?
                if not test:
                    target = self.get_labels(batch, l, n_words)
                    loss += self.loss(att_w, target)
                n_words += 1

                    #words_att.append(att_w)
                eos = (eos | (word_id == last_ids))
                if (sum(eos).item() == batch_size) or (n_words > 10):
                    kp = torch.cat((kp, word_id), dim=1)
                    #att_ws.append(att_w)
                    # TODO: return loss as well
                    break

                slice = torch.arange(batch_size).unsqueeze(-1)
                input_tokens = nodes[slice, word_id]
                coverages = coverages.clone()
                coverages[slice, word_id] += 1
            keyphrases.append(kp)


        return keyphrases, loss


class DivGraphEncoder(nn.Module):
    '''DivGraphEncoder'''
    def __init__(self, input_dim, hidden_dim, num_convs):
        super(DivGraphEncoder, self).__init__()

        self.num_convs = num_convs
        #self.linear = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.convs.append(GCNWeightedConvM(input_dim, hidden_dim))
        for i in range(num_convs - 1):
            self.convs.append(GCNWeightedConvM(hidden_dim, hidden_dim))


    def pad_arrays(self, tensors):
        max_size = len(max(tensors, key=len))
        padded = []
        for data in tensors:
            if len(data) < max_size:
                pad = torch.tensor([[-100] * data.size(1)] * (max_size - len(data)))
                data = torch.cat((data, pad), 0)
            padded.append(data)
        return padded

    def forward(self, batch): # [Data(x=(8, 100), labels=2, words=8), Data(x=(15, 100), labels=5)]
        '''Passing the data with its matrices through the conv layer GCNWeightedConvM'''
        arrays = []
        for object in batch:
            x, a_left, a_right = object.x, object.a_left, object.a_right

            for i in range(len(self.convs)):
                x = self.convs[i](x, a_left, a_right)
                x = self.drop(x)
                # TODO: Add DropOut and activation if there is more than 1 conv
                # x = F.relu(x)
                # x = F.dropout(x, p=self.dropout, training=self.training)
            arrays.append(x)
        shapes = {len(a) for a in arrays}
        # if not all the tensors in the *arrays* have the same shape...
        if len(shapes) > 1:
            # ...pad them
            arrays = self.pad_arrays(arrays)

        return torch.stack(arrays) # (batch, long_article, gcnConvOutputDim)


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
            #dropout=0.5,
        )
        self.lrelu = nn.ReLU()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        # TODO: what's so special about att_v?
        self.att_v = nn.Sequential(nn.Linear(hidden_size, 1),
                                   nn.Dropout(0.5))
        self.att_W_hidden = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                          nn.Dropout(0.5))
        self.att_W_input = nn.Sequential(nn.Linear(embed_size, hidden_size),
                                         nn.Dropout(0.5))
        self.att_W_coverage = nn.Sequential(nn.Linear(1, hidden_size),
                                            nn.Dropout(0.5))
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
        # TODO: vanishing gradients are sensitive to the batch size:
        #  1) the larger the batch size --> more van. grads
        #  2) try training with different batch sizes (e.g. batches with diff % of stopwords)
        #  '''Reasons: sparsity, stopwords?'''
        rnn_word, rnn_hidden = self.rnn(word_input, word_hidden) #BOS
        # TODO: shouldnt rnn_word be used as the next input to rnn?
        #rnn_word = rnn_word.squeeze(0)
        #print('RNN HIDDEN', rnn_hidden)
        hidden = self.lrelu((rnn_hidden.reshape((word_input.size(0), 1, self.hidden_size))
                  .expand(word_input.size(0), nodes.size(1), self.hidden_size)))
        # Attention
        term = self.att_W_hidden(hidden) + self.att_W_input(nodes) + self.att_W_coverage(coverage)
        att_coef = self.att_v(torch.tanh(term))
        norm_att = F.softmax(att_coef, dim=1)
        #print('AFTER SOFTMAX', norm_att)
        next_word_id = norm_att.argmax(axis=1)


        return norm_att.squeeze(2), rnn_hidden, next_word_id, torch.tanh(term)