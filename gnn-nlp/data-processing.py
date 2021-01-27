from collections import OrderedDict
import string
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Dataset

import fasttext as ft
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
stops = ENGLISH_STOP_WORDS
signs = string.punctuation + '“' + '’' + '”' + '*'
punct_translator = str.maketrans('', '', signs)

ft_model = ft.load_model('../models/fil9SkipGram.bin')


def position_score(pos1, pos2):
    score = 0
    for i in pos1:
        for j in pos2:
            score += max(0, 1 / (i - j))
    return score


def position_dict(ordered_vocab):
    position_dict = OrderedDict()
    for word in ordered_vocab:
        if word not in position_dict.keys():
            position_dict[word] = [i for i, w in enumerate(ordered_vocab) if w == word]
    return position_dict


def score_matrix(ordered_vocab):
    pos = position_dict(ordered_vocab)
    A_left = []
    A_right = []
    for word in pos.keys():
        left = []
        right = []
        for word2 in pos.keys():
            if word == word2:
                left.append(0)
                right.append(0)
            else:
                left_score = position_score(pos[word], pos[word2])
                left.append(left_score)
                right_score = position_score(pos[word2], pos[word])
                right.append(right_score)
        A_left.append(left)
        A_right.append(right)
    return np.array(A_left), np.array(A_right), pos.keys()


def make_matrices(vocab):
    Aleft, Aright, columns = score_matrix(vocab)
    Aleft_tilda = torch.from_numpy(Aleft + np.identity(Aleft.shape[0]))
    Aright_tilda = torch.from_numpy(Aright + np.identity(Aright.shape[0]))
    return Aleft_tilda, Aright_tilda, list(columns)


def get_kw_index(vocab, word):
    # import pdb;pdb.set_trace()
    try:
        return vocab.index(word)
    except ValueError:
        for i, v in enumerate(vocab):
            if word in v:
                return i
        return False


def make_kw_labels(keywords, vocab):
    labels = []
    for key in keywords:
        kw_labels = []
        kw = key.split()
        for word in kw:
            ix = get_kw_index(vocab, word)
            assert ix in range(len(vocab))
            kw_labels.append([ix])
        kw_labels.append([len(vocab) - 1])
        if kw_labels not in labels:
            labels.append(kw_labels)
        else:
            keywords.remove(key)

    return labels


def index_kp(keywords, vocab):
    kp = []
    for key in keywords:
        kw = key.split()
        ids = []
        for word in kw:
            ix = get_kw_index(vocab, word)
            assert ix in range(len(vocab))
            ids.append(ix)
        kp.append(ids)
    return kp


def make_DataObject(row):
    vocab, keywords = row['clean'], row['keyword']
    # adding EndOfString token to the vocabulary
    vocab = vocab.split()
    vocab.append('<EOS>')
    Aleft_tilda, Aright_tilda, columns = make_matrices(vocab)
    X = torch.tensor([ft_model.get_sentence_vector(w).tolist() for w in columns])
    # print(columns)
    data = Data(x=X)
    data.a_left = Aleft_tilda
    data.a_right = Aright_tilda
    data.words = columns
    labels = make_kw_labels(keywords, vocab)
    data.labels = labels
    data.kp = index_kp(keywords, vocab)
    return data


def prepare_data(df):
    '''
    Function assumes df has the following columns: keyword and abstract
    :param df:
    :return:
    '''
    # cleaning the text and keywords first
    # TODO: find a clever way to keep punctuation inside a word
    df['clean'] = (df['abstract']
                   .apply(lambda x: x.translate(punct_translator).lower()))
    df['keyword'] = (df['keyword']
                     .apply(lambda x: [i.translate(punct_translator).lower() for i in x]))
    df['ptgeo_DataObj'] = df.apply(make_DataObject, axis=1)

    return df

