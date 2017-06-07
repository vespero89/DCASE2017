#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 17:36:52 2017

@author: buckler
"""
import os
import numpy as np
from keras import backend as T

# def __init__(id):
#     global logger
#     logger = logging.getLogger(str(id))
#     # def crateLogger(id, logToFile):
#     #     global logger
#     #     logger = u.MyLogger(id, logToFile)

def load_A3FALL(spectrogramsPath):
    """
    Carica tutto il dataset (spettri) in una lista di elementi [filename , matrix ]
    """
    print("Loading A3FALL dataset")
    a3fall = list()
    for root, dirnames, filenames in os.walk(spectrogramsPath):
        i = 0
        for file in filenames:
            matrix = np.load(os.path.join(root, file))
            data = [file, matrix]
            a3fall.append(data)
            i += 1
    return a3fall


def awgn_padding_set(set_to_pad, loc=0.0, scale=1.0):
    print("awgn_padding_set")

    # find matrix with biggest second axis
    dim_pad = np.amax([len(k[1][2]) for k in set_to_pad])
    awgn_padded_set = []
    for e in set_to_pad:
        row, col = e[1].shape
        # crete an rowXcol matrix with awgn samples
        awgn_matrix = np.random.normal(loc, scale, size=(row, dim_pad - col))
        awgn_padded_set.append([e[0], np.hstack((e[1], awgn_matrix))])
    return awgn_padded_set

def remove_padding_set(set, label, origin_set):
    """
    remove the padding added previously from the set.
    
    :param label: label of the set (shape list of string)
    :param set: set from which to remove the padding (shape: number matrix)
    :param origin_set: the original set (list: string, matrix_sample)
    :return: a list representing the set without the padding ( the list is needed because each element has different shape) 
    """
    newSet = list()
    for (s, l, o) in zip(set, label, origin_set):
        if l == o[0]: #TODO in realtà questo controlllo non è necessario e si portrebbe togliere "label" dai parametri passati
            newSet.append( s[0, :, :o[1].shape[1]])

    return newSet

def assert_matrix_is_not_all_zero(data):
    """
    assert that all the matrix in the list named "data" are not all zeros. 
    If a matrix is all zeros set the first element: data[x][0,0] = keras.backend.epsilon()
    :param data: list of matrix ( although various dimensions)  
    :return: 
    """
    for d in data:
        # print(str(img.max()))
        if d.max() == 0:
            d[0, 0] = T.epsilon() #python use references!!!
            # print("changed:"+str(img[0,0])+" Into:"+str(img[0,0]))

    return data

def reshape_set(set_to_reshape, channels=1):
    """
    reshape the data in a form that keras want:
        -for theano dim ordering: (nsample, channel, row ,col)
        -for tensorflow not supported yet
        -other not specified yet
    :param set_to_reshape:
    :param channels:
    :return:
    """
    print("reshape_set")
    n_sample = len(set_to_reshape)
    row, col = set_to_reshape[0][1].shape
    label = []
    shaped_matrix = np.empty((n_sample, channels, row, col))
    for i in range(len(set_to_reshape)):
        label.append(set_to_reshape[i][0])
        shaped_matrix[i][0] = set_to_reshape[i][1]
    return shaped_matrix, label


def split_A3FALL_simple(data, train_tag=None):
    '''
    Splitta il dataset in train e test set: train tutti i background, mentre test tutto il resto
    (da amplicare in modo che consenta lo split per la validation)
    '''
    print("split_A3FALL_simple")

    if train_tag == None:
        train_tag = ['classic_', 'rock_', 'ha_']
    #        if test_tag=None:
    #            test_tag=[]

    data_train = [d for d in data if any(word in d[0] for word in
                                         train_tag)]  # controlla se uno dei tag è presente nnel nome del file e lo assegna al trainset
    data_test = [d for d in data if d not in data_train]  # tutto cioò che non è train diventa test

    return data_train, data_test


def split_A3FALL_from_lists(data, listpath, namelist):
    '''
    Richede in ingresso la cartella dove risiedono i file di testo che elencano i vari segnali che farano parte di un voluto set di dati.
    Inltre in namelist vanno specificati i nomi dei file di testo da usare.
    Ritorna una lista contentete le liste dei dataset di shape: (len(namelist),data.shape)
    '''
    print("split_A3FALL_from_lists")

    sets = list()
    for name in namelist:
        sets.append(select_list(os.path.join(listpath, name), data))
    return sets


def select_list(filename, dataset):
    '''
    Dato in ingesso un file di testo, resituisce una array contenete i dati corrispondenti elencati nel file
    '''
    print("select_list")

    subset = list()
    with open(filename) as f:
        content = f.readlines()
        content = [x.strip().replace('.wav', '.npy') for x in content]  # remove the '\n' at the end of string
        subset = [s for s in dataset if
                  any(name in s[0] for name in content)]  # select all the data presetn in the list
    return subset


def normalize_data(data, mean=None, std=None):
    '''
    normalizza media e varianza del dataset passato
    se data=None viene normalizzato tutto il dataset A3FALL
    se mean e variance = None essi vengono calcolati in place sui data
    '''
    print("normalize_data")

    if bool(mean) ^ bool(std):  # xor operator
        raise ("Error!!! Provide both mean and variance")
    elif mean == None and std == None:  # compute mean and variance of the passed data
        data_conc = concatenate_matrix(data)
        mean = np.mean(data_conc)
        std = np.std(data_conc)

    data_std = [[d[0], ((d[1] - mean) / std)] for d in data]  # normalizza i dati: togle mean e divide per std

    return data_std, mean, std


def concatenate_matrix(data):
    """
    concatena gli spettri in un unica matrice: vule una lista e restituisce un array

    :param data:
    :return:
    """

    print("concatenate_matrix")

    data_ = data.copy()
    data_.pop(0)
    matrix = data[0][1]
    for d in data_:
        np.append(matrix, d[1], axis=1)
    return matrix


def labelize_data(y):#TODO usare sklearn.preprocessing.LabelEncoder ?
    """
    labellzza numericamente i nomi dei file
    assegna 1 se è una caduta del manichino, 0 altrimenti
    :param y:
    :return:

    """
    print("labelize_data")

    i = 0
    numeric_labels = list()
    for d in y:
        if 'rndy' in d:
            numeric_labels.append(1)
        else:
            numeric_labels.append(0)
        i += 1

    return numeric_labels