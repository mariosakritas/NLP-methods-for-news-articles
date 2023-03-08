#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:23:31 20190
@author: marios
"""


import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, pickle, urllib.request
import os.path as op
from collections import Counter
from tqdm.notebook import tqdm
import click
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchtext
from torchtext.data import get_tokenizer

from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import spacy


def load_data(db_path):
    # !mkdir /root/data/
    #
    # # the link is from the dropbox data folder
    #
    # if not os.path.isfile('/root/data/merged_training.pkl'):
    #     !wget db_path - O / root / data / merged_training.pkl
    # else:
    #     print('Training data exists.')
    if db_path is None:
        db_path = '/Users/user/NMA_DEEP/merged_training.pkl'

    with open(db_path, 'rb') as obj:
        data = pickle.load(obj)

    X = data.text.values
    y = data.emotions.values
    for i, em in enumerate(np.unique(y)):
        print(em)
        ind = np.where(y == em)
        y[ind] = i

    return X, y

def resample(x_train, y_train, seed=0):
    np.random.seed(seed)
    data = pd.DataFrame(data = {'text': x_train, 'emotions':y_train})
    emotions = np.unique(data.emotions)
    resample_target = round(len(data) // len(emotions ))
    print('resample target: {}'.replace(resample_target))

    resampled_data = pd.DataFrame()
    for em in data.emotions.unique():
        em_data = data.loc[data.emotions == em]
        resample_idx = np.random.choice(em_data.index, replace=True, size=resample_target)
        resample_em = em_data.loc[resample_idx]
        resampled_data = pd.concat((resampled_data, resample_em))

    return resampled_data.text.values, resampled_data.emotions.values

def tokenize(train_data, test_data):

    tokenizer = get_tokenizer("basic_english")
    train_token = [tokenizer(s) for s in tqdm(train_data)]
    test_token = [tokenizer(s) for s in tqdm(test_data)]

    return train_token, test_token

def get_sorted_words(train_tokens):
    words = Counter()
    for s in train_tokens:
        for w in s:
            words[w] += 1

    sorted_words = list(words.keys())
    sorted_words.sort(key=lambda w: words[w], reverse=True)
    print(f"Number of different Tokens in our Dataset: {len(sorted_words)}")

    return sorted_words

#CHANGE THIS FUNCTION TO TAKE SENTENCES NOT VOCAB
def remove_stopwords(text):
    nltk.download('stopwords')
    stopwords_ = stopwords.words('english')
    # additional stop words, PLEASE ADD THEM IN THIS LIST AS YOU RUN INTO THEM
    add_stopwords = ['im', 'ive']
    stopwords_ = stopwords_ + add_stopwords
    text_no_stopwords = []
    # text is a string
    for txt in text:

        txt = txt.lower().split()

        a = set(txt)
        b = set(stopwords_)
        c = list(a - b)
        text_no_stopwords.append(' '.join(c))
    return text_no_stopwords


def link_entities(text):
    # here we do named entity linking
    nlp = spacy.load("en_core_web_sm")
    linked_entities_tweets = []
    for tweet in text:
        doc = nlp(tweet)
        for ent in doc.ents:
            tweet = tweet.replace(ent.text, ent.label_.lower())
        linked_entities_tweets.append(tweet)
        if len(linked_entities_tweets) % 10000 == 0:
            print(len(linked_entities_tweets))
    return text

def get_stem_func(alg):

    if alg =='porter':
        func = stem_porter
    elif alg == 'wnl':
        func = stem_wnl

    return func

def stem_porter(text):
    porter = PorterStemmer()
    temp = [porter.stem(word) for word in text.split()]

    # return a string instead
    return ' '.join(temp)

def stem_wnl(text):
    wnl = WordNetLemmatizer()
    temp = [wnl.lemmatize(word) for word in text.split()]

    # return a string instead
    return ' '.join(temp)


def make_word_to_idx_dict(sorted_words, num_words_dict):

    # We reserve two numbers for special tokens.
    most_used_words = sorted_words[:num_words_dict - 2]

    # dictionary to go from words to idx
    word_to_idx = {}
    # dictionary to go from idx to words (just in case)
    idx_to_word = {}

    # We include the special tokens first
    PAD_token = 0
    UNK_token = 1

    word_to_idx['PAD'] = PAD_token
    word_to_idx['UNK'] = UNK_token

    idx_to_word[PAD_token] = 'PAD'
    idx_to_word[UNK_token] = 'UNK'

    # We popullate our dictionaries with the most used words
    for num, word in enumerate(most_used_words):
        word_to_idx[word] = num + 2
        idx_to_word[num + 2] = word

    return word_to_idx, idx_to_word


def tokens_to_idx(sentences_tokens, word_to_idx):
    # A function to convert list of tokens to list of indexes

    sentences_idx = []
    for sent in sentences_tokens:
        sent_idx = []
        for word in sent:
            if word in word_to_idx:
                sent_idx.append(word_to_idx[word])
            else:
                sent_idx.append(word_to_idx['UNK'])
        sentences_idx.append(sent_idx)
    return sentences_idx

def get_upper_size_limit(sentences, tollerance):
    if tollerance == None:
        tollerance = 0.99

    tweet_lens = np.asarray([len(sentence) for sentence in sentences])
    # print('Max tweet word length: ', tweet_lens.max())
    # print('Mean tweet word length: ', np.median(tweet_lens))
    # print('99% percent under: ', np.quantile(tweet_lens, tollerance))

    upper_limit = np.quantile(tweet_lens, tollerance)

    return upper_limit


def padding(sentences, seq_len):
    # A function to make all the sequence have the same lenght
    # Note that the output is a Numpy matrix
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, tweet in enumerate(sentences):
        len_tweet = len(tweet)
        if len_tweet != 0:
          if len_tweet <= seq_len:
            # If its shorter, we fill with zeros (the padding Token index)
            features[ii, -len(tweet):] = np.array(tweet)[:seq_len]
          if len_tweet > seq_len:
            # If its larger, we take the last 'seq_len' indexes
            features[ii, :] = np.array(tweet)[-seq_len:]
    return features

def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: For this notebook to perform best, "
          "if possible, in the menu under `Runtime` -> "
          "`Change runtime type.`  select `GPU` ")
    else:
        print("GPU is enabled in this notebook.")

    return device

def acc(pred,label):
    # function to predict accuracy
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()



class SentimentRNN(nn.Module):
  def __init__(self,no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.1, output_dim = 6):
    super(SentimentRNN,self).__init__()

    self.output_dim = output_dim
    self.hidden_dim = hidden_dim
    self.no_layers = no_layers
    self.vocab_size = vocab_size
    self.drop_prob = drop_prob

    # Embedding Layer: transform the
    self.embedding = nn.Embedding(vocab_size, embedding_dim)

    # LSTM Layers
    self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,
                        num_layers=no_layers, batch_first=True,
                        dropout=self.drop_prob)

    # Dropout layer
    self.dropout = nn.Dropout(drop_prob)

    # Linear and Sigmoid layer
    self.fc = nn.Linear(self.hidden_dim, output_dim)

    # self.sig = nn.softmax()

  def forward(self,x,hidden):
    batch_size = x.size(0)

    # Embedding out
    embeds = self.embedding(x)
    #Shape: [batch_size x max_length x embedding_dim

    # LSTM out
    lstm_out, hidden = self.lstm(embeds, hidden)
    # Shape: [batch_size x max_length x hidden_dim]

    # Select the activation of the last Hidden Layer
    lstm_out = lstm_out[:,-1,:].contiguous()
    # Shape: [batch_size x hidden_dim]

    ## You can instead average the activations across all the times
    # lstm_out = torch.mean(lstm_out, 1).contiguous()

    # Dropout and Fully connected layer
    out = self.dropout(lstm_out)
    out = self.fc(out)


    # Sigmoid function
    # sig_out = self.sig(out)

    # return last output and hidden state
    return out, hidden

  def init_hidden(self, batch_size):
    ''' Initializes hidden state '''
    # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
    # initialized to zero, for hidden state and cell state of LSTM
    h0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
    c0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
    hidden = (h0,c0)
    return hidden



@click.group()
def cli():
    pass

@click.command(name='run')
@click.argument('db_path', type=click.Path(exists=False))
@click.option('--output_loc', '-ol', type=click.Path(exists=False))
@click.option('--ent_op', '-e', default='wln')
@click.option('--seed', '-s', default=0)
@click.option('--device', '-d', default='cuda')
@click.option('--tollerance', '-t', default=0.99)
@click.option('--batch_size', '-b', default=100)
@click.option('--vocab', '-v', default=None)
@click.option('--embedding_dim', '-ed', default=32)
@click.option('--layers', '-t', default=2)
@click.option('--hidden_dim', '-hd', default=64)
@click.option('--output_dim', '-od', default=6)
@click.option('--drop_prob', '-dp', default=0.25)
@click.option('--lr', '-l', default=0.001)
@click.option('--epochs', '-ep', default=5)
@click.option('--clip', '-c', default=5)
def cli_run(db_path=None,
        output_loc = None,
        ent_op = 'wln',
        seed = 0,
        device = 'cuda',
        tollerance = 0.99,
        batch_size = 100,
        vocab = None,
        embedding_dim = 32,
        layers = 2,
        hidden_dim = 64,
        drop_prob = 0.25,
        lr = 0.001,
        epochs = 5,
        clip = 5,
        **kwargs):

    nltk.download('wordnet')
    nltk.download('omw-1.4')
    if db_path == None:
        db_path = '/Users/user/NMA_DEEP/merged_training.pkl'
    # load data
    X, y = load_data(db_path)

    if output_loc is None:
        output_loc = '/Users/user/NMA_DEEP/NMA_NOTES/project_NLP/RNN_sentiment_classification_output'
        if not op.exists(output_loc):
            os.mkdir(output_loc)

    pre_processed_file = 'pre_processed_data.npz'
    if not op.exists(op.join(output_loc, pre_processed_file)):
        # preprocessing of full text
        X_prepro = remove_stopwords(X)
        X_prepro = link_entities(X_prepro)
        if ent_op == 'wnl':
            X_prepro = list(map(stem_wnl, X_prepro))
        elif ent_op == 'porter':
            X_prepro = list(map(stem_porter, X_prepro))
        np.savez(op.join(output_loc, pre_processed_file),
                 original_data = X,
                 preprocessed_tweets = X_prepro,
                 emotions = y,
                 metadata = {'entity_option': ent_op})
    else:
        X = np.load(op.join(output_loc, pre_processed_file), encoding="latin1", allow_pickle=True)['preprocessed_data'].tolist()

    # Split the data into train and test
    x_train_text, x_test_text, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

    #resample only training
    x_train_text, y_train = resample(x_train_text, y_train, seed = seed)

    #padding and length standardization
    # tokenize
    x_train_token, x_test_token = tokenize(x_train_text, x_test_text)
    sorted_words = get_sorted_words(x_train_token)
    vocab = len(sorted_words) # change if you decide to reduce
    word_to_idx, idx_to_word = make_word_to_idx_dict(sorted_words, len(sorted_words))

    x_train_idx = tokens_to_idx(x_train_token,word_to_idx)
    x_test_idx = tokens_to_idx(x_test_token,word_to_idx)

    max_length = get_upper_size_limit(x_train_text, tollerance=tollerance)
    # We convert our list of tokens into a numpy matrix
    # where all instances have the same lenght
    x_train_pad = padding(x_train_idx, int(max_length))
    x_test_pad = padding(x_test_idx, int(max_length))

    # We convert our target list a numpy matrix
    y_train_np = np.asarray(y_train).astype('float32')
    y_test_np = np.asarray(y_test).astype('float32')

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train_np))
    valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test_np))

    # dataloaders
    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size,drop_last = True)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size,drop_last = False)

    # Let's define our model
    model = SentimentRNN(layers, vocab, hidden_dim,
                         embedding_dim, drop_prob=drop_prob)
    # Moving to gpu
    model.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total Number of parameters: ', params)

    # loss and optimization functions
    # Binary crossentropy is a good loss function for a binary classification problem
    criterion = nn.CrossEntropyLoss()
    # We choose an Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Initial Loss value (assumed big)
    valid_loss_min = np.Inf

    # Lists to follow the evolution of the loss and accuracy
    epoch_tr_loss, epoch_vl_loss = [], []
    epoch_tr_acc, epoch_vl_acc = [], []

    # Train for a number of Epochs
    all_probabilities = []
    for epoch in range(epochs):
        train_losses = []
        train_acc = 0.0
        model.train()

        for inputs, labels in train_loader:
            # Initialize hidden state
            h = model.init_hidden(batch_size)
            # Creating new variables for the hidden state
            h = tuple([each.data.to(device) for each in h])

            # Move batch inputs and labels to gpu
            inputs, labels = inputs.to(device), labels.to(device)

            # Set gradient to zero
            model.zero_grad()

            # Compute model output
            output, h = model(inputs, h)

            # Calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            probabilities = nn.functional.softmax(output, dim = 1)
            all_probabilities.append(probabilities.squeeze())
            loss.backward()
            train_losses.append(loss.item())

            # calculating accuracy
            accuracy = acc(output, labels)
            train_acc += accuracy

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        # Evaluate on the validation set for this epoch
        val_losses = []
        val_acc = 0.0
        model.eval()
        for inputs, labels in valid_loader:
            # Initialize hidden state
            val_h = model.init_hidden(batch_size)
            val_h = tuple([each.data.to(device) for each in val_h])

            # Move batch inputs and labels to gpu
            inputs, labels = inputs.to(device), labels.to(device)

            # Compute model output
            output, val_h = model(inputs, val_h)

            # Compute Loss
            val_loss = criterion(output.squeeze(), labels.float())

            val_losses.append(val_loss.item())

            accuracy = acc(output, labels)
            val_acc += accuracy

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_train_acc = train_acc / len(train_loader.dataset)
        epoch_val_acc = val_acc / len(valid_loader.dataset)
        epoch_tr_loss.append(epoch_train_loss)
        epoch_vl_loss.append(epoch_val_loss)
        epoch_tr_acc.append(epoch_train_acc)
        epoch_vl_acc.append(epoch_val_acc)
        print(f'Epoch {epoch+1}')
        print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
        print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
        if epoch_val_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                            epoch_val_loss))
            # torch.save(model.state_dict(), '../working/state_dict.pt')
            valid_loss_min = epoch_val_loss
        print(25 * '==')

    fig = plt.figure(figsize=(20, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_tr_acc, label='Train Acc')
    plt.plot(epoch_vl_acc, label='Validation Acc')
    plt.title("Accuracy")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_tr_loss, label='Train loss')
    plt.plot(epoch_vl_loss, label='Validation loss')
    plt.title("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    file_name = 'model_loss_accuracy_output'
    fig.savefig(op.join(output_loc, file_name),
                format='pdf',
                dpi=300)

    np.save(op.join(output_loc, 'model_predictions'), all_probabilities)


cli.add_command(cli_run)



if __name__ == '__main__':
    cli()