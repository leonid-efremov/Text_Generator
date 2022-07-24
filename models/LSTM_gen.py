# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:19:42 2022

@author: Leonid_PC
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F


MODELS_PATH = 'models/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def encode(text, chars):
    """
    Encode text from string to integer
    """
    
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    
    encoded_text = np.array([char2int[ch] for ch in text])
    
    return encoded_text, chars, int2char, char2int

def get_batches(arr, batch_size, seq_length):
    """
    Create a generator that returns batches of size
    batch_size x seq_length from arr.
       
    Arguments
    ---------
    arr: Array you want to make batches from
    batch_size: Batch size, the number of sequences per batch
    seq_length: Number of encoded chars in a sequence
    """
    
    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr)//batch_size_total
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))
    
    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y
        
def get_batch(sequence, batch_size, seq_len):
    trains = []
    targets = []
    for _ in range(batch_size):
        batch_start = np.random.randint(0, len(sequence) - seq_len)
        chunk = sequence[batch_start: batch_start + seq_len]
        train = torch.LongTensor(chunk[:-1]).view(-1, 1)
        target = torch.LongTensor(chunk[1:]).view(-1, 1)
        trains.append(train)
        targets.append(target)
    return torch.stack(trains, dim=0), torch.stack(targets, dim=0)


class TextRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super(TextRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.input_size)
        
    def forward(self, x, hidden):
        x = self.encoder(x).squeeze(2)
        out, (ht1, ct1) = self.lstm(x, hidden)
        out = self.dropout(out)
        x = self.fc(out)
        return x, (ht1, ct1)
    
    def init_hidden(self, batch_size=1):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size, 
                              requires_grad=True).to(device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_size, 
                              requires_grad=True).to(device))
        return hidden
    
class TextGenerator_LSTM:
    
    def __init__(self, layers_sizes, train_text, abc, n_layers, pretrained):
        
        self.abc = abc
        self.text = train_text
        self.encoded_text, _, self.idx_to_char, self.char_to_idx = \
            encode(self.text, self.abc)
        
        if pretrained:
            self.model = torch.load(MODELS_PATH + 'LSTMmodel.torch')
        else:
            self.model = TextRNN(input_size=len(self.idx_to_char), 
                                 hidden_size=layers_sizes['hidden_size'], 
                                 embedding_size=layers_sizes['embedding_size'], 
                                 n_layers=n_layers)
        self.model.to(device)
        
    
    def train(self, batch_size, seq_len, n_epochs=10000, lr=1e-3, 
              save_model=True):
        """ 
        Training a network 
    
            Arguments
            ---------        
            batch_size: Number of mini-sequences per mini-batch, aka batch size
            seq_length: Number of character steps per mini-batch
            n_epochs: Number of epochs to train
            lr: learning rate
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               patience=5, 
                                                               verbose=True, 
                                                               factor=0.5)
        self.loss_list = []

        for epoch in range(n_epochs):
            self.model.train()
            train, target = get_batch(self.encoded_text, batch_size, seq_len)
            train = train.permute(1, 0, 2).to(device)
            target = target.permute(1, 0, 2).to(device)
            hidden = self.model.init_hidden(batch_size)

            output, hidden = self.model(train, hidden)
            loss = criterion(output.permute(1, 2, 0), target.squeeze(-1).permute(1, 0))
        
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            self.loss_list.append(loss.item())
            
            if epoch % 50 == 0:
                print('Iteration number %i/%i, Loss: %.4f' \
                      %(epoch, n_epochs, self.loss_list[-1]))
                scheduler.step(self.loss_list[-1])
                predicted_text = self.evaluate()
                print(predicted_text)
                
        if save_model:
            torch.save(self.model, MODELS_PATH + 'LSTMmodel.torch')
                
                
    def evaluate(self, start_text=' ', prediction_len=200, temp=0.3):
        """
        Given a character, predict the next character.
        Returns the predicted character and the hidden state.
        """
        self.model.eval()
        
        predicted_text = start_text        
        idx_input = [self.char_to_idx[char] for char in start_text]
        
        hidden = self.model.init_hidden()        
        train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)        
        _, hidden = self.model(train, hidden)
            
        inp = train[-1].view(-1, 1, 1)
        
        for i in range(prediction_len):
            output, hidden = self.model(inp.to(device), hidden)
            output_logits = output.cpu().data.view(-1)
            
            p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()        
            top_index = np.random.choice(len(self.char_to_idx), p=p_next)
            inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
            
            predicted_char = self.idx_to_char[top_index]
            predicted_text += predicted_char
        
        return predicted_text        
    
    
    def check_model(self):
        
        plt.plot(self.loss_list)




