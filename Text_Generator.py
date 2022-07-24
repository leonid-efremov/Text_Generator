# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 16:45:42 2021

@author: Leonid_PC

Основной класс, содержащий все модели, методы тренировки и подготовки данных
"""

import pandas as pd
from prepare_data import abc
from models.Lang_gen import trigramm_model, prepare_for_lang
from models.LSTM_gen import TextGenerator_LSTM


DATA_PATH = 'data/'


class TextGenerator:

    def __init__(self, model_type='LSTM', data_path=DATA_PATH):
        """
        Class for generating text
        """
        self.data_path = data_path
        
        self.model_type = model_type

        self.corpus = None        
        self.init_word = None
        self.num_words = 15

        self.abc = abc
    
    
    def prepare(self, file_path, layers_sizes={'hidden_size': 128, 
                'embedding_size': 128}, pretrained=True):
        """
        data - list of names or None
        Preparing model:
        Load text to train model from list 'data', 
        which contain names of texts.
        If data is None load saved file
        """
        
        self.pretrained = pretrained
        
        # loading data
        self.corpus = pd.read_csv(self.data_path + file_path)
        self.corpus = ' '.join(list(self.corpus['text']))
        
        # setup model
        if self.model_type == 'Lang':
            self.corpus, self.words2, self.words3 = prepare_for_lang(self.corpus)
        
        elif self.model_type == 'LSTM':
            # net parameters
            self.LSTM_model = TextGenerator_LSTM(layers_sizes, self.corpus, 
                                                 self.abc, n_layers=2, 
                                                 pretrained=self.pretrained)
                                                
        else:
            assert self.model_type in ['Lang', 'LSTM'], 'Select correct model type!'
            
            
    def generate(self, train_parameters={'batch_size': 64,'seq_len': 256,
                                         'n_epochs': 4000,'lr': 1e-3}):
        "Generate phrase from initial word"
        
        start_text = self.init_word if self.init_word else ' '
        
        if self.model_type == 'Lang':
            phrase = trigramm_model(self.init_word, self.words2, self.words3, 
                                    num_words=self.num_words)
            res = ' '.join(phrase)  # save and print resulting phrase
           
        elif self.model_type == 'LSTM':  
            
            if not self.pretrained:
                batch_size = train_parameters['batch_size']
                seq_len = train_parameters['seq_len']
                n_epochs =  train_parameters['n_epochs']
                lr = train_parameters['lr']
            
                self.LSTM_model.train(batch_size, seq_len, n_epochs=n_epochs, 
                                      lr=lr, save_model=True)
            
            res = self.LSTM_model.evaluate(temp=0.3, prediction_len=500, 
                                           start_text=start_text)
        print(res)
        return res
            
    

if __name__ == '__main__':  
    
    model = TextGenerator(model_type='LSTM')  # initate model    
    train_texts = 'books.csv'  # 'dialog.csv'
    
    layers_sizes =  {'hidden_size': 128, 
                     'embedding_size': 128}
    model.prepare(file_path=train_texts, layers_sizes=layers_sizes, pretrained=True)
    
    train_parameters = {'batch_size': 64,
                        'seq_len': 256,
                        'n_epochs': 5000,
                        'lr': 1e-3}
    p = model.generate(train_parameters=train_parameters)   
    
    
    
    
