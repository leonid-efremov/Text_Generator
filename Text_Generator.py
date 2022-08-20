# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 16:45:42 2021

@author: Leonid_PC

Основной класс, содержащий все модели, методы тренировки и подготовки данных
"""

import pandas as pd
from models.Lang_gen import trigramm_model, prepare_for_lang
from models.LSTM_gen import TextGenerator_LSTM
from models.GPT_gen import TextGenerator_GPT

a = ord('а')  # add all letters which needed - Russian alphabet (abc)
abc = ''.join([chr(i) for i in range(a, a + 32)] + [' '] + [chr(a + 33)])
DATA_PATH = 'data/'


class TextGenerator:

    def __init__(self, model_type='LSTM', data_path=DATA_PATH):
        """Class for generating text"""
        self.data_path = data_path

        self.model_type = model_type

        self.corpus = None
        self.init_word = None
        self.max_len = 50

        self.abc = abc

    def prepare(self, file_path, only_words=False,
                layers_sizes={'hidden_size': 128, 'embedding_size': 128}, pretrained=True):
        """
        data - list of names or None
        Preparing model:
        Load text to train model from list 'data', 
        which contain names of texts.
        If data is None load saved file
        """

        self.pretrained = pretrained

        if self.model_type in ['Lang', 'LSTM']:
            # loading data
            self.train_text = pd.read_csv(self.data_path + file_path)
            self.train_text = ' '.join(list(self.train_text['text'].astype(str)))

            self.train_text = self.train_text.replace(' nan', '')  # for text in json

            if only_words:
                self.corpus = self.train_text.lower()
                self.corpus = ''.join([i for i in self.corpus if i in self.abc])
            else:
                self.corpus = self.train_text

        # setup model
        if self.model_type == 'Lang':
            assert set(self.corpus).issubset(self.abc), 'Choose correct text preparation method'
            self.corpus, self.words2, self.words3 = prepare_for_lang(self.corpus)

        elif self.model_type == 'LSTM':
            assert set(self.corpus).issubset(self.abc), 'Choose correct text preparation method'
            # net parameters
            self.LSTM_model = TextGenerator_LSTM(layers_sizes, self.corpus,
                                                 self.abc, n_layers=2,
                                                 pretrained=self.pretrained)

        elif self.model_type == 'GPT':
            assert self.pretrained, 'Run training in Colab separately!'

            model_type = file_path.split('.')[0]
            assert model_type in ['books', 'dialogs'], 'Select one of pretrained model!'

            self.GPT_model = TextGenerator_GPT(model_type=model_type)

        else:
            assert self.model_type in ['Lang', 'LSTM', 'GPT'], 'Select correct model type!'

    def generate(self, train_parameters={'batch_size': 64, 'seq_len': 256,
                                         'n_epochs': 4000, 'lr': 1e-3}):
        """Generate phrase from initial word"""

        start_text = self.init_word if self.init_word else ' '
        res = None

        if self.model_type == 'Lang':
            phrase = trigramm_model(self.init_word, self.words2, self.words3,
                                    num_words=self.max_len)
            res = ' '.join(phrase)

        elif self.model_type == 'LSTM':
            if not self.pretrained:
                batch_size = train_parameters['batch_size']
                seq_len = train_parameters['seq_len']
                n_epochs = train_parameters['n_epochs']
                lr = train_parameters['lr']

                self.LSTM_model.train(batch_size, seq_len, n_epochs=n_epochs,
                                      lr=lr, save_model=True)

            res = self.LSTM_model.evaluate(temp=0.3, prediction_len=500,
                                           start_text=start_text)

        elif self.model_type == 'GPT':
            res = self.GPT_model.gen_from_pretrained(context=start_text, max_length=self.max_len)

        else:
            assert self.model_type in ['Lang', 'LSTM', 'GPT'], 'Select correct model type!'

        print(res)
        return res


if __name__ == '__main__':
    model = TextGenerator(model_type='GPT')

    train_texts = 'books.csv'
    model.prepare(train_texts, only_words=True, pretrained=True)

    phrase = model.generate()
