# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 16:45:42 2021

@author: Leonid_PC

Генератор теста
main файл:
    - простая грамматическая модель
    - интерфейс для вызова LSTM модели
    - основная программа
"""

import os 
import json
import pickle
from Lang_gen import trigramm_model, lang_preprocess



class TextGenerator:

    def __init__(self, model_type='LSTM'):
        """
        Class for generating text
        """
        
        self.model_type = model_type

        self.train_text = None
        self.words2 = dict()  # collecting all words and next words
        self.words3 = dict()
        
        self.init_word = None
        self.num_words = 15
        
        a = ord('а')  # add all letters which needed - russian alphabet (abc)
        self.abc = ''.join([chr(i) for i in range(a, a + 32)] 
                                + [' '] + ['\n'] + [chr(a + 33)])
        

    def fit_for_lang(self, source='WarAndPeace1.txt'):
        "Train model: for words in 'source' find next words"

        file_name, file_extension = os.path.splitext(source)
        if file_extension == '.txt':
            with open(source, 'r') as f:  # read text from file
                self.train_text = f.read()        
        elif file_extension == '.json':
            with open(source, 'r', encoding="utf-8") as f:
                t = json.load(f)  # loading ghat from telegram
                self.train_text = str()
                for i in t['messages']:
                    self.train_text += str(i['text']) + ' '  

        #with open('DontRename&PlaceInOneFolder', 'wb') as w:
        with open('words', 'wb') as w:  # save data to file
            pickle.dump(self.train_text, w)
        
            
    def generate(self):
        "Generation of phrase for initial word"
        
        if self.model_type == 'Lang':
            phrase = trigramm_model(self.init_word, self.words2, self.words3, 
                                    num_words=self.num_words)        
        res = ' '.join(phrase)  # save and print resulting phrase
        print(res)
        return res
    
    
    def prepare_model(self, data=None):
        """
        data - list of names or None
        Preparing model:
        Load text to train model from list 'data', 
        which contain names of texts.
        If data is None load saved file
        """
        
        if data is not None:  
            # train model if necessary data specified
            if self.model_type == 'Lang':
                for i in data:
                    self.fit_for_lang(source=i)
        else:
            # load data
            if self.model_type == 'Lang':
                # DontRename&PlaceInOneFolder
                with open('DontRename&PlaceInOneFolder', 'rb') as w:
                    self.train_text = pickle.load(w)
        
        self.corpus, self.words2, self.words3 = lang_preprocess(self.train_text,
                                                                self.abc) 
    
    
                   
if __name__ == '__main__':  
    
    gen = TextGenerator(model_type='Lang')  # initate model
    
    train_texts = ['WarAndPeace1.txt', 'HarryPotter.txt']  # ['result.json']
    gen.prepare_model(data=train_texts)  # data=train_texts   
    #start_word = str(input('Enter your word:'))  # generate phrase from word
    #gen.init_word = 'утро доброе'
    p = gen.generate()    
    
    
    
    
