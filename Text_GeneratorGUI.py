# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 16:45:42 2021

@author: Leonid_PC

Генератор теста:
https://gist.github.com/sslotin/d4c80a5f724a5cede5f2dfa62958074b
"""
import os 
import sys 
import json
import pickle
from PyQt5 import QtCore, QtGui, QtWidgets
from random import choice



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(450, 647)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(150, 250, 150, 70))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("QPushButton:pressed{\n"
"    background-color:orange;\n"
"}")
        self.pushButton.setObjectName("pushButton")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(105, 70, 240, 70))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.textEdit.setFont(font)
        self.textEdit.setObjectName("textEdit")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(135, 30, 180, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAutoFillBackground(False)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(150, 340, 150, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(50, 400, 350, 200))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.textBrowser.setFont(font)
        self.textBrowser.setObjectName("textBrowser")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(95, 185, 170, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(270, 180, 80, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.textEdit_2.setFont(font)
        self.textEdit_2.setObjectName("textEdit_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 450, 19))
        self.menubar.setObjectName("menubar")
        self.menutrain_data = QtWidgets.QMenu(self.menubar)
        self.menutrain_data.setObjectName("menutrain_data")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menutrain_data.addSeparator()
        self.menubar.addAction(self.menutrain_data.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Genarate"))
        self.textEdit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8.25pt;\"><br /></p></body></html>"))
        self.label.setText(_translate("MainWindow", "Enter your word here:"))
        self.label_2.setText(_translate("MainWindow", "Generated phrase:"))
        self.label_3.setText(_translate("MainWindow", "Number of words:"))
        self.textEdit_2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8.25pt;\"><br /></p></body></html>"))
        self.menutrain_data.setTitle(_translate("MainWindow", "Data"))



class TextGenerator:

    def __init__(self):
        """
        Class for generating text
        """

        self.train_text = None
        self.words2 = dict()  # collecting all words and next words
        self.words3 = dict()
        
        self.init_word = None
        self.num_words = 15
        
        a = ord('а')  # add all letters which needed a.e. all russian alphabet
        self.alphabet = ''.join([chr(i) for i in range(a, a + 32)] 
                                + [' '] + ['\n'] + [chr(a + 33)])
        

    def fit(self, source='WarAndPeace1.txt'):
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

        train_text = self.train_text.lower()  # convert to lowercase

        # remove all sybols which are not letters
        train_text = ''.join([i for i in train_text if i in self.alphabet])

        train_text = train_text.split()  # split text in single words

        # find unique words and next words for each in bigramm model
        for prev, curr in zip(train_text, train_text[1:]):
            if prev not in self.words2.keys():
                self.words2[prev] = []
            self.words2[prev].append(curr)
        # trigramm model    
        for prev, curr, n in zip(train_text, train_text[1:], train_text[2:]):
            if (prev, curr) not in self.words3.keys():
                self.words3[(prev, curr)] = []
            self.words3[(prev, curr)].append(n)
            
        # remove empty elements
        self.words2 = dict([(k, v) for k, v in self.words2.items() if len(v) > 0])
        self.words3 = dict([(k, v) for k, v in self.words3.items() if len(v) > 0])
        
        # DontRename&PlaceInOneFolder
        save_data = [self.words2, self.words3]
        with open('words', 'wb') as w:  # save data to file
            pickle.dump(len(save_data), w)
            for d in save_data:
                pickle.dump(d, w)
        
   
        
    def bigramm_model(self):
        "Generate phrase in bigramm model"
        
        phrase = list()
        words = [item for sublist in list(self.words2.values()) for item in sublist]
        
        # set initial word or pick random
        if self.init_word is not None:
            self.init_word = self.init_word.split()
            for i in self.init_word:
                phrase.append(i) 
        else:
            phrase.append(choice(words))         
        li = len(phrase)
        
        for _ in range(self.num_words - li):
            # if word appear in train data pick random next word
            if phrase[-1] in list(self.words2.keys()):
                candidates = self.words2[phrase[-1]]
                phrase.append(choice(candidates))
            # else pick just random word
            else:
                phrase.append(choice(words))
        
        return phrase
    


    def trigramm_model(self):
        "Generate phrase in trigramm model"
        
        phrase = list()
        words = [item for sublist in list(self.words3.values()) for item in sublist]
        
        # set two initial words or pick random
        if self.init_word is not None:                    
            self.init_word = self.init_word.split()
            for i in self.init_word:
                phrase.append(i)
        else:
            phrase.append(choice(words))  
        # if word appear in train data pick random next word
        if len(phrase) < 2:
            if phrase[-1] in list(self.words2.keys()):
                candidates = self.words2[phrase[-1]]
                phrase.append(choice(candidates))
            # else pick just random word
            else:
                phrase.append(choice(words))
        li = len(phrase)
            
        for i in range(self.num_words - li):
            # if word appear in train data pick random next word
            if (phrase[-2], phrase[-1]) in list(self.words3.keys()):
                candidates = self.words3[(phrase[-2], phrase[-1])]
                phrase.append(choice(candidates))
            # elif pick word from bigramm model
            elif phrase[-1] in list(self.words2.keys()):  
                candidates = self.words2[phrase[-1]]
                phrase.append(choice(candidates))
            # else pick just random word
            else:
                phrase.append(choice(words))
        
        return phrase
    
        
        
    def generate(self):
        "Generation of phrase for initial word"
        
        phrase = self.trigramm_model()        
        res = ' '.join(phrase)  # save and print resulting phrase
        #print(res)
        return res
    
    
    def prepare_model(self, data=None):
        """
        data - list of names or None
        Preparing model:
        Load text to train model from list 'data', 
        which contain names of text names
        If data is None load saved file
        """
        
        self.words2 = dict()
        self.words3 = dict()
        
        if data is not None:  # train model if necessary data specified
            for i in data:
                self.fit(source=i)
        else:
            # DontRename&PlaceInOneFolder
            save_data = []
            with open('DontRename&PlaceInOneFolder', 'rb') as w:  # load data
                for _ in range(pickle.load(w)):
                    save_data.append(pickle.load(w))
            self.words2 = save_data[0]
            self.words3 = save_data[1]


class ExampleApp(QtWidgets.QMainWindow, Ui_MainWindow, TextGenerator):
    
    def __init__(self):
        """
        Class for operating with GUI
        Contain GUI elements and model functions
        """
        
        super().__init__()  # initiate design of app
        self.setupUi(self)        
        TextGenerator.__init__(self)  # initiate model
        # add actions to buttons
        self.menutrain_data.addAction('Browse your train data', self.browse_file)        
        self.pushButton.clicked.connect(self.generate_phrase)
        
        
    def browse_file(self):
        "Choose txt files to train model"
        
        self.words = dict()  # clear existing data
        filenames = QtWidgets.QFileDialog.getOpenFileNames(self, "Выберите txt-файл")
        self.prepare_model(filenames[0])        
        return filenames
    
    
    def generate_phrase(self):
        "Generate phrase"
        
        word = self.textEdit.toPlainText()  # pick initial word if specified
        if len(word) > 0:
            self.init_word = word
        
        num = self.textEdit_2.toPlainText()  # get number of word
        if len(num) > 0:
            self.num_words = int(num)
        
        p = self.generate()  # generate phrase and display  
        self.textBrowser.append(p)
        return p
                
    

if __name__ == '__main__':    
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = ExampleApp()  # Создаём объект класса ExampleApp
    window.prepare_model()
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение
    
    
    
    
