"""
Конвертация исходных данных в pd.DataFrame для удобного чтения
"""


import os
import json
import pandas as pd


a = ord('а')  # add all letters which needed - Russian alphabet (abc)
abc_full = ''.join([chr(i) for i in range(a, a + 32)] + [chr(a + 33)] \
                 + [chr(i).upper() for i in range(a, a + 32)] + [chr(a + 33).upper()] \
                 + [' '] + ['!', '?', '.', ',', ')', '(', ':', ';', '-'])


def prepare_txt(file_paths, out_path, save=False):
    """
    Конвертация из .txt формата в .csv
    Список файлов задается листом
    """
    texts = dict.fromkeys([i.split('.')[0] for i in file_paths])

    for i in file_paths:
        path = i
        with open(path, 'r') as f:  # read text from file
            texts[i.split('.')[0]] = f.read()
    
    res = pd.DataFrame()
    res['title'] = texts.keys()
    res['text'] = texts.values()

    if save:  # save data to file
        res.to_csv(out_path, index=False)

    return res

def prepare_json(file_path, out_path, save=False):
    """
    Конвертация из .json формата .csv
    Для единичного файла
    """
    
    dates = []
    messages = []

    with open(file_path, 'r', encoding="utf-8") as f:
        text = json.load(f)  # loading chat from telegram
        for i in text['messages']:
            messages.append(i['text'])
            dates.append(i['date'])
    
    res = pd.DataFrame()
    res['date'] = dates
    res['text'] = messages


    if save:  # save data to file
        res.to_csv(out_path, index=False)

    return res

def prepare_text(text):     
    
    # remove all sybols which are not letters 
    res = str(text).replace('\n', ' ')
    #res = res.lower() # convert to lowercase
    res = ''.join([i for i in res if i in abc_full])
    res = ' '.join(res.split())

    return res

def preprocess_data(file_path, out_path, save=False):
    """
    Обработка и подготовка текста
    file_path - list or str
    out_path - str
    """

    # check input argument type
    if isinstance(file_path, str):
        file_type = os.path.splitext(file_path)[-1]
    elif isinstance(file_path, list):
        file_type = os.path.splitext(file_path[0])[-1]
    else:
        assert isinstance(file_path, (str, list)), 'Unknown file path type!'

    # check input file type
    if file_type == '.txt':
        res = prepare_txt(file_path, out_path, save=False)
    elif file_type == '.json':    
        res = prepare_json(file_path, out_path, save=False)
    else:
        assert isinstance(file_type, ('.txt', '.json')), 'Unknown file type!'

    corpus = res
    corpus['text'] = corpus['text'].apply(prepare_text)

    if save:  # save data to file
        corpus.to_csv(out_path, index=False)

    return corpus



if __name__ == '__main__':

    DATA_PATH = 'data/'

    txt = preprocess_data([DATA_PATH + 'HarryPotter.txt', 
                           DATA_PATH + 'WarAndPeace1.txt'], 
                           out_path= DATA_PATH + 'books.csv', save=True)
    json = preprocess_data(DATA_PATH + 'result.json', 
                           DATA_PATH + 'dialog.csv', save=True)