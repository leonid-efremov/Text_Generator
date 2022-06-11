"""
Конвертация исходных данных в pd.DataFrame для удобного чтения
"""

import json
import pandas as pd


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
    res['message'] = messages

    if save:  # save data to file
        res.to_csv(out_path, index=False)

    return res



if __name__ == '__main__':

    DATA_PATH = 'data/'

    txt = prepare_txt([DATA_PATH + 'HarryPotter.txt', 
                       DATA_PATH + 'WarAndPeace1.txt'], 
                       out_path= DATA_PATH + 'books.csv', save=True)
    json = prepare_json(DATA_PATH + 'result.json', 
                        DATA_PATH + 'dialog.csv', save=True)