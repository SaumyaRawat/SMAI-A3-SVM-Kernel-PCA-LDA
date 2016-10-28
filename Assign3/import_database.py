import numpy as np
import pandas as pd

def import_database(db_name='arcene'):
    if db_name == 'arcene' or db_name == 'Arcene':
        y=np.loadtxt('dataset/arcene_valid.labels')
        train_data = pd.read_csv(
            filepath_or_buffer='dataset/arcene_valid.data', 
            header=None,
            sep=" ",
            engine = 'python')
        train_data=train_data.fillna(0)
        X = (train_data.values)
        return X,y

    elif db_name == 'dexter' or db_name == 'Dexter':
        y=np.loadtxt('dataset/dexter_valid.labels')
        lines = open('dataset/dexter_valid.data', 'r').readlines()
        dataList = [line.rstrip('\n') for line in lines]
        n = len(dataList)
        X = np.zeros((n,20000))
        i = 0
        for line in (dataList):
            line_ele = line.split()
            for element in line_ele:
                col = element.split(':')[0]
                attr_value = element.split(':')[1]
                X[i][int(col)] = attr_value
            i = i+1
        return X,y