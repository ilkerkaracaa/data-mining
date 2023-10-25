import pandas as pd
import numpy as np
import openpyxl

file_name = 'FIM_Demo3.xlsx'
wb = openpyxl.load_workbook(file_name)
sheet = wb[wb.sheetnames[0]]

database = np.array(pd.DataFrame(sheet.values))
wb.close()

allItems = []
for i in range(0, database.shape[0]):
    allItems.append(database[i][0])
    transaction = database[i][1:]
    for item in transaction:
        if item != None:
            allItems.append(item)

allItems = np.array(allItems)
uniqeItems = np.unique(allItems)
uniqeItems = [str(item) for item in uniqeItems]

numOfUniqeItems = len(uniqeItems)
numOfTransactions = database.shape[0]

dataMatrix = np.zeros((numOfTransactions, numOfUniqeItems))
for i in range (0,database.shape[0]):
    transaction = database[i,:]
    for item in transaction:
        if item != None:
            I = uniqeItems.index((item))
            dataMatrix[i][I] = 1
            
counts = np.sum(dataMatrix, axis=0)
minSup = 4
I = np.nonzero(counts >= minSup)[0]
uniqeItems = [uniqeItems[i] for i in I]
dataMatrix = dataMatrix[:,I]
counts = counts[I]

