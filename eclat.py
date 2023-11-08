import pandas as pd
import numpy as np
import openpyxl
from pytictoc import TicToc

file_name = "excel1000.xlsx"
wb = openpyxl.load_workbook(filename=file_name, read_only=True)
sheet = wb[wb.sheetnames[0]]
database = np.array(pd.DataFrame(sheet.values))
wb.close()

numOfTransactions = database.shape[0]

allItems = []
for i in range(0, database.shape[0]):
    transaction = database[i, :]
    for item in transaction:
        if item != None:
            allItems.append(item)

allItems = np.array(allItems)
uniqueItems = np.unique(allItems)
uniqueItems = [str(item) for item in uniqueItems]
numOfUniqueItems = len(uniqueItems)

dataMatrix = np.zeros((numOfTransactions, numOfUniqueItems), dtype="int8")
for i in range(0, database.shape[0]):
    transaction = database[i, :]
    for item in transaction:
        if item != None:
            I = uniqueItems.index(item)
            dataMatrix[i, I] = 1

del database

counts = np.sum(dataMatrix, axis=0) / numOfTransactions

minSupp = 0.20
runTime = TicToc()
runTime.tic()

I = np.nonzero(minSupp <= counts)[0]
uniqueItems = [uniqueItems[i] for i in I]
dataMatrix = dataMatrix[:, I]
counts = counts[I]
numOfUniqueItems = len(uniqueItems)

R = uniqueItems
tidList = []
for item in R:
    I = np.nonzero(dataMatrix[:, R.index(item)] == 1)[0]
    tidList.append(list(I))
del dataMatrix

def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item


def dfsLoop(item, R, tid, tidList):
    E = R[R.index(item[-1]) + 1:]
    for suffix in E:
        suffixTid = tidList[R.index(suffix)]
        #ortaTid = list(set(tid).intersection(suffixTid)) Calismiyor
        ortaTid = list(uniq(sorted(tid, reverse=True)))
        if minSupp <= len(ortaTid) / numOfTransactions:
            print(item + suffix, len(ortaTid) / numOfTransactions)
            dfsLoop(item + suffix, E, ortaTid, tidList)


for item, tid in zip(R, tidList):
    print(item, len(tid) / numOfTransactions)
    dfsLoop(item, R, tidList, tid)
