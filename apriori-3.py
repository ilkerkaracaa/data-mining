import pandas as pd
import numpy as np
import openpyxl
from pytictoc import TicToc

file_name = "excel100.xlsx"
wb = openpyxl.load_workbook(filename=file_name, read_only=True)
sheet = wb[wb.sheetnames[0]]
database = np.array(pd.DataFrame(sheet.values))
wb.close()

numOfTransactions = database.shape[0]


allItems = []
for i in range(0, numOfTransactions):
    transaction = database[i, :]
    for item in transaction:
        if item != None:
            allItems.append(item)

allItems = np.array(allItems)
uniqueItems = np.unique(allItems)
uniqueItems = [str(item) for item in uniqueItems]
numOfUniqueItems = len(uniqueItems)

dataMatrix = np.zeros((numOfTransactions, numOfUniqueItems), dtype="int8")
for i in range(0, numOfTransactions):
    transaction = database[i, :]
    for item in transaction:
        if item != None:
            I = uniqueItems.index(item)
            dataMatrix[i, I] = 1

del database

counts = np.sum(dataMatrix, axis=0) / numOfTransactions

minSupp = 100

# Timer
runTime = TicToc()
runTime.tic()

I = np.nonzero(minSupp <= counts)[0]
uniqueItems = [uniqueItems[i] for i in I]
dataMatrix = dataMatrix[:, I]
counts = counts[I]
numOfUniqueItems = len(uniqueItems)


def findSupport(itemSet, dataMatrix):
    supportOfItemSet = sum(np.prod(dataMatrix[:, itemSet], axis=1)) / numOfTransactions
    return supportOfItemSet


def candidateGeneration(fk, numOfUniqueItems):
    ck = []
    for i in range(0, len(fk)):
        itemSet1 = list(fk[i])
        lastElement = itemSet1[-1]
        for item2 in range(lastElement + 1, numOfUniqueItems):
            xxxx = list(itemSet1)
            xxxx.append(item2)
            ck.append(xxxx)
    return ck


F = []
S = []
k = 1
fk = []
for item, count in zip(uniqueItems, counts):
    print(item, count)
    if minSupp <= count:
        fk.append([uniqueItems.index(item)])
        F.append([item])
        S.append(count)

k = 2

while 0 < len(fk):
    Ck = candidateGeneration(fk, numOfUniqueItems)
    fk = []
    for itemSet in Ck:
        support = findSupport(itemSet, dataMatrix)
        if minSupp <= support:
            fk.append(itemSet)
            F.append([uniqueItems[item] for item in itemSet])
            S.append(support)

print("----------------------------------------")
i = 0
for FI, support in zip(F, S):
    i += 1
    print("#", i, " ", FI, "  supp.:", support)

runTime.toc()
print("Run time:", runTime.elapsed)

print("------Kurallar Oluşturuluyor---------")
#function to convert
def listToString(s):
    str1 = ""
    return (str1.join(s))

minconf = 0.80
import itertools
for frequentItemSet,supportXY in zip(F,S):
    if len(frequentItemSet) > 1:
        for L in range(1,len(frequentItemSet)):
            for antecedent in itertools.combinations(frequentItemSet,L):
                antecedent = list(antecedent)
                consequent = list(set(frequentItemSet))
                consequent.sort()
                supportX = S[F.index(antecedent)]
                supportY = S[F.index(consequent)]
                confXY = supportXY/supportX
                inteXY = abs(supportXY-supportX*supportY)
                if minconf <= confXY:
                    if 0 < inteXY:
                        print(listToString(antecedent),'--->',listToString(consequent))
