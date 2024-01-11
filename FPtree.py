import pandas as pd
import numpy as np
import openpyxl
from pytictoc import TicToc
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ------------------------------------------------------------------------------
plt.close('all')


# ------------------------------------------------------------------------------
def line(point1, point2):
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    plt.plot(x_values, y_values)
    return


# ------------------------------------------------------------------------------
def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


# ------------------------------------------------------------------------------
def FindEqualTo(itm, X):
    I = [i for i in range(0, len(X)) if X[i] == itm]
    return I


# ------------------------------------------------------------------------------
def FindTheLeavesInTheTree(parent, Leaves, dpth, DEPTH, NODES):
    IndexOfChilds = np.array(FindEqualTo(parent, NODES))
    if len(IndexOfChilds) == 0:
        Leaves.append(parent)
    else:
        dpth = dpth + 1;
        DEPTH[IndexOfChilds] = dpth;
        for i in range(0, IndexOfChilds.size):
            parent = IndexOfChilds[i]
            [Leaves, DEPTH] = FindTheLeavesInTheTree(parent, Leaves, dpth, DEPTH, NODES);
    return Leaves, DEPTH


# ------------------------------------------------------------------------------
def treelayout(NODES):
    parent = 0;
    Leaves = [];
    dpth = 1;
    DEPTH = np.ones((len(NODES)), dtype=int)
    [Leaves, DEPTH] = FindTheLeavesInTheTree(parent, Leaves, dpth, DEPTH, NODES)
    MaxDepth = max(DEPTH)

    dx = 1 / (len(Leaves) + 1)
    dy = 1 / (MaxDepth + 1)

    # ...vertical coordinates
    y = 1 - DEPTH * dy;

    # ...horizontal coordinates of the leaves
    x = np.zeros(len(y), dtype='float64')
    for i in range(0, len(Leaves)):
        itm = Leaves[i];
        x[itm] = (i + 1) * dx;

    # ...horizontal coordinates of remaining nodes
    for dpth in range(MaxDepth - 1, 0, -1):
        items = np.array(FindEqualTo(dpth, DEPTH))
        for i in range(0, len(items)):
            parent = items[i]
            IndexOfChilds = np.array(FindEqualTo(parent, NODES))
            if DEPTH[parent] == dpth:
                if len(IndexOfChilds) > 0:
                    x[parent] = mean(x[IndexOfChilds])
    return x, y


# ------------------------------------------------------------------------------
def plot_loop(parent, x, y, NODES):
    IndexOfChilds = np.array(FindEqualTo(parent, NODES))
    for i in range(0, len(IndexOfChilds)):
        child = IndexOfChilds[i]
        plt.plot(x[[parent, child]], y[[parent, child]], color=[.2, .8, 1, 1], linewidth=0.5)
        plot_loop(child, x, y, NODES)
    return


# ------------------------------------------------------------------------------
def PlotTree(NODES, TREEITEMS, COUNTS, SingleItems):
    fig, ax = plt.subplots()

    [x, y] = treelayout(NODES)
    plot_loop(0, x, y, NODES)

    circle_facecolor = [.2, .8, 1, 1]
    circle_edgecolor = [.2, .8, 1, 1]
    nodeindex_color = [0, 0, 0, 1]
    circle = Circle((x[0], y[0]), 0.03, fc=circle_facecolor, ec=circle_edgecolor, fill=True, linewidth=0.5)
    ax.add_patch(circle)
    plt.text(x[0], y[0], 'root', weight='bold', color=nodeindex_color, fontsize=10, family='calibri', style='normal',
             horizontalalignment='center', verticalalignment='center')
    for j in range(1, len(x)):
        circle = Circle((x[j], y[j]), 0.03, fc=circle_facecolor, ec=circle_edgecolor, fill=True, linewidth=0.5)
        ax.add_patch(circle)
        plt.text(x[j], y[j], str(j), weight='bold', color=nodeindex_color, fontsize=10, family='calibri',
                 style='normal', horizontalalignment='center', verticalalignment='center')

    for j in range(1, len(x)):
        tmp = SingleItems[TREEITEMS[j]]
        plt.text(x[j] + 0.03, y[j], tmp + ':' + str(COUNTS[j]), color=[1, 0, 0, 1], fontsize=10, family='consolas',
                 style='normal', horizontalalignment='left', verticalalignment='center')
        plt.text(x[j] + 0.03, y[j], tmp, color=[0, 0, 0, 1], fontsize=10, family='consolas', style='normal',
                 horizontalalignment='left', verticalalignment='center')

    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()
    return

def update(NewItem,spprt,FREQUENTITEMSETS,SUPPORTS):
    FREQUENTITEMSETS.append(NewItem)
    SUPPORTS.appedn(spprt)
    return FREQUENTITEMSETS,SUPPORTS
#--------------------------------------------------------------------------------
def ConStructTree(DATABASE,OCCS,SingleItems,MinAbsSupp):
    supports = np.sum(DATABASE*OCCS.reshape(-1,1),axis=0)
    I = np.argsort(-supports)
    DATABASE = DATABASE[:,I]
    SingleItems = SingleItems[I]
    supports = supports[I]
    NODES = np.array([-1]); TREEITEMS =np.array([-1]); COUNTS = np.array([-1])
    I = np.nonzero(MinAbsSupp<=supports)[0]
    if I.shape[0]>0:
        for trind in range(0,DATABASE.shape[0]):
            Transaction = DATABASE[trind,:]
            if sum(Transaction)>0:
                TransactionIndices = np.nonzero(Transaction==1)[0]
                if trind == 0:
                    NODES = np.array([-1]); TREEITEMS = np.array([-1]); COUNTS = np.array([-1])
                    for j in range(0,len(TransactionIndices)):
                        NODES = np.hstack((NODES,[j]))
                        TREEITEMS = np.hstack((TREEITEMS,[TransactionIndices[j]]))
                        COUNTS = np.hstack((COUNTS,[OCCS[trind]]))
                else:
                    ParentIndex = 0
                    index = 0
                    loop = 1
                    while loop:
                        I = np.nonzero(NODES == ParentIndex)[0]
                        ChildNodeItems = []
                        ChildNodeIndices = []
                        for i in I:
                            ChildNodeItems.append(TREEITEMS[i])
                            ChildNodeIndices.append(i)
                            I = np.nonzero(TransactionIndices[index] == ChildNodeItems)[0]
                        if I.shape[0]>0:
                            COUNTS[ChildNodeIndices[I[0]]] += OCCS[trind]
                            ParentIndex = ChildNodeIndices[I[0]]
                            index += 1
                        else:
                            loop = 0
                        if index == TransactionIndices.shape[0]:
                            loop = 0

                    for i in range(index,TransactionIndices.shape[0]):
                        NODES = np.hstack((NODES,[ParentIndex]))
                        TREEITEMS = np.hstack((TREEITEMS,[TransactionIndices[i]]))
                        COUNTS = np.hstack((COUNTS,[OCCS[trind]]))
                        ParentIndex = len(NODES)-1
    return NODES,TREEITEMS,COUNTS,DATABASE,SingleItems
#--------------------------------------------------------------------------------------------------------------------------
def FIM_PrefixTree_BottomUp_Loop(prefixitemindex, Newitem, NODESx, TREEITEMSx, COUNTSx, Singleitemsx, MinAbsSupp, FREQUENTITEMSETS, SUPPORTS):
    SuffixNodeIDs = np.nonzero(prefixitemindex==TREEITEMSx)[0]
    DATABASEx = np.zeros((SuffixNodeIDs.shape[0],Singleitemsx.shape[0]),dtype="int64")
    OCCSx = np.zeros(len(SuffixNodeIDs),dtype="int64")
    for i in range(0,len(SuffixNodeIDs)):
        node = SuffixNodeIDs[i]; trTREEITEMS = np.array([],dtype="int32"); nodecount = COUNTSx[SuffixNodeIDs[i]];
        while node!= 0:
            node = NODESx[node]
            if node!= 0:
                trTREEITEMS = np.hstack((np.array(TREEITEMSx[node],dtype="int32"),trTREEITEMS))
            if trTREEITEMS.shape[0] > 0:
                DATABASEx[i,trTREEITEMS] = 1
                OCCSx[i] = 1*nodecount
    I = np.nonzero(OCCSx>0)[0]
    DATABASEx = DATABASEx[I,:]
    OCCSx = OCCSx[I]
    (NODESxX,TREEITEMSxX,COUNTSxX,DATABASExX,SingleitemsxX) = ConStructTree(DATABASEx,OCCSx,Singleitemsx,MinAbsSupp)
    proSUPPORTS = np.zeros((np.max(TREEITEMSxX)+1),dtype="int64")
    for i in range(0,proSUPPORTS.shape[0]):
        I = np.nonzero(np.array(TREEITEMSxX) == i)[0]
        proSUPPORTS[i] = sum(np.array(COUNTSxX)[I])
    Is = np.nonzero(proSUPPORTS>=MinAbsSupp)[0]
    if len(Is)>0:
        for j in range(len(Is)-1,-1,-1):
            prefixitemindexX = Is[j]
            NewItemX = list(Newitem)
            #hata olabilir
            NewItemX.insert(0,SingleitemsxX[prefixitemindexX][0])
            (FREQUENTITEMSETS,SUPPORTS) = update(NewItemX,proSUPPORTS[Is[j]],FREQUENTITEMSETS,SUPPORTS)
            (FREQUENTITEMSETS,SUPPORTS) = FIM_PrefixTree_BottomUp_Loop(prefixitemindexX,NewItemX,NODESxX,TREEITEMSxX,COUNTSxX,SingleitemsxX,MinAbsSupp,FREQUENTITEMSETS,SUPPORTS)
    return FREQUENTITEMSETS,SUPPORTS
#-------------------------------------------------
def Display_Results(FREQUENTITEMSETS,SUPPORTS):
    results = []
    for i in range(0,len(FREQUENTITEMSETS)):
        temp_support=np.around(SUPPORTS[i],5)
        new_item = FREQUENTITEMSETS[i]
        temp_result = ""
        for j in range(0,len(new_item)):
            temp_item = new_item[j]
            if temp_item:
                temp_result = temp_result + new_item[j]
        TMP = "#" + str(i+1) + " / " + str(temp_result) + " / " + "supp: " + str(temp_support)
        print(TMP)
    return results
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
file_name = "excel.xlsx"
wb=openpyxl.load_workbook(filename=file_name,read_only=True)
sheet=wb[wb.sheetnames[0]]
database=np.array(pd.DataFrame(sheet.values))
wb.close()
#----------------
numOfTransactions=database.shape[0];
#---------------
allItems=[]
for i in range(0,database.shape[0]):
    transaction=database[i,:]
    for item in transaction:
        if item!=None:
            allItems.append(item)
allItems=np.array(allItems)
uniqueItems=np.unique(allItems)
uniqueItems=[str(item) for item in uniqueItems]
#-----------------
numOfUniqueItems=len(uniqueItems)
#-----------------
datamatrix=np.zeros((numOfTransactions,numOfUniqueItems),dtype='int8')
for i in range(0,database.shape[0]):
    transaction=database[i,:]
    for item in transaction:
        if item!=None:
            I=uniqueItems.index(item)
            datamatrix[i,I] = 1
del database
minTransactionLength = min(np.sum(datamatrix,axis=1))
maxTransactionLength = max(np.sum(datamatrix,axis=1))
aveTransactionLength = np.mean((np.sum(datamatrix,axis=1)))
density = np.sum(datamatrix)/(datamatrix.shape[0]*datamatrix.shape[1])

print("__________________________________________")
print("                                          ")
print("            DATABASE STATISTICS           ")
print("                                          ")
print("__________________________________________")
print("Number of Transactions...............:",str(numOfTransactions))
print("Number of Unique Items..............:",str(numOfUniqueItems))
print("Min. Transaction Length..............:",str(minTransactionLength))
print("Max. Transaction Length..............:",str(maxTransactionLength))
print("Ave. Transaction Length..............:",str(aveTransactionLength))
print("Density..............................:",str(density))
#---------------------
counts = np.sum(datamatrix,axis=0)
#---------------------
I = np.argsort(-counts)
counts = counts[I]
uniqueItems = [uniqueItems[i] for i in I]
datamatrix = datamatrix[:,I]

print("__________________________________________________ ")
print("                                                   ")
print("  FIM_PrefixTree Algorithm BottomUp Recursive DFS  ")
print("                                                   ")
print("__________________________________________________ ")
ElapsedTime = TicToc()
ElapsedTime.tic()

DATABASE = datamatrix
SingleItems = np.array(uniqueItems)
MinSupp = 0.001
MinAbsSupp = MinSupp*DATABASE.shape[0]

NumOfTransaction = DATABASE.shape[0]
InitialSupports = np.sum(DATABASE,axis=0)
ItemsToBeRemained = np.nonzero(InitialSupports>=MinAbsSupp)[0]
DATABASE = DATABASE[:,ItemsToBeRemained]
SingleItems = SingleItems[ItemsToBeRemained]
InitialSupports = InitialSupports[ItemsToBeRemained]

print("agaç olusturuluyor")
OCCS = np.ones(DATABASE.shape[0],dtype="int8")
(NODES,TREEITEMS,COUNTS,DATABASE,SingleItems) = ConStructTree(DATABASE,OCCS,SingleItems,MinAbsSupp)
PlotTree(NODES,TREEITEMS,COUNTS,SingleItems)
supports = np.sum(DATABASE,axis=0)
print("...agactan ogesetleri elde ediliyor")
FREQUENTITEMSETS = [];SUPPORTS = [];
for k in range(DATABASE.shape[1]-1,-1,-1):
    prefixitemndex = np.array([k])
    NewItem = SingleItems[k]
    (FREQUENTITEMSETS,SUPPORTS) = update(list(NewItem),supports[k],FREQUENTITEMSETS,SUPPORTS)
    (FREQUENTITEMSETS,SUPPORTS) = FIM_PrefixTree_BottomUp_Loop(prefixitemndex,NewItem,NODES,TREEITEMS,COUNTS,SingleItems,MinAbsSupp,FREQUENTITEMSETS,SUPPORTS)

I = np.argsort(-np.array(SUPPORTS))
FREQUENTITEMSETS = [FREQUENTITEMSETS[i] for i in I]
SUPPORTS = [SUPPORTS[i] for i in I]
ElapsedTime.toc("Elapsed Times is")
print("--------------------------------------")
F = []
for i in range(0,len(FREQUENTITEMSETS)):
    fi = list(FREQUENTITEMSETS[i])
    if len(fi)>1:
        fi.sort()
    F.append(fi)
S = [supp/NumOfTransaction for supp in SUPPORTS]

















