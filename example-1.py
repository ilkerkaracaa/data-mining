# Python3 code to select
# data from excel
import xlwings as xw

# Specifying a sheet
ws = xw.Book("FIM_Demo3.xlsx").sheets["Sheet1"]

# Selecting data from
# a single cell
arr = []
string = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
for i in string:
    v1 = ws.range(f"{i}1:{i}10").value
    # v2 = ws.range("F5").value
    if all(eleman is None for eleman in v1):
        break
    arr.append(v1)

arrTwo = []
count = 0
for i in arr:
    for j in i:
        if j is None:
            continue
        else:
            arrTwo.append(j)
            count += 1
print(count)
unique_set = set(arrTwo)
liste = list(unique_set)
listeSayac = []
for i in liste:
    count = 0
    for j in arrTwo:
        if i == j:
            count += 1
    listeSayac.append(count)

for eleman1, eleman2 in zip(liste, listeSayac):
    print(eleman1, eleman2)

for i in range(len(listeSayac)):
    if listeSayac[i] >= 5:
        print(f"{liste[i]} : {listeSayac[i]}")
