#Dfs Algorithm
R = ["a", "b", "c", "d", "e"]


def dfsLoop(item, R):
    E = R[R.index(item[-1]) + 1 :]
    for suffix in E:
        print(item + suffix)
        dfsLoop(item + suffix, E)


for item in R:
    print(item)
    dfsLoop(item, R)
