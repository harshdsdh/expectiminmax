inp = [['3', '0', '0', '0', '0', '0'], ['0', '0', '0', '0', '0', '0'], ['0', '0', '0', '0', '0', '3'],
       ['0', '0', '0', '0', '0', '0'], ['0', '0', '0', '0', '0', '0'], ['0', '0', '0', '3', '0', '0']]

inp1 = [['3', '0', '0', '0', '0', '0','0'], ['0', '3', '0', '0', '0', '0','0'], ['0', '0', '0', '0', '0','0', '3'],
       ['0', '0', '3', '0', '0', '0','0'], ['0', '0', '0', '3', '0', '0','0'], ['0', '0', '0', '3', '0', '0','0']]

inp2 = [['3', '0', '0', '0', '0', '0','0','0'], ['0', '3', '0', '0', '0', '0','0','0'], ['0', '0', '0', '0', '0','0', '3', '0'],
       ['0', '0', '3', '0', '0', '0','0','0'], ['0', '0', '0', '3', '0', '0','0','0'], ['0', '0', '0', '3', '0', '0','0','3']]

inp3 = [['3', '0', '0', '0', '0', '3'], ['0', '0', '0', '0', '3', '0'], ['0', '0', '0', '0', '0', '3'],
       ['0', '3', '0', '3', '0', '0'], ['3', '0', '0', '0', '0', '0'], ['0', '0', '0', '3', '0', '0']]

inp4 = [['3', '0', '0', '0', '0', '3','0'], ['0', '0', '0', '0', '3', '0','0'], ['0', '0', '0', '0', '0', '3','0'],
       ['0', '3', '0', '3', '0', '0','0'], ['3', '0', '0', '0', '0', '0','0'], ['0', '0', '0', '3', '0', '0','0']]


#nidhi
inp5 = [['3', '0', '0', '0', '0', '0','0'], ['3', '0', '0', '0', '0', '0','0'], ['0', '0', '0', '0', '0', '3','0'],
       ['0', '0', '0', '3', '0', '0','0'], ['0', '0', '0', '0', '0', '0','0'], ['3', '0', '0', '3', '0', '0','0']]

inp6 = [['3', '0', '0', '0', '0', '0'], ['0', '3', '0', '0', '0', '0'], ['3', '0', '0', '0', '0', '3'],
       ['3', '0', '3', '0', '0', '0'], ['0', '0', '0', '3', '0', '0'], ['0', '0', '0', '3', '0', '0']]

inp7 = [['3', '0', '0', '0', '0', '0','0','3','0'], ['0', '3', '0', '0', '0', '0','3','0','0'], ['3', '0', '0', '0', '0','0', '3', '0','0'],
       ['3', '0', '3', '0', '0', '0','0','0','0'], ['3', '0', '0', '3', '0', '0','0','0','0'], ['3', '0', '0', '3', '0', '0','0','0','3']]

inp8 = [['3', '0', '3', '0', '0', '3'], ['0', '3', '0', '0', '3', '0'], ['0', '0', '0', '0', '0', '3'],
       ['0', '3', '0', '3', '0', '0'], ['3', '0', '3', '0', '0', '0'], ['3', '0', '0', '3', '0', '0']]

inp9 = [['3', '0', '0', '0', '0', '0','0','3','0','0'], ['0', '3', '0', '0', '0', '0','3','0','0','0'], ['3', '0','0', '0', '0', '0','0', '3', '0','0'],
       ['3', '0', '3', '0', '0', '0','0','0','0','0'], ['3', '0', '0', '3', '0', '0','0','0','0','0'], ['3', '0', '0', '3', '0', '0','0','0','3','0']]

#riya
inp10 = [['3', '0', '0', '0', '0', '0'], ['0', '0', '0', '0', '0', '0'], ['0', '0', '0', '0', '0', '3'],
       ['0', '0', '0', '0', '0', '0'], ['0', '0', '0', '0', '0', '0'], ['0', '0', '0', '3', '0', '0']]

inp11 = [['3', '0', '0', '0', '0', '0','0'], ['0', '3', '0', '0', '0', '0','0'], ['0', '0', '0', '0', '0','0', '3'],
       ['0', '0', '3', '0', '0', '0','0'], ['0', '0', '0', '3', '0', '0','0'], ['0', '0', '0', '3', '0', '0','0']]
inp12 = [['3', '0', '0', '0', '0', '0','0','3','0'], ['0', '3', '0', '0', '0', '0','3','0','0'], ['3', '0', '0', '0', '0','0', '3', '0','0'],
       ['3', '0', '3', '0', '0', '0','0','0','0'], ['3', '0', '0', '3', '0', '0','0','0','0'], ['3', '0', '0', '3', '0', '0','0','0','3']]

inp13 = [['3', '0', '3', '0', '0', '3'], ['0', '3', '0', '0', '3', '0'], ['0', '0', '0', '0', '0', '3'],
       ['0', '3', '0', '3', '0', '0'], ['3', '0', '3', '0', '0', '0'], ['3', '0', '0', '3', '0', '0']]
inp14 = [['3', '0', '0', '0', '0', '3','0'], ['0', '0', '0', '0', '3', '0','0'], ['0', '0', '0', '0', '0', '3','0'],
       ['0', '3', '0', '3', '0', '0','0'], ['3', '0', '0', '0', '0', '0','0'], ['0', '0', '0', '3', '0', '0','0']]


import copy
import random

def getRandomNumbers(n):
    l=[]
    while(len(l)<=4):
        i = random.randint(0,n-1)
        if i not in l:
            l.append(i)
    return l


def traverseMatrix(matrix, random_r,random_c,r,c):
    i, j = random_r, random_c
    score = 0
    if matrix[i][j] == '0':
        score+=1
        i, j = random_r, random_c
        i += 1
        c1 = 0
        while (i >= 0 and i < r and matrix[i][j] != '3' and matrix[i][j] != '1' and matrix[i][j] != '2' and c1<4):
            score += 1
            i += 1
            c1 += 1


        i, j = random_r, random_c
        c1 = 0
        i -= 1
        while (i >= 0 and i < r and matrix[i][j] != '3' and matrix[i][j] != '1' and matrix[i][j] != '2' and c1<4):
            score+=1
            i -= 1
            c1 += 1


        i, j = random_r, random_c
        c1 = 0
        j += 1
        while (j >= 0 and j < c and matrix[i][j] != '3' and matrix[i][j] != '1' and matrix[i][j] != 2 and c1<4):
            score+=1
            j += 1
            c1 += 1


        i, j = random_r, random_c
        c1 = 0
        j -= 1
        while (j >= 0 and j < c and matrix[i][j] != '3' and matrix[i][j] != '1' and matrix[i][j] != 2 and c1<4):
            score+=1
            j -= 1
            c1 += 1

    return score

def Player(matrix,player):
    r = len(matrix)
    c = len(matrix[0])
    #generate 8 random sets of numbers within boundaries of matrix
    ind=0
    m={}
    random_row = getRandomNumbers(r)
    random_col = getRandomNumbers(c)
    for i in range(0,len(random_row)):
        s = traverseMatrix(matrix,random_row[i],random_col[i],r,c)

        ind+=1
        m[ind] = [s,random_row[i],random_col[i]]

    for j in range(0,len(random_row)):
        t = len(random_row)-j-1
        s = traverseMatrix(matrix,random_row[t],random_col[j],r,c)

        ind+=1
        m[ind] = [s,random_row[t],random_col[j]]
    print(m)
    rowScore,colScore = playEMM(m)
    if player=='first':
        print('f',rowScore,colScore)
        matrix = markBoardP1(matrix,rowScore,colScore,r,c)

    return matrix


def markBoardP1(matrix,random_r,random_c,r,c):
    i, j = random_r, random_c
    if matrix[i][j] == '0':
        matrix[i][j]='1'
        i, j = random_r, random_c
        i += 1
        c1 = 0
        while (i >= 0 and i < r and matrix[i][j] != '3' and matrix[i][j] != '1' and matrix[i][j] != '2' and c1 < 4):
            if matrix[i][j]=='0':
                matrix[i][j]='x'
            elif matrix[i][j]=='y':
                matrix[i][j]='C'
            i += 1
            c1 += 1

        i, j = random_r, random_c
        c1 = 0
        i -= 1
        while (i >= 0 and i < r and matrix[i][j] != '3' and matrix[i][j] != '1' and matrix[i][j] != '2' and c1 < 4):
            if matrix[i][j] == '0':
                matrix[i][j] = 'x'
            elif matrix[i][j] == 'y':
                matrix[i][j] = 'C'
            i -= 1
            c1 += 1

        i, j = random_r, random_c
        c1 = 0
        j += 1
        while (j >= 0 and j < c and matrix[i][j] != '3' and matrix[i][j] != '1' and matrix[i][j] != 2 and c1 < 4):
            if matrix[i][j] == '0':
                matrix[i][j] = 'x'
            elif matrix[i][j] == 'y':
                matrix[i][j] = 'C'
            j += 1
            c1 += 1

        i, j = random_r, random_c
        c1 = 0
        j -= 1
        while (j >= 0 and j < c and matrix[i][j] != '3' and matrix[i][j] != '1' and matrix[i][j] != 2 and c1 < 4):
            if matrix[i][j] == '0':
                matrix[i][j] = 'x'
            elif matrix[i][j] == 'y':
                matrix[i][j] = 'C'
            j -= 1
            c1 += 1
    elif matrix[i][j] == 'x' or matrix[i][j]=='y':
        print("wrong move p1 laser present")
    return matrix

def markBoardP2(matrix,random_r,random_c,r,c):
    i, j = random_r, random_c
    if matrix[i][j] == '0':
        matrix[i][j]='2'
        i, j = random_r, random_c
        i += 1
        c1 = 0
        while (i >= 0 and i < r and matrix[i][j] != '3' and matrix[i][j] != '1' and matrix[i][j] != '2' and c1 < 4):
            if matrix[i][j]=='0':
                matrix[i][j]='y'
            elif matrix[i][j]=='x':
                matrix[i][j]='C'
            i += 1
            c1 += 1

        i, j = random_r, random_c
        c1 = 0
        i -= 1
        while (i >= 0 and i < r and matrix[i][j] != '3' and matrix[i][j] != '1' and matrix[i][j] != '2' and c1 < 4):
            if matrix[i][j] == '0':
                matrix[i][j] = 'y'
            elif matrix[i][j] == 'x':
                matrix[i][j] = 'C'
            i -= 1
            c1 += 1

        i, j = random_r, random_c
        c1 = 0
        j += 1
        while (j >= 0 and j < c and matrix[i][j] != '3' and matrix[i][j] != '1' and matrix[i][j] != 2 and c1 < 4):
            if matrix[i][j] == '0':
                matrix[i][j] = 'y'
            elif matrix[i][j] == 'x':
                matrix[i][j] = 'C'
            j += 1
            c1 += 1

        i, j = random_r, random_c
        c1 = 0
        j -= 1
        while (j >= 0 and j < c and matrix[i][j] != '3' and matrix[i][j] != '1' and matrix[i][j] != 2 and c1 < 4):
            if matrix[i][j] == '0':
                matrix[i][j] = 'y'
            elif matrix[i][j] == 'x':
                matrix[i][j] = 'C'
            j -= 1
            c1 += 1
    elif matrix[i][j] == 'x' or matrix[i][j]=='y':
        print("wrong move p2 laser present")
    return matrix

def chanceMode(minmap):
    i = 1
    t = 0
    chanceScoreMap = {}
    while (i < len(minmap)):
        random_var = random.randint(0, 1)
        if random_var == 0:
            t += 1
            chanceScoreMap[t] = minmap[i]
        elif random_var==1:
            t += 1
            chanceScoreMap[t] = minmap[i + 1]
        i += 2
    return chanceScoreMap

def playEMM(scoresMap):
    # key is index and values = score,row,col
    maxMap = maximizeS(scoresMap)
    print('maxMap', maxMap)
    minMap = minimizeS(maxMap)
    print('minMap', minMap)
    finalmaxMap = chanceMode(minMap)
    print('chanceMap', finalmaxMap)
    return (finalmaxMap[1][1],finalmaxMap[1][2])
def minimizeS(scoresMap):
    i = 1
    t = 0
    minScoreMap = {}
    while (i < len(scoresMap)):
        if scoresMap[i][0] <= scoresMap[i + 1][0]:
            t += 1
            minScoreMap[t] = scoresMap[i]
        else:
            t += 1
            minScoreMap[t] = scoresMap[i + 1]
        i += 2
    return minScoreMap

def maximizeS(scoresMap):
    i=1
    t=0
    maxScoreMap={}
    while(i<len(scoresMap)):
        if scoresMap[i][0]>=scoresMap[i+1][0]:
            t+=1
            maxScoreMap[t] = scoresMap[i]
        else:
            t += 1
            maxScoreMap[t] = scoresMap[i+1]
        i+=2
    return maxScoreMap
def playGame(matrix,count_p1,count_p2):
    #test_matrix from main
    t=0
    print(matrix)

    while(t<5):

        #matrix = Player(matrix, 'first')
        randr = 0
        randc = 0
        r,c  = len(matrix),len(matrix[0])
        while (matrix[randr].count('0') <= 0):
            randr += 1

        randc = random.randint(0,c-1)
        matrix = markBoardP2(matrix, randr, randc, r, c)

        matrix = Player(matrix, 'first')
        (p1,p2,z) = calScore(matrix)
        print(matrix)
        print("p1,p2:",p1,p2)
        if p1 > p2:
            count_p1+=1
        elif p1 < p2:
            count_p2+=1
        if z==0:
            print("no space left break")
            print("p1,p2:",p1,p2)
            break
        t+=1
    return count_p1, count_p2
def calScore(m):
    s1 = 0
    s2 = 0
    countZeroes = 0
    for i in m:
        s1 += i.count('x')
        s1 += i.count('1')
        s2 += i.count('y')
        s2 += i.count('2')
        s1 += i.count('C')
        s2 += i.count('C')
        countZeroes+=i.count('0')

    #print("p1,p2:", s1, s2)
    return (s1,s2,countZeroes)
def main():
    player1=0
    player2=0
    t=0
    while(t<100):
        count_p1 = 0
        count_p2 = 0
        test_matrix = copy.deepcopy(inp4)
        p1,p2 = playGame(test_matrix,count_p1,count_p2)
        if p1>=p2:
            player1+=1
        elif p1<=p2:
            player2+=1
        t+=1
    print("total games played: ", t)
    print("total wins by player 1 and player 2: ", player1, player2)


main()