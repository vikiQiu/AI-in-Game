'''
    黑白棋（Reversi）
    Policy：一步最优
    Evaluation：当步翻转棋的个数
    Author：Wenqing
    Info：http://www.botzone.org/games#Reversi
'''

import json
import numpy as np
import random
import time
import random
random.seed(1)

DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)) # 方向向量

# 放置棋子，计算新局面
DIM_BOARD = 8

def place(board, x, y, color):
    if x < 0:
        return False # No change on the board
    board[x][y] = color
    valid = False
    ch_num = 0 # Number reversed
    old_board = board.copy()
    for d in range(8):
        i = x + DIR[d][0]
        j = y + DIR[d][1]
        # 找方向d上的连续的对方棋子的最后一个位置
        while 0 <= i and i < 8 and 0 <= j and j < 8 and old_board[i][j] == -color:
            i += DIR[d][0]
            j += DIR[d][1]
        # 若方向d上对方棋子的最后一个位置之后是我方棋子，则将之前搜索的对方棋子全部变成我方棋子。
        if 0 <= i and i < 8 and 0 <= j and j < 8 and old_board[i][j] == color:
            while True:
                i -= DIR[d][0]
                j -= DIR[d][1]
                if i == x and j == y:
                    break
                valid = True
                board[i][j] = color
                ch_num += 1
    return valid, ch_num, board


def cal_board(board):
    num_black = num_white = 0
    for i in range(DIM_BOARD):
        for j in range(DIM_BOARD):
            if board[i, j] == 1:
                num_black += 1
            elif board[i, j] == -1:
                num_white += 1
    return num_black, num_black


# 随机产生决策
def choice(board, color):
    '''
    Make decision
    @param color: 1 -> black; -1 -> white
    '''
    x = y = -1
    moves = []
    earnings = []
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                newBoard = board.copy()
                valid, ch_num, _ = place(newBoard, i, j, color)
                if valid:
                    moves.append((i, j))
                    earnings.append(ch_num)
    if len(moves) == 0:
        return -1, -1
    earnings = np.array(earnings) / sum(earnings)
    return moves[np.random.choice(np.arange(len(moves)), p=earnings)]


# 处理输入，还原棋盘
def init_board(full_input):
    '''
    @param full_input: A Json 
    {"requests":[{"x":-1,"y":-1}],"responses":[]}
    '''
    requests = full_input["requests"]
    responses = full_input["responses"]
    board = np.zeros((8, 8), dtype=np.int)
    board[3][4] = board[4][3] = 1
    board[3][3] = board[4][4] = -1
    myColor = 1 # balck
    if requests[0]["x"] >= 0:
        # 对方先下，将对方第一步下完
        myColor = -1 # white
        place(board, requests[0]["x"], requests[0]["y"], -myColor)
    turn = len(responses)
    for i in range(turn):
        place(board, responses[i]["x"], responses[i]["y"], myColor)
        if i + 1 < len(requests):
            place(board, requests[i + 1]["x"], requests[i + 1]["y"], -myColor)
    return board, myColor


def init_board_submit():
    full_input = json.loads(input())
    return init_board(full_input)


if __name__ == '__main__':
    board, myColor = init_board_submit()
    x, y = choice(board, myColor)
    print(json.dumps({"response": {"x": int(x), "y": int(y)}}))






