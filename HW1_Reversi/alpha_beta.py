'''
    黑白棋（Reversi）
    Policy：k步MinMax最优
    Evaluation：k步后局面棋数差
    Author：Wenqing
    Info：http://www.botzone.org/games#Reversi
    prob_rand V.S. alpha_beta -> 16:33:1 (prob_rand:alpha_beta:draw)
'''

import json
import numpy as np
import random
import time
import random
random.seed(1)

DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))  # 方向向量

# 放置棋子，计算新局面
DIM_BOARD = 8
K_STEP = 6
INF = np.Inf
TIMELIMIT = 5.75


def get_weight(board, color):
    w = np.zeros((DIM_BOARD, DIM_BOARD))
    w_part = np.array([[90, -60, 10, 10],
                       [-60, -80, 5, 5],
                       [10, 5, 1, 1],
                       [10, 5, 1, 1]])
    w[0:4, 0:4] = w_part
    w[4:8, 0:4] = w_part[::-1, ]
    w[0:4, 4:8] = w_part[:, ::-1]
    w[4:8, 4:8] = w_part[::-1, ::-1]
    if board[0, 0] == color:
        w[0, 1] = w[1, 0] = 60
        w[1, 1] = 5
    if board[7, 0] == color:
        w[6, 0] = w[7, 1] = 60
        w[6, 1] = 5
    if board[0, 7] == color:
        w[1, 7] = w[0, 6] = 60
        w[1, 6] = 5
    if board[7, 7] == color:
        w[6, 7] = w[7, 6] = 60
        w[6, 6] = 5
    return w


# def get_weight(x=900, y=-600):
#     w = np.ones((DIM_BOARD, DIM_BOARD))
#     w[0, 0] = w[DIM_BOARD-1, DIM_BOARD-1] = w[0, DIM_BOARD-1] = w[DIM_BOARD-1, 0] = x
#     w[DIM_BOARD-2, DIM_BOARD-2] = w[1, 1] = w[1, DIM_BOARD-2] = w[DIM_BOARD-2, 1] = y
#     return w


def place(board, x, y, color):
    '''
    Place (x, y) on the current board, with the given color
    :param board: A 2d list (8*8)
    :param x: [0, 7] int
    :param y: [0, 7] int
    :param color: Black -> 1; White -> -1
    :return: valid, ch_num, board
    valid: Is (x, y) a valid position on the current board. Boolean
    ch_num: Calculate the number of the opposite chess reversed.
    board: the new board
    '''
    if x < 0:
        return False  # No change on the board
    board[x][y] = color
    valid = False
    ch_num = 0  # Number reversed
    for d in range(8):
        i = x + DIR[d][0]
        j = y + DIR[d][1]
        # 找方向d上的连续的对方棋子的最后一个位置
        while 0 <= i < 8 and 0 <= j < 8 and board[i][j] == -color:
            i += DIR[d][0]
            j += DIR[d][1]
        # 若方向d上对方棋子的最后一个位置之后是我方棋子，则将之前搜索的对方棋子全部变成我方棋子。
        if 0 <= i < 8 and 0 <= j < 8 and board[i][j] == color:
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
    num_black = np.sum(board == 1)
    num_white = np.sum(board == -1)

    return num_black, num_white


def legal_actions(board, color):
    '''
    Get legal actions.
    '''
    moves = []
    boards = []
    scores = []
    for i in range(8):
        for j in range(8):
            # (i, j) is an action
            if board[i][j] == 0:
                newBoard = board.copy()
                valid, ch_num, newBoard = place(newBoard, i, j, color)
                # Whether this action is legal
                if valid:
                    moves.append((i, j))
                    boards.append(newBoard)
                    scores.append(evaluation_middle(newBoard, color))

    # If there is no legal action, the board stays the origin one.
    if len(moves) == 0:
        boards.append(board)
    else:
        tmp = sorted(zip(scores, moves, boards))
        tmp.reverse()
        moves = [x for _, x, _ in tmp]
        boards = [y for _, _, y in tmp]

    return moves, boards


def evaluation(board, color):
    num_black = np.sum(board == 1)
    num_white = np.sum(board == -1)
    if color == 1:
        return num_black - num_white
    else:
        return num_white - num_black


def evaluation_middle(board, color, num=None):
    if num is None:
        num = np.sum(np.abs(board))
    w = get_weight(board, color)
    if num <= 55:
        score = color * np.sum(w*board)
    else:
        score = evaluation(board, color)
    return score


class AlphaBetaTree:
    def __init__(self, board, color, steps=1):
        '''
        The tree to search for a better choice.
        @param board: The current given board State.
        @param color: Now which color role need to play.
        @param steps: Number of steps take into account. Must >= 1.
        '''
        # root: The current state.
        self.board = board
        self.color = color
        self.steps = steps  # 一共搜索的步数
        self.num = np.sum(np.abs(board))  # 当前局数
        self.start_time = time.time()

        # 搜索步数的调整
        if self.num > 55:
            self.steps = 64 - self.num

    def run_ppt(self, father_board, color, depth, alpha=-INF, beta=INF):
        if depth == self.steps or time.time() - self.start_time > TIMELIMIT:
            return evaluation_middle(father_board, -color, self.num)

        if color == self.color:
            moves, boards = legal_actions(father_board, color)
            for i, move in enumerate(moves):
                if move in [(0, 0), (0, 7), (7, 0), (7, 7)]:
                    return max(alpha, self.run_ppt(boards[i], -color, depth+1, alpha, beta))
            for child_board in boards:
                alpha = max(alpha, self.run_ppt(child_board, -color, depth+1, alpha, beta))
                if beta <= alpha:
                    break
            return alpha

        else:
            moves, boards = legal_actions(father_board, color)
            for i, move in enumerate(moves):
                if move in [(0, 0), (0, 7), (7, 0), (7, 7)]:
                    return min(beta, self.run_ppt(boards[i], -color, depth+1, alpha, beta))
            for child_board in boards:
                beta = min(beta, self.run_ppt(child_board, -color, depth+1, alpha, beta))
                if beta <= alpha:
                    break
            return beta

    def alpha_beta_search(self):
        '''
        Min-Max Tree Method
        @param
        '''
        moves, boards = legal_actions(self.board, self.color)

        # If there is no legal action, return (-1, -1)
        if len(moves) == 0:
            return -1, -1
        elif len(moves) == 1:
            return moves[0]

        for move in moves:
            if move in [(0, 0), (0, 7), (7, 0), (7, 7)]:
                return move

        alpha = -INF
        beta = INF
        best_move = moves[0]
        # Calculate the value of each successor
        for i, child_board in enumerate(boards):
            value = self.run_ppt(child_board, -self.color, 1, alpha, beta)
            if value > alpha:
                best_move = moves[i]
                alpha = value
        return best_move


# Min-Max Method
def choose_alpha_beta(board, color, is_random=False):
    '''
    Make decision
    @:param board:
    @:param color: 1 -> black; -1 -> white
    @:param is_random: Whether to choose action with its value estimated
    '''
    min_max_tree = AlphaBetaTree(board, color, K_STEP)
    return min_max_tree.alpha_beta_search()


# 处理输入，还原棋盘
def initBoard(fullInput):
    '''
    @param fullInput: A Json
    {"requests":[{"x":-1,"y":-1}],"responses":[]}
    '''
    requests = fullInput["requests"]
    responses = fullInput["responses"]
    board = np.zeros((8, 8), dtype=np.int)
    board[3][4] = board[4][3] = 1
    board[3][3] = board[4][4] = -1
    myColor = 1  # balck
    if requests[0]["x"] >= 0:
        # 对方先下，将对方第一步下完
        myColor = -1  # white
        place(board, requests[0]["x"], requests[0]["y"], -myColor)
    turn = len(responses)
    for i in range(turn):
        place(board, responses[i]["x"], responses[i]["y"], myColor)
        if i + 1 < len(requests):
            place(board, requests[i + 1]["x"], requests[i + 1]["y"], -myColor)
    return board, myColor


def initBoardSubmit():
    fullInput = json.loads(input())
    return initBoard(fullInput)


if __name__ == '__main__':
    current_board, myColor = initBoardSubmit()
    # fullInput = {"requests":[{"x":2,"y":3},{"x":2,"y":5},{"x":4,"y":2},{"x":1,"y":2},{"x":3,"y":2},{"x":5,"y":4},{"x":4,"y":1},{"x":5,"y":5},{"x":4,"y":5},{"x":5,"y":3},{"x":3,"y":5},{"x":1,"y":5},{"x":1,"y":3},{"x":2,"y":7},{"x":4,"y":7},{"x":0,"y":4}],"responses":[{"y":4,"x":2},{"y":2,"x":2},{"x":2,"y":6},{"y":1,"x":5},{"x":5,"y":2},{"y":2,"x":0},{"y":4,"x":6},{"y":0,"x":3},{"x":3,"y":6},{"x":1,"y":4},{"y":6,"x":5},{"x":0,"y":5},{"x":0,"y":3},{"x":3,"y":7},{"y":1,"x":3}]}
    # current_board, myColor = initBoard(fullInput)
    x, y = choose_alpha_beta(current_board, myColor)
    print(json.dumps({"response": {"x": int(x), "y": int(y)}}))



