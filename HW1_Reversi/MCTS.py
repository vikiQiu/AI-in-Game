'''
    黑白棋（Reversi）
    Policy：MCTS
    Evaluation：k步后局面棋数差 + 四角占有数差
    Author：Wenqing
    Info：http://www.botzone.org/games#Reversi
'''

import json
import numpy as np
import random
import time
from math import log, sqrt

DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))  # 方向向量

# 放置棋子，计算新局面
DIM_BOARD = 8
K_STEP = 5
INF = np.Inf
TIMELIMIT  = 5.9


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


def only_legal_actions(board, color):
    moves = []
    for i in range(DIM_BOARD):
        for j in range(DIM_BOARD):
            if board[i, j] == 0:
                valid, _, _ = place(board.copy(), i, j, color)
                if valid:
                    moves.append((i, j))
    return moves


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


class MCTS:
    def __init__(self, board, color, max_actions=500, time=3):
        '''
        The tree to search for a better choice.
        @param board: The current given board State.
        @param color: Now which color role need to play.
        @param steps: Number of steps take into account. Must >= 1.
        '''
        # root: The current state.
        self.board = board
        self.color = color
        self.calculation_time = float(time)  # 最大运算时间
        self.max_actions = max_actions  # 每次模拟对局最多进行的步数
        self.confident = 1.96
        self.plays = {}
        self.wins = {}
        self.max_depth = 1

    def get_action(self):
        moves, boards = legal_actions(self.board, self.color)
        if len(moves) == 0:
            return [(-1, -1)], [1]
        elif len(moves) == 1:
            return moves, [1]

        for move in moves:
            if move in [(0, 0), (0, 7), (7, 0), (7, 7)]:
                return [move], [1]

        self.plays = {}
        self.wins = {}
        simulations = 0
        begin = time.time()
        while time.time() - begin < self.calculation_time:
            board, color, visited_states, depth = self.run_selection(self.board.copy(), self.color)
            who_win = self.run_simulation(board, color)

            if depth > self.max_depth:
                self.max_depth = depth

            for visited_color, move in visited_states:
                if (visited_color, move) not in self.plays:
                    self.plays[(visited_color, move)] = 0
                    self.wins[(visited_color, move)] = 0
                self.plays[(visited_color, move)] += 1  # 当前路径上所有着法的模拟次数加1
                if visited_color == who_win:
                    self.wins[(visited_color, move)] += 1  # 获胜玩家的所有着法的胜利次数加1
            # print('Simulation %d' % simulations)
            simulations += 1

        # print("[INFO]total simulations=", simulations)

        moves, percent_wins = self.select_one_move(moves)  # 选择最佳着法
        # print('Maximum depth searched:', self.max_depth)

        # print("AI move: %d,%d\n" % (move[0], move[1]))

        return moves, percent_wins

    def select_one_move(self, moves):
        percent_wins = [self.wins.get((self.color, move), 0)/self.plays.get((self.color, move), 1) for move in moves]
        return moves, percent_wins

    @staticmethod
    def has_a_winner(board, color):
        '''
        Is there a winner
        :param board: The recent board
        :param color: The current player
        :return: 1 -> Black win; -1 -> White win; 0 -> Draw; None -> Game not over yet.
        '''
        if len(only_legal_actions(board, color)) > 0:
            return None
        if len(only_legal_actions(board, -color)) > 0:
            return None
        num_black, num_white = cal_board(board)
        delta = num_black - num_white
        if delta > 0:
            return 1
        elif delta < 0:
            return -1
        else:
            return 0

    def run_selection(self, board, color, visited_states=set(), depth=1):
        plays = self.plays.copy()
        wins = self.wins.copy()
        moves, _ = legal_actions(board, self.color)
        if len(moves) == 0:
            return board, -color, visited_states, depth
        if all(plays.get((color, move)) for move in moves):
            log_total = self.my_log(sum(plays[(color, move)] for move in moves))
            _, move = max(
                ((wins[(color, move)] / plays[(color, move)]) +
                 sqrt(self.confident * log_total / plays[(color, move)]), move)
                for move in moves)
        else:
            # 否则随机选择一个着法
            move = random.choice(moves)
        _, _, board = place(board.copy(), move[0], move[1], color)
        visited_states.add(move)

        if move in self.plays:
            return self.run_selection(board.copy(), -color, visited_states, depth+1)

        else:
            return board, -color, visited_states, depth

    def run_simulation(self, board, color):
        has_win = self.has_a_winner(board.copy(), color)
        if has_win is not None:
            return has_win
        move, board = self.sim_choice(board.copy(), color)
        return self.run_simulation(board, -color)

    @staticmethod
    def my_log(x):
        '''
        Define my log. Return -INF if x == 0
        :param x:
        :return:
        '''
        if x == 0:
            return -INF
        else:
            return log(x)

    @staticmethod
    def sim_choice(board, color):
        '''
        Make decision with probability given
        @:param board:
        @param color: 1 -> black; -1 -> white
        '''
        moves = []
        earnings = []
        boards = []
        num = np.sum(np.abs(board))
        for i in range(8):
            for j in range(8):
                if board[i][j] == 0:
                    new_board = board.copy()
                    valid, ch_num, new_board = place(new_board, i, j, color)
                    if valid:
                        moves.append((i, j))
                        boards.append(new_board)
                        earnings.append(evaluation_middle(new_board, color, num))
        if len(moves) == 0:
            return (-1, -1), board
        earnings = np.array(earnings)
        if min(earnings) <= 0:
            earnings -= (np.min(earnings) - 0.01)
        earnings = earnings / np.sum(earnings)
        ind = np.random.choice(np.arange(len(moves)), p=earnings)
        return moves[ind], boards[ind]


# Min-Max Method
def choose_mcts(board, color, is_random=False):
    '''
    Make decision
    @:param board:
    @:param color: 1 -> black; -1 -> white
    @:param is_random: Whether to choose action with its value estimated
    '''
    mcts = MCTS(board, color, time=TIMELIMIT)
    moves, earnings = mcts.get_action()
    if len(moves) == 1:
        return moves[0]
    if is_random:
        earnings = np.array(earnings) / sum(earnings)
        return moves[np.random.choice(np.arange(len(moves)), p=earnings)]
    else:
        moves = np.array(moves)
        return random.choice(moves[list(np.array(earnings) == max(earnings))])


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
    # fullInput = {"requests":[{"x":2,"y":3},{"x":2,"y":5},{"x":4,"y":2},{"x":3,"y":5},{"x":3,"y":2},{"x":1,"y":2},{"x":5,"y":2},{"x":1,"y":5},{"x":5,"y":0},{"x":2,"y":0},{"x":0,"y":5},{"x":5,"y":4},{"x":4,"y":5},{"x":3,"y":0},{"x":4,"y":0},{"x":0,"y":0},{"x":1,"y":0},{"x":2,"y":6},{"x":2,"y":7},{"x":0,"y":7},{"x":7,"y":0},{"x":6,"y":3},{"x":6,"y":2},{"x":7,"y":3},{"x":7,"y":5},{"x":6,"y":6},{"x":4,"y":7},{"x":4,"y":6},{"x":0,"y":3},{"x":6,"y":7}],"responses":[{"y":4,"x":2},{"x":2,"y":2},{"x":1,"y":4},{"x":5,"y":1},{"y":1,"x":4},{"x":3,"y":6},{"y":1,"x":1},{"y":1,"x":6},{"x":3,"y":1},{"x":2,"y":1},{"y":3,"x":1},{"y":5,"x":6},{"y":4,"x":0},{"x":5,"y":6},{"y":6,"x":0},{"y":6,"x":1},{"y":1,"x":0},{"x":3,"y":7},{"x":5,"y":3},{"x":1,"y":7},{"x":7,"y":1},{"y":4,"x":6},{"y":4,"x":7},{"y":0,"x":6},{"y":2,"x":7},{"x":7,"y":6},{"x":5,"y":5},{"y":7,"x":5},{"x":0,"y":2}]}
    # current_board, myColor = initBoard(fullInput)
    x, y = choose_mcts(current_board, myColor)
    print(json.dumps({"response": {"x": int(x), "y": int(y)}}))
