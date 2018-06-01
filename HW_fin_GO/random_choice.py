import json
import time
import os
from queue import Queue
from collections import OrderedDict
import numpy as np

komi = 0.75 # komi(贴目), black gives white 0.75 point
WIDTH = 8   # width of the board: 8x8
TOTAL = WIDTH**2 # total size of the board
direct = [-WIDTH, 1, WIDTH, -1] # direction: up, right, down, left
empty = '.'*TOTAL # empty board


class Utils:
    # 处理json输入，将字符串变为json
    @staticmethod
    def json_input():
        raw = input()
        obj = json.loads(raw, object_pairs_hook=OrderedDict)
        return OrderedDict(obj)

    # 处理json输出，将json变为字符串
    @staticmethod
    def json_output(obj):
        raw = json.dumps(obj)
        print(raw)

    # 用来将二维坐标变成一维索引
    @staticmethod
    def xy_to_l(x, y):
        return (y-1) * WIDTH + x - 1

    # 用来将一维索引变为二维坐标
    @staticmethod
    def l_to_xy(l):
        return l % WIDTH + 1, l // WIDTH + 1


def slice_group(board, l, cal_owner=False):
    '''
    Get the group's index which contains l
    Use Queue to get all the related stones
    :param board: A string
    :param l:
    :var the owner: 1 -> black; -1 -> white; 0 -> both.
    :return: group, the owner
    '''
    color = board[l]
    group = [l]
    owner = []

    visit = [0]*TOTAL
    visit[l] = 1
    q = Queue()
    q.put(l)
    while not q.empty():
        here = q.get()
        for d in direct:
            tmp = here + d
            if tmp < 0 or tmp >= TOTAL or visit[tmp] != 0 or board[tmp] != color:
                continue
            if board[tmp] != color:
                owner = owner.append(board[tmp])
                continue
            visit[tmp] = 1
            q.put(tmp)
            group.append(tmp)

    if cal_owner:
        if 'X' in owner:
            if 'O' in owner:
                return group, 0
            else:
                return group, 1
        elif 'O' in owner:
            return group, -1
        else:
            return group, 0
    else:
        return group


def take(board, group):
    '''
    Take away the captured group
    :param board: A string board
    :param group: The group has no liberty
    :return: The new board string
    '''
    copy_board = bytearray(board, 'utf-8')
    for l in group:
        copy_board[l] = ord('.')
    return copy_board.decode('utf-8')


def count_liberty(board, group):
    '''
    Count liberty of a group
    :param board:
    :param group:
    :return:
    '''
    counted = [0] * TOTAL
    count = 0
    for l in group:
        for d in direct:
            tmp = l + d
            if tmp < 0 or tmp >= TOTAL or counted[tmp] != 0:
                continue
            if board[tmp] == '.':
                counted[tmp] = 1
                count += 1
    return count


def settle(board, x, y, just_try=False, color=1):
    '''
    Place the stone on the board
    :param board: A string board
    :param x:
    :param y:
    :param just_try:
    :param color: Black is 1; white is 0
    :var take_num: The number this move take away the opponent's stone
    :var liberty: The liberty of the group contains the move
    :return: is it a possible move, the new board, liberty, take_num
    '''
    my_color = 'X' if color == 1 else 'O'
    op_color = 'X' if 1-color == 1 else 'O'

    linear = Utils.xy_to_l(x, y)     # 用线性坐标
    if linear < 0 or linear >= TOTAL:
        return False, board, -1, 0
    if board[linear] != '.':
        return False, board, -1, 0

    # string cannot be changed. string need to transfer to bytearray
    new_board = bytearray(board, 'utf-8')
    new_board[linear] = ord(my_color)
    new_board = new_board.decode('utf-8')

    take_num = 0
    for d in direct:    # 只会影响到新放置的棋子周围的棋子
        tmp = linear + d
        if tmp < 0 or tmp >= TOTAL:
            continue
        if new_board[tmp] == op_color:   # 周围的对方棋子先受到影响
            affects = slice_group(new_board, tmp)
            liberty = count_liberty(new_board, affects)
            if liberty < 1:     # 对方这团棋子已经是死子了
                new_board = take(new_board, affects)
                take_num += len(affects)

    # 提完对方棋子，再看自己是不是死棋
    new_group = slice_group(new_board, linear)
    liberty = count_liberty(new_board, new_group)
    if liberty < 1:     # 自杀棋步，不合规则
        return False, new_board, liberty, take_num
    elif just_try:
        return True, new_board, liberty, take_num
    else:
        return True, new_board, liberty, take_num


def possible_moves(board, color, history):
    '''
    Get possible moves.
    :param board:
    :param color: Current color
    :param history: list of history board
    :return: moves, liberty of the moves
    '''
    possibles = []
    liberties = []
    for x in range(1, WIDTH + 1):
        for y in range(1, WIDTH + 1):
            good_move, next_board, liberty, _ = settle(board, x, y, True, color)
            if good_move & (next_board not in history):
                possibles.append([x, y])
                liberties.append(liberty)
    return possibles, liberties


def cal_board(board):
    '''
    Calculate the board
    :param board:
    :return: black score, white score
    '''
    black_score, white_score = cal_board_num(board)
    blank = set([x for x in range(TOTAL) if board[x] == '.'])

    while len(blank) > 0:
        group, owner = slice_group(board, list(blank)[0])
        blank = blank - set(group)
        if owner == 1:
            black_score += len(group)
        elif owner == -1:
            white_score += len(group)
        else:
            black_score += len(group)/2
            white_score += len(group)/2

    return black_score, white_score


def cal_board_num(board):
    black_num, white_num = 0, 0
    for i in range(TOTAL):
        black_num += board[i] == 'X'
        white_num += board[i] == 'O'
    return black_num, white_num


def recover_from_input(requests, responses):
    '''
    Recover the board
    :param requests:
    :param responses:
    :var bot_color:  Black is 1; white is 0
    :return:
    '''
    bot_color = 0
    history_boards = []

    # 如果第一回合收到的是-2的request，说明我方是执黑先行
    if not (requests[0]["x"] == -2 and requests[0]["y"] == -2):
        bot_color = 1

    turn_num = len(responses)
    count_for_pass = 0      # 记录当前连续“虚着”次数，一旦达到两次，棋局终了
    board = empty
    for i in range(turn_num):
        # 对方的棋步（也可能是第一回合的-2）
        x, y = requests[i]["x"], requests[i]["y"]
        if x > 0 and y > 0:
            good_move, board, _ = settle(board, x, y, color=1-bot_color)
            history_boards.append(board)

        # 我方的棋步
        x, y = responses[i]["x"], responses[i]["y"]
        if x > 0 and y > 0:
            good_move, board, _ = settle(board, x, y, color=bot_color)
            history_boards.append(board)

    # 对方刚刚落的子
    x, y = requests[turn_num]["x"], requests[turn_num]["y"]
    if x == -1 and y == -1:
        count_for_pass += 1
    elif x > 0 and y > 0:
        good_move, board, _ = settle(board, x, y, color=1-bot_color)
        history_boards.append(board)

    observe = \
        {"board": board, "count_for_pass": count_for_pass, "my_color": bot_color, "history": history_boards}

    return observe


def random_choose(board, color, history):
    moves, liberties = possible_moves(board, color, history)
    moves = [moves[i] for i in range(len(liberties)) if liberties[i] > 1]
    if len(moves) == 0:
        return -1, -1
    else:
        ind = np.random.randint(0, len(moves))
        return moves[ind][0], moves[ind][1]

