import json
import numpy as np
import random
import time
import sys
import os

from prob_rand_choice import init_board, choice, place
from min_max import choice_min_max
from alpha_beta import choose_alpha_beta
from MCTS import choose_mcts

DIM_BOARD = 8
ROUND = 0
BLACK_FUN = choose_mcts
WHITE_FUN = choice


# print board
def print_board(board, color):
    num_black, num_white = cal_board(board)
    color = 'Black' if color == 1 else 'White'
    print('[Round %d, Next is %s] Black number = %d; White number = %d' % (ROUND, color, num_black, num_white))
    new_board = np.array2string(board)
    new_board = new_board.replace('-1', ' W')
    new_board = new_board.replace('0', '-')
    new_board = new_board.replace('1', 'B')
    print(new_board)


def one_step(full_input, is_print=True):
    board, my_color = init_board(full_input)
    if is_print:
        print_board(board, my_color)
    if my_color == 1:
        x, y = BLACK_FUN(board, my_color)
        requests = full_input['requests'][1:]
        responses = full_input['responses']
    else:
        x, y = WHITE_FUN(board, my_color)
        requests = full_input['requests']
        responses = [{'x': -1, 'y': -1}]
        responses.extend(full_input['responses'])
    response = {"x": x, "y": y}
    responses.append(response)
    new_input = {'requests': responses, 'responses': requests}
    return new_input


def who_win(board):
    num_black, num_white = cal_board(board)
    if num_black > num_white:
        flag = 1
        print('Black Win!')
    elif num_black < num_white:
        flag = -1
        print('White Win!')
    else:
        flag = 0
        print('Draw!')
    return flag


def cal_board(board):
    num_black = num_white = 0
    for i in range(DIM_BOARD):
        for j in range(DIM_BOARD):
            if board[i, j] == 1:
                num_black += 1
            elif board[i, j] == -1:
                num_white += 1
    return num_black, num_white


def test(is_print, time_sleep=1):
    global ROUND
    full_input = {"requests": [{"x": -1, "y": -1}], "responses": []}
    for i in range(int(DIM_BOARD*DIM_BOARD/2)):
        ROUND += 1
        for _ in [-1, 1]:
            if is_print:
                os.system('clear')
            full_input = one_step(full_input, is_print)
            time.sleep(time_sleep)

    fin_board, _ = init_board(full_input)
    return who_win(fin_board)


if __name__ == '__main__':
    test(is_print=True, time_sleep=0.05)
    # res = []
    # os.system('clear')
    # for i in range(50):
    #     res.append(test(is_print=False, time_sleep=0))
    # res = np.array(res)
    # print('Black Win Number: ', len(res[res == 1]))
    # print('White Win Number: ', len(res[res == -1]))
    # print('Draw Number: ', len(res[res == 0]))

