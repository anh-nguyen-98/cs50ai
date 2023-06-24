"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    num_occupied = 0
    for row in board:
        for cell in row:
            num_occupied += 1 if cell != EMPTY else 0
    return X if num_occupied % 2 == 0 else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = []
    for r in range(3):
        for c in range(3):
            if board[r][c] == EMPTY:
                actions.append((r,c))
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if board[action[0]][action[1]] != EMPTY:
        raise Exception("Invalid action")
    new_board = copy.deepcopy(board)
    new_board[action[0]][action[1]] = player(board)
    return new_board
    # raise NotImplementedError


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # check rows
    for row in board:
        if row[0] == row[1] == row[2] != EMPTY:
            return row[0]
    
    # check columns
    for i in range(3):
        if board[0][i] == board[1][i] == board[2][i] != EMPTY:
            return board[0][i]
    
    # check diagonals
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]

    return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    return winner(board) is not None or actions(board) == []
   
def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    game_winner = winner(board)
    return 1 if game_winner == X else -1 if game_winner == O else 0

def max_value(board):
    if terminal(board):
        return [None, utility(board)]
    best_value = -math.inf
    best_action = None
    for action in actions(board):
        _, current_value = min_value(result(board, action))
        if best_value < current_value:
            best_value = current_value
            best_action = action
    return [best_action, best_value]

def min_value(board):
    if terminal(board):
        return [None, utility(board)]
    best_value = math.inf
    best_action = None
    for action in actions(board):
        _, current_value = max_value(result(board, action))
        if best_value > current_value:
            best_value = current_value
            best_action = action
    return [best_action, best_value]

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    # which player is playing
    current_player = player(board)
    # if player is X, then we want to maximize the score
    if current_player == X:
        optimal_move, _ = max_value(board)
        return optimal_move
    # if player is 0, then we want to minimize the score
    if current_player == O:
        optimal_move, _ = min_value(board)
        return optimal_move
    raise Exception("Invalid player")
