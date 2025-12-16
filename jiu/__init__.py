"""
久棋完整规则实现
兼容 AlphaZero / 自博弈 Agent 调用
"""
from .jiutypes import Player, Point, Skip_eat, Go, Decision, board_size, board_gild
from .jiuboard_fast import Move, Board, GameState
from .scoring import GameResult, compute_game_result
from .utils import print_board, move_str, tuple_move, tuple_point, init_info, go_to_points

__all__ = [
    'Player', 'Point', 'Skip_eat', 'Go', 'Decision', 'board_size', 'board_gild',
    'Move', 'Board', 'GameState',
    'GameResult', 'compute_game_result',
    'print_board', 'move_str', 'tuple_move', 'tuple_point', 'init_info', 'go_to_points'
]
