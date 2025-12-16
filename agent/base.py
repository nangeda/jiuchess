import os, sys
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CUR_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from jiu.jiutypes import board_size

__all__ = [
    'Agents',
    'Agent',
]

class Agents:
    """智能体基类（旧版本兼容）"""
    def __init__(self):
        pass

    def select_move(self, game_state):
        raise NotImplementedError()

    def diagnostics(self):
        return {}


class Agent:
    """智能体基类"""
    def __init__(self):
        self.board_size = board_size
        self.board_grid = board_size * board_size

    def select_move(self, game_state):
        raise NotImplementedError()

    def diagnostics(self):
        return {}
