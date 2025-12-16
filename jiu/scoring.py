from collections import namedtuple
from .jiutypes import Player, Point

class GameResult(namedtuple('GameResult', 'b w')):
    @property
    def winner(self):
        if self.b > self.w:
            return Player.black
        elif self.w > self.b:
            return Player.white
        return None

def compute_game_result(game_state):
    b = game_state.board.get_player_total(Player.black)
    w = game_state.board.get_player_total(Player.white)
    return GameResult(b, w)
