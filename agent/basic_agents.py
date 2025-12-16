# agent/basic_agents.py
import random
from .base import Agent
from jiu.jiuboard_fast import Move

class RandomAgent(Agent):
    """随机选择一个合法走法"""
    def select_move(self, game_state):
        candidates = game_state.legal_moves()
        if not candidates:
            return None, []
        dec = random.choice(candidates)
        return self._decision_to_move(dec)

    def _decision_to_move(self, dec):
        from jiu.jiutypes import Decision
        if dec.act == 'put_piece':
            return Move.put_piece(dec.points), []
        if dec.act == 'is_go':
            return Move.go_piece(dec.points), []
        if dec.act == 'skip_move':
            return Move.move_skip(dec.points), []
        if dec.act == 'skip_eat_seq':
            return Move.move_skip_seq(dec.points), []
        if dec.act == 'fly':
            return Move.fly_piece(dec.points), []
        if dec.act == 'eat_point':
            return Move.eat(dec.points), []
        return None, []

class GreedyCaptureAgent(Agent):
    """优先选择吃子数多的走法"""
    def select_move(self, game_state):
        candidates = game_state.legal_moves()
        if not candidates:
            return None, []
        candidates.sort(key=lambda d: d.eats, reverse=True)
        max_eats = candidates[0].eats
        best = [d for d in candidates if d.eats == max_eats]
        dec = random.choice(best)
        return self._decision_to_move(dec)

    def _decision_to_move(self, dec):
        if dec.act == 'put_piece':
            return Move.put_piece(dec.points), []
        if dec.act == 'is_go':
            return Move.go_piece(dec.points), []
        if dec.act == 'skip_move':
            return Move.move_skip(dec.points), []
        if dec.act == 'skip_eat_seq':
            return Move.move_skip_seq(dec.points), []
        if dec.act == 'fly':
            return Move.fly_piece(dec.points), []
        if dec.act == 'eat_point':
            return Move.eat(dec.points), []
        return None, []

class FirstMoveAgent(Agent):
    """总是选择第一个合法走法（调试用）"""
    def select_move(self, game_state):
        candidates = game_state.legal_moves()
        if not candidates:
            return None, []
        dec = candidates[0]
        return self._decision_to_move(dec)

    def _decision_to_move(self, dec):
        if dec.act == 'put_piece':
            return Move.put_piece(dec.points), []
        if dec.act == 'is_go':
            return Move.go_piece(dec.points), []
        if dec.act == 'skip_move':
            return Move.move_skip(dec.points), []
        if dec.act == 'skip_eat_seq':
            return Move.move_skip_seq(dec.points), []
        if dec.act == 'fly':
            return Move.fly_piece(dec.points), []
        if dec.act == 'eat_point':
            return Move.eat(dec.points), []
        return None, []
