# agent/dyt_agent.py
"""使用训练好的DyT模型的AI Agent"""
import torch
import numpy as np
from typing import Optional, Tuple
from .base import Agent
from jiu.jiuboard_fast import Move, GameState
from jiu.jiutypes import Player, Decision
from dyt.dyt_model import RegisterDyT, DyTNetConfig
from dyt.candidate_features import build_features_for_candidates


def encode_board_state(state: GameState, history: list = None) -> np.ndarray:
    """
    将GameState编码为 (6, H, W) 的观察张量（与训练数据一致）
    通道0: 白子
    通道1: 黑子
    通道2: 空位
    通道3-5: 历史 1-3 步的己方子（当前行棋方）

    Args:
        state: 当前游戏状态
        history: 历史棋盘状态列表（最近的在后面）
    """
    from jiu.jiutypes import Point, Player, board_size

    board = state.board
    H, W = board_size, board_size
    obs = np.zeros((6, H, W), dtype=np.float32)

    # 通道0-2: 当前局面（白子、黑子、空位）
    for r in range(1, H + 1):
        for c in range(1, W + 1):
            pt = Point(r, c)
            pl = board.get(pt)
            if pl == Player.white:
                obs[0, r - 1, c - 1] = 1.0
            elif pl == Player.black:
                obs[1, r - 1, c - 1] = 1.0
            else:
                obs[2, r - 1, c - 1] = 1.0

    # 通道3-5: 历史帧（最近 3 步的己方棋子）
    if history is not None and len(history) > 0:
        me = state.next_player
        hist_list = history[-3:]  # 最近3步
        for i, past_board in enumerate(hist_list):
            for r in range(1, H + 1):
                for c in range(1, W + 1):
                    pt = Point(r, c)
                    if past_board.get(pt) == me:
                        obs[3 + i, r - 1, c - 1] = 1.0

    return obs


def get_phase_id(state: GameState) -> int:
    """获取当前阶段ID: 0=布局, 1=对战, 2=飞子"""
    from jiu.jiutypes import board_gild
    
    if state.step < board_gild:
        return 0  # 布局阶段
    elif hasattr(state, '_is_flying_stage') and state._is_flying_stage():
        return 2  # 飞子阶段
    else:
        return 1  # 正常对战阶段


def decision_to_dict(dec: Decision) -> dict:
    """将Decision转换为字典，用于特征提取"""
    from jiu.jiutypes import Point, Go, Skip_eat
    
    if dec.act == 'put_piece':
        # points是一个Point对象
        p = dec.points if isinstance(dec.points, Point) else Point(1, 1)
        return {'act': 'put_piece', 'point': {'r': p.row, 'c': p.col}}
    
    elif dec.act == 'is_go':
        # points是一个Go对象，有go和to属性
        go_obj = dec.points if isinstance(dec.points, Go) else Go(Point(1, 1), Point(1, 1))
        return {'act': 'is_go', 'go': {'r': go_obj.go.row, 'c': go_obj.go.col}, 
                'to': {'r': go_obj.to.row, 'c': go_obj.to.col}}
    
    elif dec.act == 'fly':
        # points是一个Go对象
        go_obj = dec.points if isinstance(dec.points, Go) else Go(Point(1, 1), Point(1, 1))
        return {'act': 'fly', 'go': {'r': go_obj.go.row, 'c': go_obj.go.col}, 
                'to': {'r': go_obj.to.row, 'c': go_obj.to.col}}
    
    elif dec.act == 'skip_move':
        # points是一个Skip_eat对象
        se = dec.points if isinstance(dec.points, Skip_eat) else Skip_eat(Point(1, 1), Point(1, 1), Point(1, 1))
        return {'act': 'skip_move', 'go': {'r': se.go.row, 'c': se.go.col}, 
                'to': {'r': se.to.row, 'c': se.to.col},
                'eat': {'r': se.eat.row, 'c': se.eat.col}}
    
    elif dec.act == 'skip_eat_seq':
        # points是一个Skip_eat对象列表
        seq = []
        if isinstance(dec.points, list):
            for se in dec.points:
                if isinstance(se, Skip_eat):
                    seq.append({'go': {'r': se.go.row, 'c': se.go.col}, 
                               'to': {'r': se.to.row, 'c': se.to.col},
                               'eat': {'r': se.eat.row, 'c': se.eat.col}})
        return {'act': 'skip_eat_seq', 'seq': seq}
    
    elif dec.act == 'eat_point':
        # 可能有eat_point动作
        p = dec.points if isinstance(dec.points, Point) else Point(1, 1)
        return {'act': 'eat_point', 'point': {'r': p.row, 'c': p.col}}
    
    return {'act': 'unknown'}


class DyTAgent(Agent):
    """基于DyT神经网络的AI Agent"""

    def __init__(self, model_path: str, device: str = 'cuda', temperature: float = 0.3):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature

        # 历史状态维护（最近3步的棋盘状态）
        self.history = []

        # 加载模型
        cfg = DyTNetConfig()
        self.model = RegisterDyT(cfg).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print(f"DyT model loaded from {model_path} on {self.device}")
    
    def select_move(self, game_state: GameState) -> Tuple[Optional[Move], list]:
        """选择最佳走法"""
        candidates = game_state.legal_moves()
        if not candidates:
            return None, []

        if len(candidates) == 1:
            move, _ = self._decision_to_move(candidates[0])
            # 更新历史
            self._update_history(game_state)
            return move, []

        # 编码当前状态（包含历史）
        obs = encode_board_state(game_state, self.history)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)  # (1, 6, H, W)

        phase_id = get_phase_id(game_state)
        phase_tensor = torch.tensor([phase_id], dtype=torch.long, device=self.device)

        # 判断是否飞子阶段
        flying = (hasattr(game_state, '_is_flying_stage') and game_state._is_flying_stage())

        # 构建候选特征 - 转换Decision为字典列表
        cand_dicts = [decision_to_dict(dec) for dec in candidates]
        cand_features = build_features_for_candidates(cand_dicts, phase_id, flying)  # (N, 14)
        cand_tensor = torch.from_numpy(cand_features).to(self.device)

        # 模型推理
        with torch.no_grad():
            logits_list, value = self.model.score_candidates(obs_tensor, phase_tensor, [cand_tensor])
            logits = logits_list[0]  # (N,)

            # 应用温度采样
            if self.temperature > 0:
                probs = torch.softmax(logits / self.temperature, dim=0)
                idx = torch.multinomial(probs, 1).item()
            else:
                idx = torch.argmax(logits).item()

        selected_dec = candidates[idx]
        move, _ = self._decision_to_move(selected_dec)

        # 更新历史
        self._update_history(game_state)

        return move, []

    def _update_history(self, game_state: GameState):
        """更新历史棋盘状态（保留最近3步）"""
        from copy import deepcopy
        self.history.append(deepcopy(game_state.board))
        if len(self.history) > 3:
            self.history.pop(0)
    
    def _decision_to_move(self, dec: Decision) -> Tuple[Move, list]:
        """将Decision转换为Move"""
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

