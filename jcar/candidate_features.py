"""
增强版候选特征模块

在原有基础特征上，添加：
1. 成方检测 - 走法后是否成方，成几个方
2. 跳吃数量 - 跳吃时吃几颗子
3. 安全性评估 - 走后是否会被对方吃
4. 潜在方检测 - 是否形成3子准方
5. 破坏对方 - 是否破坏对方的潜在成方
6. 棋子差距 - 当前局势评估
"""

from typing import Dict, List, Tuple, Optional
from copy import deepcopy
import numpy as np
import torch

from jiu.jiutypes import board_size, Point, Go, Skip_eat
from jiu.jiuboard_fast import (
    GameState, Board, Player, formed_squares_at, count_squares, potential_square_triples,
    find_all_dalians, count_independent_dalians, detect_dalian_by_piece, Dalian
)


ACTS = ['put_piece', 'is_go', 'fly', 'skip_move', 'skip_eat_seq']
ACT2IDX = {a: i for i, a in enumerate(ACTS)}


def _norm_coord(r: int, c: int) -> Tuple[float, float]:
    """归一化坐标到[0,1]"""
    N = float(max(1, board_size - 1))
    return (float(r - 1) / N, float(c - 1) / N)


def _check_will_form_square(board: Board, player: Player, src: Point, dst: Point) -> int:
    """
    检查从src移动到dst后，在dst位置能形成几个方
    返回成方数量 (0-4)
    """
    # 模拟移动
    test_board = deepcopy(board)
    if test_board.get(src) == player:
        test_board.remove_stone(src)
    test_board.play_stone(player, dst)
    
    # 检查dst位置成方数量
    return formed_squares_at(test_board, player, dst)


def _check_is_safe(board: Board, player: Player, dst: Point) -> bool:
    """
    检查走到dst后是否安全（不会被对方立即跳吃）
    返回 True 表示安全
    """
    opp = player.other
    
    # 检查8个方向是否有跳吃威胁
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        # 对方棋子位置（跳过dst的位置）
        opp_r, opp_c = dst.row - dr, dst.col - dc
        # 对方落点位置
        land_r, land_c = dst.row + dr, dst.col + dc
        
        if 1 <= opp_r <= board_size and 1 <= opp_c <= board_size:
            if 1 <= land_r <= board_size and 1 <= land_c <= board_size:
                opp_pt = Point(opp_r, opp_c)
                land_pt = Point(land_r, land_c)
                
                # 检查对方是否有棋子在opp_pt，且land_pt为空
                if board.get(opp_pt) == opp and board.get(land_pt) is None:
                    return False  # 不安全，会被跳吃
    
    return True


def _check_creates_triple(board: Board, player: Player, dst: Point) -> int:
    """
    检查走到dst后是否形成"三子准方"（差1子就能成方）
    返回形成的准方数量 (0-4)
    """
    test_board = deepcopy(board)
    test_board.play_stone(player, dst)
    
    cnt = 0
    for dr in (0, -1):
        for dc in (0, -1):
            a = Point(dst.row + dr, dst.col + dc)
            if 1 <= a.row < board_size and 1 <= a.col < board_size:
                pts = [a, Point(a.row+1, a.col), Point(a.row, a.col+1), Point(a.row+1, a.col+1)]
                owners = [test_board.get(p) for p in pts]
                if owners.count(player) == 3 and owners.count(None) == 1:
                    cnt += 1
    return cnt


def _check_breaks_opponent_potential(board: Board, player: Player, dst: Point) -> int:
    """
    检查走到dst后是否破坏对方的潜在成方（三子准方）
    返回破坏的准方数量
    """
    opp = player.other
    
    # 计算走棋前对方的准方数量
    before = potential_square_triples(board, opp)
    
    # 模拟走棋
    test_board = deepcopy(board)
    test_board.play_stone(player, dst)
    
    # 计算走棋后对方的准方数量
    after = potential_square_triples(test_board, opp)
    
    return max(0, before - after)


# =========================================================
# 褡裢相关特征检测函数
# =========================================================

def _check_creates_dalian(board: Board, player: Player, src: Point, dst: Point) -> bool:
    """
    检查从src移动到dst后是否形成新的褡裢
    褡裢：一个棋子在当前位置能成方，且移动到相邻空位也能成方
    
    Returns:
        True 如果形成了新的褡裢
    """
    # 模拟移动
    test_board = deepcopy(board)
    if test_board.get(src) == player:
        test_board.remove_stone(src)
    test_board.play_stone(player, dst)
    
    # 检查走子后的褡裢数量
    after_dalians = count_independent_dalians(test_board, player)
    
    # 检查走子前的褡裢数量
    before_dalians = count_independent_dalians(board, player)
    
    return after_dalians > before_dalians


def _check_uses_dalian(board: Board, player: Player, src: Point, dst: Point) -> bool:
    """
    检查这步走法是否利用了褡裢结构进行吃子
    即：当前src位置是褡裢的游子，且这步走法形成了方（触发吃子）
    
    Returns:
        True 如果利用褡裢吃子
    """
    # 首先检查src是否是某个褡裢的游子
    dalians = detect_dalian_by_piece(board, player, src)
    if not dalians:
        return False
    
    # 检查移动到dst后是否成方
    test_board = deepcopy(board)
    if test_board.get(src) == player:
        test_board.remove_stone(src)
    test_board.play_stone(player, dst)
    
    formed = formed_squares_at(test_board, player, dst)
    return formed > 0


def _check_breaks_opponent_dalian(board: Board, player: Player, src: Point, dst: Point, 
                                   eaten_points: List[Point] = None) -> bool:
    """
    检查这步走法是否破坏对方的褡裢
    
    Args:
        board: 当前棋盘
        player: 走子方
        src: 起始位置
        dst: 目标位置
        eaten_points: 被吃掉的棋子位置列表（跳吃时）
    
    Returns:
        True 如果破坏了对方的褡裢
    """
    opp = player.other
    
    # 计算走棋前对方的褡裢数量
    before_dalians = count_independent_dalians(board, opp)
    if before_dalians == 0:
        return False  # 对方本来就没有褡裢
    
    # 模拟走棋
    test_board = deepcopy(board)
    if test_board.get(src) == player:
        test_board.remove_stone(src)
    test_board.play_stone(player, dst)
    
    # 如果有吃子，移除被吃的棋子
    if eaten_points:
        for pt in eaten_points:
            if test_board.get(pt) == opp:
                test_board.remove_stone(pt)
    
    # 计算走棋后对方的褡裢数量
    after_dalians = count_independent_dalians(test_board, opp)
    
    return after_dalians < before_dalians


def build_enhanced_features(
    cands: List[Dict], 
    state: GameState,
    phase_id: int, 
    flying: bool
) -> np.ndarray:
    """
    构建增强版候选特征
    
    特征维度 (共32维):
    - [0:5]   act_onehot: 动作类型 one-hot (5维)
    - [5:9]   go_row, go_col, to_row, to_col: 归一化坐标 (4维)
    - [9:11]  d_row, d_col: 归一化位移 (2维)
    - [11]    seq_len_norm: 连跳长度归一化 (1维)
    - [12]    phase_id: 阶段 (1维)
    - [13]    flying: 是否飞子 (1维)
    --- 以上为原有特征 (14维) ---
    - [14]    will_form_square: 是否成方 (0/1)
    - [15]    square_count_norm: 成方数量归一化 (0-1, /4)
    - [16]    eat_count_norm: 跳吃数量归一化 (0-1, /8)
    - [17]    is_safe: 是否安全 (0/1)
    - [18]    creates_triple: 是否形成准方 (0/1)
    - [19]    triple_count_norm: 准方数量归一化 (0-1, /4)
    - [20]    breaks_opp_potential: 是否破坏对方潜在方 (0/1)
    - [21]    breaks_count_norm: 破坏数量归一化 (0-1, /4)
    - [22]    piece_diff_norm: 棋子差距归一化 (-1到1)
    - [23]    my_squares_norm: 我方成方数归一化 (0-1, /10)
    - [24]    opp_squares_norm: 对方成方数归一化 (0-1, /10)
    - [25]    is_capture_move: 是否吃子走法 (0/1)
    --- 以上为增强特征 (12维) ---
    --- 以下为褡裢特征 (6维) ---
    - [26]    my_dalian_count_norm: 我方褡裢数归一化 (0-1, /4)
    - [27]    opp_dalian_count_norm: 对方褡裢数归一化 (0-1, /4)
    - [28]    creates_dalian: 是否形成褡裢 (0/1)
    - [29]    uses_dalian: 是否利用褡裢吃子 (0/1)
    - [30]    breaks_opp_dalian: 是否破坏对方褡裢 (0/1)
    - [31]    creates_pre_dalian: 是否形成准褡裢 (0/1)
    
    Args:
        cands: 候选动作字典列表
        state: 当前游戏状态
        phase_id: 阶段ID (0=布局, 1=对战, 2=飞子)
        flying: 当前玩家是否飞子
    
    Returns:
        特征数组 (N, 32)
    """
    if not cands:
        return np.zeros((0, 32), dtype=np.float32)
    
    board = state.board
    player = state.next_player
    opp = player.other
    
    # 预计算全局信息
    my_count = board.get_player_total(player)
    opp_count = board.get_player_total(opp)
    piece_diff = (my_count - opp_count) / max(1, my_count + opp_count)  # 归一化到[-1,1]
    
    my_squares = count_squares(board, player)
    opp_squares = count_squares(board, opp)
    
    # 预计算褡裢信息（仅在对战/飞子阶段计算，布局阶段为0）
    if phase_id > 0:
        my_dalian_count = count_independent_dalians(board, player)
        opp_dalian_count = count_independent_dalians(board, opp)
    else:
        my_dalian_count = 0
        opp_dalian_count = 0
    
    feats: List[List[float]] = []
    
    for m in cands:
        a = m.get('act', '')
        
        # === 原有基础特征 ===
        onehot = [0.0] * len(ACTS)
        idx = ACT2IDX.get(a, None)
        if idx is not None:
            onehot[idx] = 1.0
        
        go_r = go_c = to_r = to_c = 1
        seq_len = 0
        eat_count = 0
        
        if a == 'put_piece':
            p = m.get('point', {'r': 1, 'c': 1})
            go_r = to_r = int(p['r'])
            go_c = to_c = int(p['c'])
        elif a in ('is_go', 'fly'):
            g = m.get('go', {'r': 1, 'c': 1})
            t = m.get('to', {'r': 1, 'c': 1})
            go_r, go_c = int(g['r']), int(g['c'])
            to_r, to_c = int(t['r']), int(t['c'])
        elif a == 'skip_move':
            g = m.get('go', {'r': 1, 'c': 1})
            t = m.get('to', {'r': 1, 'c': 1})
            go_r, go_c = int(g['r']), int(g['c'])
            to_r, to_c = int(t['r']), int(t['c'])
            eat_count = 1  # 单跳吃1子
        elif a == 'skip_eat_seq':
            seq = m.get('seq', [])
            seq_len = len(seq)
            eat_count = seq_len  # 连跳吃seq_len子
            if seq_len > 0:
                g0 = seq[0]['go']
                tl = seq[-1]['to']
                go_r, go_c = int(g0['r']), int(g0['c'])
                to_r, to_c = int(tl['r']), int(tl['c'])
        
        ngr, ngc = _norm_coord(go_r, go_c)
        ntr, ntc = _norm_coord(to_r, to_c)
        dr = ntr - ngr
        dc = ntc - ngc
        
        # === 新增增强特征 ===
        src_pt = Point(go_r, go_c)
        dst_pt = Point(to_r, to_c)
        
        # 成方检测
        if phase_id == 0:
            # 布局阶段：检查放子后是否成方
            square_count = 0  # 布局阶段暂不计算
            will_form_square = 0.0
        else:
            # 对战/飞子阶段
            square_count = _check_will_form_square(board, player, src_pt, dst_pt)
            will_form_square = 1.0 if square_count > 0 else 0.0
        
        # 安全性检测
        is_safe = 1.0 if _check_is_safe(board, player, dst_pt) else 0.0
        
        # 准方检测（3子差1成方）
        triple_count = _check_creates_triple(board, player, dst_pt)
        creates_triple = 1.0 if triple_count > 0 else 0.0
        
        # 破坏对方潜在方
        breaks_count = _check_breaks_opponent_potential(board, player, dst_pt)
        breaks_opp_potential = 1.0 if breaks_count > 0 else 0.0
        
        # 是否吃子走法
        is_capture = 1.0 if (a in ('skip_move', 'skip_eat_seq') or square_count > 0) else 0.0
        
        # === 褡裢相关特征 ===
        # 收集被吃掉的棋子位置（用于判断是否破坏对方褡裢）
        eaten_points = []
        if a == 'skip_move':
            se = m.get('go_to') or m
            if 'eat' in m:
                eaten_points = [Point(m['eat']['r'], m['eat']['c'])]
        elif a == 'skip_eat_seq':
            seq = m.get('seq', [])
            for step in seq:
                if 'eat' in step:
                    eaten_points.append(Point(step['eat']['r'], step['eat']['c']))
        
        if phase_id > 0:
            # 对战/飞子阶段：计算褡裢特征
            creates_dalian = 1.0 if _check_creates_dalian(board, player, src_pt, dst_pt) else 0.0
            uses_dalian = 1.0 if _check_uses_dalian(board, player, src_pt, dst_pt) else 0.0
            breaks_opp_dalian = 1.0 if _check_breaks_opponent_dalian(board, player, src_pt, dst_pt, eaten_points) else 0.0
            creates_pre_dalian = 0.0
        else:
            # 布局阶段：褡裢相关特征全部为0
            creates_dalian = 0.0
            uses_dalian = 0.0
            breaks_opp_dalian = 0.0
            creates_pre_dalian = 0.0
        
        # 组装特征
        feat = (
            onehot +  # [0:5]
            [ngr, ngc, ntr, ntc] +  # [5:9]
            [dr, dc] +  # [9:11]
            [float(seq_len) / 8.0] +  # [11]
            [float(phase_id)] +  # [12]
            [float(bool(flying))] +  # [13]
            # --- 增强特征 ---
            [will_form_square] +  # [14]
            [float(square_count) / 4.0] +  # [15]
            [float(eat_count) / 8.0] +  # [16]
            [is_safe] +  # [17]
            [creates_triple] +  # [18]
            [float(triple_count) / 4.0] +  # [19]
            [breaks_opp_potential] +  # [20]
            [float(breaks_count) / 4.0] +  # [21]
            [piece_diff] +  # [22]
            [float(my_squares) / 10.0] +  # [23]
            [float(opp_squares) / 10.0] +  # [24]
            [is_capture] +  # [25]
            # --- 褡裢特征 ---
            [float(my_dalian_count) / 4.0] +  # [26]
            [float(opp_dalian_count) / 4.0] +  # [27]
            [creates_dalian] +  # [28]
            [uses_dalian] +  # [29]
            [breaks_opp_dalian] +  # [30]
            [creates_pre_dalian]  # [31]
        )
        
        feats.append(feat)
    
    arr = np.asarray(feats, dtype=np.float32)
    assert arr.shape[1] == 32, f"特征维度错误: 期望32, 实际{arr.shape[1]}"
    return arr


def build_features_for_candidates(
    cands: List[Dict], 
    phase_id: int, 
    flying: bool,
    state: Optional[GameState] = None
) -> np.ndarray:
    """
    兼容接口：根据是否提供state选择使用增强特征或基础特征
    
    如果提供了state，使用增强特征(26维)
    否则使用基础特征(14维) - 向后兼容
    """
    if state is not None:
        return build_enhanced_features(cands, state, phase_id, flying)
    
    # 基础特征（向后兼容）
    feats: List[List[float]] = []
    for m in cands:
        a = m.get('act', '')
        onehot = [0.0] * len(ACTS)
        idx = ACT2IDX.get(a, None)
        if idx is not None:
            onehot[idx] = 1.0
        go_r = go_c = to_r = to_c = 1
        seq_len = 0
        if a == 'put_piece':
            p = m.get('point', {'r': 1, 'c': 1})
            go_r = to_r = int(p['r']); go_c = to_c = int(p['c'])
        elif a in ('is_go', 'fly', 'skip_move'):
            g = m.get('go', {'r': 1, 'c': 1}); t = m.get('to', {'r': 1, 'c': 1})
            go_r = int(g['r']); go_c = int(g['c'])
            to_r = int(t['r']); to_c = int(t['c'])
        elif a == 'skip_eat_seq':
            seq = m.get('seq', [])
            seq_len = len(seq)
            if seq_len > 0:
                g0 = seq[0]['go']; tl = seq[-1]['to']
                go_r = int(g0['r']); go_c = int(g0['c'])
                to_r = int(tl['r']); to_c = int(tl['c'])
        ngr, ngc = _norm_coord(go_r, go_c)
        ntr, ntc = _norm_coord(to_r, to_c)
        dr = ntr - ngr
        dc = ntc - ngc
        feat = onehot + [ngr, ngc, ntr, ntc, dr, dc, float(seq_len) / 8.0, float(phase_id), float(bool(flying))]
        feats.append(feat)
    if not feats:
        return np.zeros((0, 14), dtype=np.float32)
    return np.asarray(feats, dtype=np.float32)


def to_tensor_list(batch_cand_feats: List[np.ndarray], device: torch.device) -> List[torch.Tensor]:
    """转换为tensor列表"""
    ts: List[torch.Tensor] = []
    for arr in batch_cand_feats:
        ts.append(torch.from_numpy(arr).to(device))
    return ts


# === 特征维度常量 ===
BASIC_FEAT_DIM = 14       # 基础特征维度
ENHANCED_FEAT_DIM = 32    # 增强特征维度（包含褡裢特征）
DALIAN_FEAT_START = 26    # 褡裢特征起始索引
DALIAN_FEAT_DIM = 6       # 褡裢特征维度

