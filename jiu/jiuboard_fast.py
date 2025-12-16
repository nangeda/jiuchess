import copy
import numpy as np
from copy import deepcopy
from collections import deque
from .jiutypes import Player, Point, Decision, Skip_eat, Go, board_size, board_gild

try:
    from .utils import print_board
except Exception:
    print_board = None

__all__ = ['Move','Board','GameState']

# =========================================================
# Move
# =========================================================
class Move:
    def __init__(self, point=None, skip_eat_points=None, go_to=None,
                 eat_point=False, is_put=False, is_go=False,
                 is_skip_eat=False, is_skip_eat_seq=False,
                 is_fly=False, is_resign=False):
        assert (point is not None) ^ is_resign or is_go or is_skip_eat or is_skip_eat_seq or is_fly or is_put or eat_point
        self.point = point
        self.eat_point = eat_point

        self.go_to = go_to
        self.is_go = is_go
        self.is_fly = is_fly

        self.skip_eat_points = skip_eat_points
        self.is_skip_eat = is_skip_eat
        self.is_skip_eat_seq = is_skip_eat_seq

        self.is_put = is_put
        self.is_resign = is_resign

    @classmethod
    def eat(cls,point):                  return Move(point=point,eat_point=True)
    @classmethod
    def put_piece(cls,point):            return Move(point=point,is_put=True)
    @classmethod
    def go_piece(cls,go_to):             return Move(go_to=go_to,is_go=True,point=go_to.go)
    @classmethod
    def move_skip(cls,skip_eat_points):  return Move(skip_eat_points=skip_eat_points,is_skip_eat=True,point=skip_eat_points.go)
    @classmethod
    def move_skip_seq(cls,seq):          return Move(skip_eat_points=seq,is_skip_eat_seq=True,point=seq[0].go)
    @classmethod
    def fly_piece(cls,go_to):            return Move(point=go_to.go,is_fly=True,go_to=go_to)
    @classmethod
    def resign_game(cls):                return Move(is_resign=True)

# =========================================================
# Board（稀疏网格 + 三张 numpy 盘面）
# =========================================================
class Board:
    def __init__(self, num_rows=None, num_cols=None):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid = {}  # {Point: Player}
        sz = (num_rows+2, num_cols+2)
        self.board_white = np.zeros(sz, dtype=np.int8)
        self.board_black = np.zeros(sz, dtype=np.int8)
        self.board_total = np.zeros(sz, dtype=np.int8)

    def is_on_grid(self, p: Point):
        return 1 <= p.row <= self.num_rows and 1 <= p.col <= self.num_cols

    def get(self, p: Point):
        return self._grid.get(p, None)

    def stones(self, who: Player):
        return [pt for pt, pl in self._grid.items() if pl == who]

    def empties(self):
        res = []
        for r in range(1, self.num_rows+1):
            for c in range(1, self.num_cols+1):
                p = Point(r,c)
                if p not in self._grid:
                    res.append(p)
        return res

    def get_player_total(self, player):
        return sum(1 for pt,pl in self._grid.items() if pl==player)

    def play_stone(self, player, p: Point):
        assert self.is_on_grid(p) and self._grid.get(p) is None
        self._grid[p] = player
        if player == Player.white:
            self.board_white[p.row][p.col] = 1
            self.board_total[p.row][p.col] = +1
        else:
            self.board_black[p.row][p.col] = 1
            self.board_total[p.row][p.col] = -1

    def remove_stone(self, p: Point):
        assert self._grid.get(p) is not None
        pl = self._grid.pop(p)
        if pl == Player.white:
            self.board_white[p.row][p.col] = 0
        else:
            self.board_black[p.row][p.col] = 0
        self.board_total[p.row][p.col] = 0

    def move_stone(self, player, src: Point, dst: Point):
        assert self.get(src) == player and self.get(dst) is None
        self.remove_stone(src)
        self.play_stone(player, dst)

    def move_stone_reversible(self, player, src: Point, dst: Point):
        """可逆的移动棋子，返回撤销信息"""
        assert self.get(src) == player and self.get(dst) is None
        self.remove_stone(src)
        self.play_stone(player, dst)
        return ('move', player, src, dst)

    def remove_stone_reversible(self, p: Point):
        """可逆的移除棋子，返回撤销信息"""
        player = self._grid.get(p)
        assert player is not None
        self.remove_stone(p)
        return ('remove', player, p)

    def undo_operation(self, undo_info):
        """撤销操作"""
        op_type = undo_info[0]
        if op_type == 'move':
            _, player, src, dst = undo_info
            self.remove_stone(dst)
            self.play_stone(player, src)
        elif op_type == 'remove':
            _, player, p = undo_info
            self.play_stone(player, p)

# =========================================================
# 规则工具（成方、潜在方、连跳等）
# =========================================================
def formed_squares_at(board: Board, who: Player, at: Point) -> int:
    cnt = 0
    for dr in (0,-1):
        for dc in (0,-1):
            a = Point(at.row+dr, at.col+dc)
            if 1 <= a.row < board.num_rows and 1 <= a.col < board.num_cols:
                pts = [a, Point(a.row+1,a.col), Point(a.row,a.col+1), Point(a.row+1,a.col+1)]
                if all(board.get(p)==who for p in pts):
                    cnt += 1
    return cnt

def count_squares(board: Board, who: Player) -> int:
    cnt = 0
    for r in range(1, board.num_rows):
        for c in range(1, board.num_cols):
            pts = [Point(r,c), Point(r+1,c), Point(r,c+1), Point(r+1,c+1)]
            if all(board.get(p) == who for p in pts):
                cnt += 1
    return cnt

def potential_square_triples(board: Board, who: Player) -> int:
    cnt = 0
    for r in range(1, board.num_rows):
        for c in range(1, board.num_cols):
            pts = [Point(r,c), Point(r+1,c), Point(r,c+1), Point(r+1,c+1)]
            owners = [board.get(p) for p in pts]
            if owners.count(who)==3 and owners.count(None)==1:
                cnt += 1
    return cnt

def single_jumps_from(board: Board, who: Player, src: Point):
    res = []
    for to in src.neighbors_eight_far2():
        mid = Point((src.row+to.row)//2, (src.col+to.col)//2)
        if board.is_on_grid(mid) and board.get(mid)==who.other and board.get(to) is None:
            res.append(Skip_eat(src, mid, to))
    return res

def jump_sequences(board: Board, who: Player, start: Point):
    sequences = []
    def dfs(cur_board: Board, cur: Point, path, visited):
        steps = single_jumps_from(cur_board, who, cur)
        if not steps:
            if path:
                sequences.append(path[:])
            return
        for se in steps:
            if se.to in visited: 
                continue
            nb = copy.deepcopy(cur_board)
            nb.move_stone(who, se.go, se.to)
            nb.remove_stone(se.eat)
            visited.add(se.to)
            path.append(se)
            dfs(nb, se.to, path, visited)
            path.pop()
            visited.remove(se.to)
    dfs(board, start, [], {start})
    return sequences

def choose_removal_targets_for_squares(board: Board, who: Player, formed_cnt: int):
    opp = who.other
    opp_stones = sorted(board.stones(opp), key=lambda p:(p.row,p.col))
    def in_opp_square(pt: Point) -> bool:
        for dr in (0,-1):
            for dc in (0,-1):
                a = Point(pt.row+dr, pt.col+dc)
                if 1 <= a.row < board.num_rows and 1 <= a.col < board.num_cols:
                    ps = [a, Point(a.row+1,a.col), Point(a.row,a.col+1), Point(a.row+1,a.col+1)]
                    if all(board.get(p)==opp for p in ps):
                        return True
        return False
    non_square = [p for p in opp_stones if not in_opp_square(p)]
    square     = [p for p in opp_stones if in_opp_square(p)]
    seq = []
    for bag in (non_square, square):
        for p in bag:
            if len(seq) < formed_cnt:
                seq.append(p)
    return seq


# =========================================================
# 褡裢检测（Dalian Detection）
# =========================================================
class Dalian:
    """褡裢结构，记录组成该褡裢的所有棋子位置"""
    def __init__(self, pieces: set, trigger_piece: Point, empty_pos: Point = None):
        self.pieces = pieces     # 组成褡裢的所有棋子位置集合
        self.trigger = trigger_piece  # 游子位置
        self.empty = empty_pos   # 空位位置（可能没有）
        
    def __repr__(self):
        return f"Dalian(pieces={len(self.pieces)}, trigger={self.trigger}, empty={self.empty})"


def detect_dalian_by_piece(board: Board, player: Player, piece: Point) -> list:
    """
    检测某个棋子是否构成褡裢
    褡裢定义（根据规则）：一个棋子在当前位置能成方，并且向周围移动一格也能成方
    这样的7粒子（游子+周围6个相关棋子）构成一个褡裢
    
    具体检测：
    1. 检查这个棋子当前位置是否在某个方中
    2. 检查移动到相邻某个空位后，是否也能形成方
    3. 如果满足，记录所有相关的棋子
    """
    if board.get(piece) != player:
        return []
    
    dalians = []
    
    # 检查这个棋子当前是否在某个方中
    current_squares = []
    for dr in (0, -1):
        for dc in (0, -1):
            a = Point(piece.row + dr, piece.col + dc)
            if 1 <= a.row < board.num_rows and 1 <= a.col < board.num_cols:
                pts = [a, Point(a.row+1, a.col), Point(a.row, a.col+1), Point(a.row+1, a.col+1)]
                if all(board.get(p) == player for p in pts):
                    current_squares.append(pts)
    
    # 如果当前不在任何方中，不是褡裢
    if not current_squares:
        return dalians
    
    # 尝试移动到每个相邻的空位，看是否也能成方
    # 注：只检查四向相邻，与走子规则一致
    for neighbor in piece.neighbors_four():
        if not board.is_on_grid(neighbor):
            continue
        if board.get(neighbor) is not None:
            continue
        
        # 模拟移动到这个位置，检查是否能成方
        move_squares = []
        for dr in (0, -1):
            for dc in (0, -1):
                a = Point(neighbor.row + dr, neighbor.col + dc)
                if 1 <= a.row < board.num_rows and 1 <= a.col < board.num_cols:
                    pts = [a, Point(a.row+1, a.col), Point(a.row, a.col+1), Point(a.row+1, a.col+1)]
                    # 检查这4个点：piece位置算空，neighbor位置算有己方棋子
                    valid = True
                    for p in pts:
                        if p == piece:
                            valid = False
                            break
                        elif p == neighbor:
                            continue  # 这是要移动到的位置
                        elif board.get(p) != player:
                            valid = False
                            break
                    if valid:
                        move_squares.append(pts)
        
        # 如果移动后也能成方，这就是一个褡裢
        if move_squares:
            # 收集所有相关的棋子（7粒子）
            all_pieces = set()
            all_pieces.add(piece)  # 游子本身
            
            # 从当前的方中收集棋子
            for square in current_squares:
                for p in square:
                    if board.get(p) == player:
                        all_pieces.add(p)
            
            # 从移动后的方中收集棋子（除了neighbor，因为它现在是空的）
            for square in move_squares:
                for p in square:
                    if board.get(p) == player:
                        all_pieces.add(p)
            
            # 创建褡裢对象（不再区分单/双褡裢）
            dalians.append(Dalian(all_pieces, piece, neighbor))
    
    return dalians


def find_all_dalians(board: Board, player: Player) -> list:
    """查找指定玩家的所有褡裢（单褡裢和双褡裢）"""
    dalians = []
    
    # 遍历所有己方棋子，检测每个棋子是否构成褡裢
    my_stones = board.stones(player)
    seen_dalians = set()  # 用于去重
    
    for stone in my_stones:
        piece_dalians = detect_dalian_by_piece(board, player, stone)
        for dalian in piece_dalians:
            # 使用frozenset作为key去重（因为相同的褡裢可能从不同方向检测到）
            key = (frozenset(dalian.pieces), dalian.trigger, dalian.empty)
            if key not in seen_dalians:
                seen_dalians.add(key)
                dalians.append(dalian)
    
    return dalians


def compute_maximum_independent_dalians(dalians: list) -> int:
    """
    计算褡裢的最大独立集
    如果两个褡裢共享棋子，则它们不是独立的
    返回最多能有多少个独立的褡裢
    """
    if not dalians:
        return 0
    
    n = len(dalians)
    if n == 1:
        return 1
    
    # 构建冲突图：如果两个褡裢共享棋子，则有边
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if dalians[i].pieces & dalians[j].pieces:  # 有共同棋子
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # 使用回溯法求最大独立集
    max_count = [0]
    
    def backtrack(index, selected, used_pieces):
        if index == n:
            max_count[0] = max(max_count[0], len(selected))
            return
        
        # 剪枝：即使选择所有剩余的也无法超过当前最大值
        if len(selected) + (n - index) <= max_count[0]:
            return
        
        # 尝试不选择当前褡裢
        backtrack(index + 1, selected, used_pieces)
        
        # 尝试选择当前褡裢（如果不与已选择的冲突）
        if not any(j in selected for j in conflicts[index]):
            backtrack(index + 1, selected | {index}, used_pieces | dalians[index].pieces)
    
    backtrack(0, set(), set())
    return max_count[0]


def count_independent_dalians(board: Board, player: Player) -> int:
    """计算指定玩家的独立褡裢数量"""
    dalians = find_all_dalians(board, player)
    return compute_maximum_independent_dalians(dalians)


# =========================================================
# GameState（完整规则）
# =========================================================
class GameState:
    def __init__(self, board: Board, next_player: Player, previous, move, step):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        self.last_move = move
        self.step = step
        self.black_edge_history = deque([], maxlen=12)  # 规则 16
        self._cached_legal_moves = None  # 缓存合法走法

    # ---------- 起局 ----------
    @classmethod
    def new_game(cls, board_size_in: int):
        n = int(board_size_in)
        board = Board(n, n)
        return GameState(board, Player.white, None, None, 0)

    @classmethod
    def change_player(cls, game):
        return GameState(game.board, game.next_player.other, game.previous_state, game.last_move, game.step)

    # ---------- 阶段辅助 ----------
    def _center_points(self):
        c1 = Point(self.board.num_rows//2 + 1, self.board.num_cols//2 + 1)
        c2 = Point(self.board.num_rows//2,     self.board.num_cols//2)
        return (c1, c2)

    def _exchange_boxes_remove_centers(self, board_after_full: Board):
        c1, c2 = self._center_points()
        if board_after_full.get(c1) is not None: board_after_full.remove_stone(c1)
        if board_after_full.get(c2) is not None: board_after_full.remove_stone(c2)

    def _is_flying_stage(self):
        return (self.board.get_player_total(Player.white) <= self.board.num_rows) or \
               (self.board.get_player_total(Player.black) <= self.board.num_rows)

    # ---------- 合法着法 ----------
    def _legal_put_moves(self):
        if self.step == 0:
            # 支持两种开局规则：(8,8) 中心开局 或 (7,7) 开局
            c1, c2 = self._center_points()  # c1=(8,8), c2=(7,7)
            candidates = []
            if self.board.get(c1) is None:
                candidates.append(Decision('put_piece', c1, 0))
            if self.board.get(c2) is None:
                candidates.append(Decision('put_piece', c2, 0))
            return candidates
        elif self.step == 1:
            # 第二步只能下另一个中心点
            c1, c2 = self._center_points()
            pt = c1 if self.board.get(c1) is None else c2
            return [Decision('put_piece', pt, 0)] if self.board.get(pt) is None else []
        else:
            return [Decision('put_piece', p, 0) for p in self.board.empties()]

    def _generate_single_jumps(self, board: Board, player: Player, src: Point):
        """生成单次跳吃走法（只能直线跳，不能斜着跳）"""
        res = []
        # 只能直线方向跳吃（上下左右），不能斜着跳
        for to in src.neighbors_four_far2():
            mid = Point((src.row+to.row)//2, (src.col+to.col)//2)
            if board.is_on_grid(to) and board.is_on_grid(mid) and \
               board.get(to) is None and board.get(mid) == player.other:
                res.append(Skip_eat(src, mid, to))
        return res

    def _dfs_jump_sequences(self, board: Board, player: Player, start: Point):
        """优化版：使用撤销操作代替深拷贝"""
        sequences = []
        seen = set()

        def dfs(cur: Point, path, visited):
            if path:
                key = tuple(path)
                if key not in seen:
                    seen.add(key)
                    sequences.append(path[:])
            steps = self._generate_single_jumps(board, player, cur)
            if not steps:
                return
            for se in steps:
                if se.to in visited:
                    continue
                # 使用可逆操作代替深拷贝
                undo_move = board.move_stone_reversible(player, se.go, se.to)
                undo_eat = board.remove_stone_reversible(se.eat)

                visited.add(se.to)
                path.append(se)
                dfs(se.to, path, visited)
                path.pop()
                visited.remove(se.to)

                # 撤销操作
                board.undo_operation(undo_eat)
                board.undo_operation(undo_move)

        dfs(start, [], {start})
        return sequences

    def _filter_black_line_repetition(self, candidates):
        if not self.previous_state:
            return candidates
        edges = []
        st = self
        k = 0
        while st and k < 12:
            mv = st.last_move
            if mv and (mv.is_go or mv.is_fly or mv.is_skip_eat or mv.is_skip_eat_seq):
                if st.next_player == Player.white:  # 这一步是黑走
                    # 修复：根据单跳/连跳分别取起讫点
                    if mv.is_go or mv.is_fly:
                        u, v = mv.go_to.go, mv.go_to.to
                    elif mv.is_skip_eat_seq:
                        u, v = mv.skip_eat_points[0].go, mv.skip_eat_points[-1].to
                    else:  # is_skip_eat（单跳）
                        u, v = mv.skip_eat_points.go, mv.skip_eat_points.to
                    edges.append((u,v))
                    k += 1
            st = st.previous_state

        def is_back_and_forth_three_times(edges_list):
            if len(edges_list) < 6:
                return None
            tail = list(reversed(edges_list))[:6]
            A1,B1 = tail[0]; A2,B2 = tail[1]; A3,B3 = tail[2]
            A4,B4 = tail[3]; A5,B5 = tail[4]; A6,B6 = tail[5]
            cond = (A1,B1)==(A3,B3)==(A5,B5) and (A2,B2)==(A4,B4)==(A6,B6) and (A1,B1)==(B2,A2)
            if cond: return (A1,B1)
            return None

        ban_edge = is_back_and_forth_three_times(edges)
        if not ban_edge:
            return candidates

        A,B = ban_edge
        filtered = []
        for d in candidates:
            if d.act in ('is_go','fly'):
                u,v = d.points.go, d.points.to
            elif d.act=='skip_move':      # 单跳
                u,v = d.points.go, d.points.to
            elif d.act=='skip_eat_seq':   # 连跳
                u,v = d.points[0].go, d.points[-1].to
            else:
                filtered.append(d); continue
            if (u,v)==(A,B) or (u,v)==(B,A):
                continue
            filtered.append(d)
        return filtered

    def _legal_playing_moves(self):
        me = self.next_player
        res = []
        min_jump = 2 if self._is_flying_stage() else 1

        # 走子/飞子
        my_stones = self.board.stones(me)
        can_fly = (self.board.get_player_total(me) <= self.board.num_rows)
        for src in my_stones:
            if can_fly:
                for dst in self.board.empties():
                    if dst != src:
                        res.append(Decision('fly', Go(src,dst), 0))
            else:
                # 走子只能四向相邻（上下左右），不能斜向
                for nb in src.neighbors_four():
                    if self.board.get(nb) is None:
                        res.append(Decision('is_go', Go(src,nb), 0))

        # 跳吃/连跳
        jump_exist = False
        for src in my_stones:
            seqs = self._dfs_jump_sequences(self.board, me, src)
            for seq in seqs:
                if len(seq) >= min_jump:
                    jump_exist = True
                    if len(seq)==1:
                        res.append(Decision('skip_move', seq[0], 0))
                    else:
                        res.append(Decision('skip_eat_seq', seq, 0))

        # 若存在跳吃，通常只允许跳吃（避免弱走）
        if jump_exist:
            pass

        # 规则 16：黑方往返三次限制
        if me == Player.black:
            res = self._filter_black_line_repetition(res)

        return res

    def legal_moves(self):
        # 使用缓存避免重复计算
        if self._cached_legal_moves is not None:
            return self._cached_legal_moves

        if self.step < 2:
            result = self._legal_put_moves()
        elif self.step < board_gild:
            result = self._legal_put_moves()
        else:
            result = self._legal_playing_moves()

        self._cached_legal_moves = result
        return result

    # ---------- 布局→吃子阶段转换 ----------
    def _finish_placement_then_remove_centers(self, next_state):
        self._exchange_boxes_remove_centers(next_state.board)
        next_state.next_player = Player.black  # 规则 7：吃子阶段黑先

    # ---------- 成方吃子 ----------
    def _finish_square_captures(self, next_board: Board, me: Player, landing: Point):
        formed = formed_squares_at(next_board, me, landing)
        if formed > 0:
            targets = choose_removal_targets_for_squares(next_board, me, formed)
            for t in targets:
                if next_board.get(t) == me.other:
                    next_board.remove_stone(t)

    # ---------- 执行 ----------
    def apply_move(self, move: Move, fang_eats=None):
        if move.is_resign:
            return GameState(deepcopy(self.board), self.next_player.other, self, move, self.step+1)

        # 布局阶段
        if self.step < board_gild:
            assert move.is_put
            nb = deepcopy(self.board)
            nb.play_stone(self.next_player, move.point)
            ns = GameState(nb, self.next_player.other, self, move, self.step+1)
            if ns.step == board_gild:
                self._finish_placement_then_remove_centers(ns)
            return ns

        # 吃子阶段
        me = self.next_player
        nb = deepcopy(self.board)

        if move.is_go:
            src, dst = move.go_to.go, move.go_to.to
            nb.move_stone(me, src, dst)
            self._finish_square_captures(nb, me, dst)

        elif move.is_fly:
            src, dst = move.go_to.go, move.go_to.to
            nb.move_stone(me, src, dst)
            self._finish_square_captures(nb, me, dst)

        elif move.is_skip_eat or move.is_skip_eat_seq:
            seq = move.skip_eat_points if move.is_skip_eat_seq else [move.skip_eat_points]
            if self._is_flying_stage():
                assert len(seq) >= 2
            for step in seq:
                nb.move_stone(me, step.go, step.to)
                nb.remove_stone(step.eat)
            dst = seq[-1].to
            self._finish_square_captures(nb, me, dst)

        elif move.eat_point:
            pass
        else:
            raise ValueError("Unknown move type")

        ns = GameState(nb, me.other, self, move, self.step+1)
        # 维护黑方边历史（规则 16） —— 修复单跳与连跳分别取起讫点
        if me == Player.black and (move.is_go or move.is_fly or move.is_skip_eat or move.is_skip_eat_seq):
            if move.is_go or move.is_fly:
                u,v = move.go_to.go, move.go_to.to
            elif move.is_skip_eat_seq:
                u,v = move.skip_eat_points[0].go, move.skip_eat_points[-1].to
            else:  # is_skip_eat（单跳）
                u,v = move.skip_eat_points.go, move.skip_eat_points.to
            ns.black_edge_history = deepcopy(self.black_edge_history)
            ns.black_edge_history.append((u,v))
        return ns

    # ---------- 胜负 ----------
    def is_over(self):
        return self.winner() is not None

    def winner_by_timeout(self):
        """
        超时判定：根据剩余棋子数量判断胜负
        用于达到最大步数限制时的判定

        判定规则：
        1. 棋子数量差距 >= 5: 棋子多的一方获胜
        2. 棋子数量差距 < 5: 比较成方数量
           - 成方数量差距 >= 2: 成方多的一方获胜
           - 否则: 平局

        Returns:
            Player.white / Player.black / None(平局)
        """
        # 布局期：不判终局
        if self.step < board_gild:
            return None

        black_count = self.board.get_player_total(Player.black)
        white_count = self.board.get_player_total(Player.white)

        # 1. 棋子数量差距显著（>=5子）
        piece_diff = abs(black_count - white_count)
        if piece_diff >= 5:
            return Player.black if black_count > white_count else Player.white

        # 2. 棋子数量接近，比较成方数量
        black_squares = count_squares(self.board, Player.black)
        white_squares = count_squares(self.board, Player.white)

        square_diff = abs(black_squares - white_squares)
        if square_diff >= 2:
            return Player.black if black_squares > white_squares else Player.white

        # 3. 棋子和成方都接近，判平局
        return None

    def winner(self, verbose=False):
        # 布局期：不判终局
        if self.step < board_gild:
            return None

        # 获取双方棋子数量
        white_total = self.board.get_player_total(Player.white)
        black_total = self.board.get_player_total(Player.black)

        # 获胜规则：对手棋子 ≤3 子/被吃光即判负
        for p in (Player.white, Player.black):
            opp = p.other
            opp_total = self.board.get_player_total(opp)
            if opp_total == 0:
                if verbose:
                    print(f"[判胜依据] 第{self.step}步，{p}获胜：对方{opp}被吃光(0子)")
                return p
            # 对方棋子数≤3，判己方获胜
            if opp_total <= 3:
                p_total = self.board.get_player_total(p)
                if verbose:
                    print(f"[判胜依据] 第{self.step}步，{p}获胜：对方{opp}只剩{opp_total}子(≤3)，己方{p_total}子")
                return p

        # 无合法走法 → 判负
        if not self.legal_moves():
            winner = self.next_player.other
            if verbose:
                print(f"[判胜依据] 第{self.step}步，{winner}获胜：{self.next_player}无合法走法（白{white_total}子 vs 黑{black_total}子）")
            return winner

        return None
    
    def _can_destroy_dalian(self, attacker: Player, defender: Player) -> bool:
        """
        检查攻击方是否能通过合法走法破坏防守方的褡裢
        返回True表示可以破坏，False表示无法破坏
        
        判定规则：
        1. 如果攻击方也有褡裢（有效阵型），认为可以抗衡
        2. 如果攻击方子力太少（<= 4子），认为无法有效破坏
        3. 否则，检查是否能通过各种走法（跳吃、走子、成方吃子等）破坏防守方的褡裢结构
        """
        # 获取防守方的所有褡裢
        defender_dalians = find_all_dalians(self.board, defender)
        if not defender_dalians:
            return True  # 没有褡裢，不需要破坏
        
        # 1. 检查攻击方是否也有褡裢（有效阵型）
        attacker_dalian_count = count_independent_dalians(self.board, attacker)
        if attacker_dalian_count >= 1:
            return True  # 攻击方有有效阵型，可以抗衡
        
        # 2. 检查攻击方子力
        attacker_pieces = self.board.get_player_total(attacker)
        if attacker_pieces <= 4:
            return False  # 子力太少，无法有效破坏
        
        # 3. 检查攻击方是否能通过各种走法破坏褡裢
        # 检查所有类型的走法：跳吃、普通走子、成方吃子
        temp_state = GameState(self.board, attacker, self.previous_state, self.last_move, self.step)
        original_count = compute_maximum_independent_dalians(defender_dalians)
        
        for move_decision in temp_state.legal_moves():
            try:
                nb = deepcopy(self.board)
                
                # 根据不同的走法类型执行
                if move_decision.act == 'skip_move':
                    # 单次跳吃
                    step = move_decision.points
                    nb.move_stone(attacker, step.go, step.to)
                    nb.remove_stone(step.eat)
                    
                elif move_decision.act == 'skip_eat_seq':
                    # 连续跳吃
                    for step in move_decision.points:
                        nb.move_stone(attacker, step.go, step.to)
                        nb.remove_stone(step.eat)
                        
                elif move_decision.act == 'move':
                    # 普通走子
                    nb.move_stone(attacker, move_decision.from_point, move_decision.to_point)
                    
                elif move_decision.act == 'form_eat':
                    # 成方吃子
                    nb.move_stone(attacker, move_decision.from_point, move_decision.to_point)
                    if hasattr(move_decision, 'eat_points') and move_decision.eat_points:
                        for eat_point in move_decision.eat_points:
                            nb.remove_stone(eat_point)
                            
                elif move_decision.act == 'fly':
                    # 飞子（不吃子）
                    nb.move_stone(attacker, move_decision.from_point, move_decision.to_point)
                    
                elif move_decision.act == 'fly_eat':
                    # 飞子成方吃子
                    nb.move_stone(attacker, move_decision.from_point, move_decision.to_point)
                    if hasattr(move_decision, 'eat_points') and move_decision.eat_points:
                        for eat_point in move_decision.eat_points:
                            nb.remove_stone(eat_point)
                else:
                    # 其他走法类型
                    continue
                
                # 检查执行后防守方的褡裢数量
                new_dalians = find_all_dalians(nb, defender)
                new_count = compute_maximum_independent_dalians(new_dalians)
                
                # 如果能减少褡裢数量，说明可以破坏
                if new_count < original_count:
                    return True
                    
            except Exception:
                continue
        
        return False
