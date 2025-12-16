# agent/mcts_agent.py
"""基于蒙特卡洛树搜索(MCTS)的AI Agent"""
import random
import math
import time
from typing import Optional, Tuple, List
from .base import Agent
from jiu.jiuboard_fast import Move, GameState
from jiu.jiutypes import Player, Decision


class MCTSNode:
    """MCTS树节点"""
    def __init__(self, state: GameState, parent=None, move: Decision = None):
        self.state = state
        self.parent = parent
        self.move = move  # 到达此节点的动作
        self.children = []
        self.untried_moves = state.legal_moves()
        self.wins = 0.0
        self.visits = 0
    
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def is_terminal(self):
        return self.state.is_over()
    
    def best_child(self, exploration_weight=1.414):
        """使用UCB1公式选择最佳子节点"""
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                ucb1_value = float('inf')
            else:
                # UCB1 = win_rate + C * sqrt(ln(parent_visits) / child_visits)
                win_rate = child.wins / child.visits
                exploration = exploration_weight * math.sqrt(
                    math.log(self.visits) / child.visits
                )
                ucb1_value = win_rate + exploration
            choices_weights.append(ucb1_value)
        
        return self.children[choices_weights.index(max(choices_weights))]
    
    def expand(self):
        """扩展一个未尝试的动作"""
        decision = self.untried_moves.pop()
        # 将Decision转换为Move
        move_obj = self._decision_to_move_internal(decision)
        next_state = self.state.apply_move(move_obj)
        child_node = MCTSNode(next_state, parent=self, move=decision)
        self.children.append(child_node)
        return child_node
    
    def _decision_to_move_internal(self, dec: Decision):
        """将Decision转换为Move对象"""
        if dec.act == 'put_piece':
            return Move.put_piece(dec.points)
        if dec.act == 'is_go':
            return Move.go_piece(dec.points)
        if dec.act == 'skip_move':
            return Move.move_skip(dec.points)
        if dec.act == 'skip_eat_seq':
            return Move.move_skip_seq(dec.points)
        if dec.act == 'fly':
            return Move.fly_piece(dec.points)
        if dec.act == 'eat_point':
            return Move.eat(dec.points)
        return None


class MCTSAgent(Agent):
    """基于蒙特卡洛树搜索的AI Agent"""
    
    def __init__(self, simulation_time: float = 2.0, max_simulations: int = 500):
        """
        Args:
            simulation_time: 每步思考时间（秒）
            max_simulations: 最大模拟次数
        """
        super().__init__()
        self.simulation_time = simulation_time
        self.max_simulations = max_simulations
    
    def select_move(self, game_state: GameState) -> Tuple[Optional[Move], list]:
        """使用MCTS选择最佳走法"""
        candidates = game_state.legal_moves()
        if not candidates:
            return None, []
        
        if len(candidates) == 1:
            return self._decision_to_move(candidates[0])
        
        # 运行MCTS
        root = MCTSNode(game_state)
        start_time = time.time()
        simulations = 0
        
        while (time.time() - start_time < self.simulation_time and 
               simulations < self.max_simulations):
            node = self._select(root)
            
            if not node.is_terminal():
                if not node.is_fully_expanded():
                    node = node.expand()
                reward = self._simulate(node.state)
            else:
                reward = self._get_terminal_reward(node.state, game_state.next_player)
            
            self._backpropagate(node, reward)
            simulations += 1
        
        # 选择访问次数最多的子节点
        if not root.children:
            return self._decision_to_move(random.choice(candidates))
        
        best_child = max(root.children, key=lambda c: c.visits)
        
        print(f"MCTS: {simulations} simulations, best move visits: {best_child.visits}/{root.visits}")
        
        return self._decision_to_move(best_child.move)
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """选择阶段：从根节点向下选择到叶子节点"""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            else:
                node = node.best_child()
        return node
    
    def _simulate(self, state: GameState) -> float:
        """模拟阶段：从当前状态随机模拟到游戏结束"""
        from jiu.jiutypes import board_gild

        current_state = state
        max_depth = 200  # 限制模拟深度，避免无限循环
        depth = 0

        while not current_state.is_over() and depth < max_depth:
            decisions = current_state.legal_moves()
            if not decisions:
                break

            # 布局阶段：使用更好的启发式
            if current_state.step < board_gild:
                decision = self._select_placement_move(current_state, decisions)
            else:
                # 对战阶段：优先选择吃子多的走法
                decisions.sort(key=lambda d: d.eats, reverse=True)
                if decisions[0].eats > 0:
                    # 如果有吃子走法，从前3个中随机选
                    top_decisions = [d for d in decisions if d.eats == decisions[0].eats][:3]
                    decision = random.choice(top_decisions)
                else:
                    # 否则随机选择
                    decision = random.choice(decisions)

            # 将Decision转换为Move
            move_obj, _ = self._decision_to_move(decision)
            if move_obj is None:
                break

            current_state = current_state.apply_move(move_obj)
            depth += 1

        # 返回奖励
        return self._get_terminal_reward(current_state, state.next_player)

    def _select_placement_move(self, state: GameState, decisions: list):
        """布局阶段的智能选择策略"""
        from jiu.jiutypes import Point

        if not decisions:
            return None

        # 如果只有少数选择，直接随机
        if len(decisions) <= 5:
            return random.choice(decisions)

        # 计算每个位置的评分
        scored_decisions = []
        board = state.board
        center = board.num_rows // 2 + 1

        for dec in decisions:
            if dec.act != 'put_piece':
                continue

            point = dec.points
            score = 0.0

            # 1. 避免过于集中：惩罚周围有太多己方棋子的位置
            # 使用四向相邻（与走子规则一致）
            neighbors = point.neighbors_four()
            my_neighbors = sum(1 for p in neighbors if board.get(p) == state.next_player)
            score -= my_neighbors * 2.0  # 周围己方棋子越多，分数越低

            # 2. 均匀分布：鼓励分散布局
            # 计算到最近己方棋子的距离
            my_stones = board.stones(state.next_player)
            if my_stones:
                min_dist = min(abs(point.row - s.row) + abs(point.col - s.col) for s in my_stones)
                score += min(min_dist, 5) * 0.5  # 距离越远越好（但有上限）

            # 3. 中心区域略微加分（但不要太集中）
            dist_to_center = abs(point.row - center) + abs(point.col - center)
            if 3 <= dist_to_center <= 8:
                score += 1.0  # 中等距离的中心区域

            # 4. 避免边缘（前期）
            if state.step < 50:
                if point.row <= 2 or point.row >= board.num_rows - 1 or \
                   point.col <= 2 or point.col >= board.num_cols - 1:
                    score -= 2.0

            scored_decisions.append((dec, score))

        if not scored_decisions:
            return random.choice(decisions)

        # 选择分数最高的前20%，然后随机选一个
        scored_decisions.sort(key=lambda x: x[1], reverse=True)
        top_count = max(1, len(scored_decisions) // 5)
        top_decisions = [d for d, s in scored_decisions[:top_count]]

        return random.choice(top_decisions)
    
    def _get_terminal_reward(self, state: GameState, player: Player) -> float:
        """获取终局奖励"""
        winner = state.winner()
        
        if winner is None:
            # 游戏未结束或平局，根据局面评估
            return self._evaluate_position(state, player)
        elif winner == player:
            return 1.0
        else:
            return 0.0
    
    def _evaluate_position(self, state: GameState, player: Player) -> float:
        """评估局面（简单启发式）"""
        my_pieces = state.board.get_player_total(player)
        opp_pieces = state.board.get_player_total(player.other)

        # 基于棋子数的评估
        total_pieces = my_pieces + opp_pieces
        if total_pieces == 0:
            return 0.5

        piece_ratio = my_pieces / total_pieces

        return max(0.0, min(1.0, piece_ratio))
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """反向传播：更新路径上的所有节点"""
        while node is not None:
            node.visits += 1
            # 从节点玩家的角度更新胜率
            # 如果当前节点是对手的回合，奖励要反转
            if node.parent is not None and node.state.next_player != node.parent.state.next_player:
                node.wins += reward
            else:
                node.wins += (1.0 - reward)
            node = node.parent
    
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

