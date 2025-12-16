#!/usr/bin/env python3
"""
Arena Match - 模型对战评估脚本

两个AI模型对打，统计胜率
"""
import os
import sys
import argparse
import time
from typing import Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm

from jiu.jiuboard_fast import GameState, Move, Player
from jiu.jiutypes import Decision, board_gild
from agent.base import Agent
from agent.dyt_agent import DyTAgent, encode_board_state, get_phase_id, decision_to_dict
from dyt.candidate_features import build_features_for_candidates


class JiuqiNetAgent(Agent):
    """基于JiuqiNet的AI Agent"""
    
    def __init__(self, model_path: str, device: str = 'cuda', temperature: float = 0.1):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature
        self.history = []
        
        # 加载模型
        from jcar.model import JiuqiNet
        from jcar.config import JiuqiNetConfig
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 获取配置
        if 'model_config' in checkpoint:
            cfg_dict = checkpoint['model_config']
            cfg = JiuqiNetConfig(**cfg_dict)
        else:
            cfg = JiuqiNetConfig()
        
        self.model = JiuqiNet(cfg).to(self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"✅ JiuqiNet loaded from {model_path}")
    
    def select_move(self, game_state: GameState) -> Tuple[Optional[Move], list]:
        """选择最佳走法"""
        candidates = game_state.legal_moves()
        if not candidates:
            return None, []
        
        if len(candidates) == 1:
            move = self._decision_to_move(candidates[0])
            self._update_history(game_state)
            return move, []
        
        # 编码状态
        obs = encode_board_state(game_state, self.history)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        
        phase_id = get_phase_id(game_state)
        phase_tensor = torch.tensor([phase_id], dtype=torch.long, device=self.device)
        
        flying = (hasattr(game_state, '_is_flying_stage') and game_state._is_flying_stage())
        
        # 构建候选特征
        cand_dicts = [decision_to_dict(dec) for dec in candidates]
        cand_features = build_features_for_candidates(cand_dicts, phase_id, flying)
        cand_tensor = torch.from_numpy(cand_features).to(self.device)
        
        # 模型推理
        with torch.no_grad():
            logits_list, value = self.model.score_candidates(obs_tensor, phase_tensor, [cand_tensor])
            logits = logits_list[0]
            
            if self.temperature > 0:
                probs = torch.softmax(logits / self.temperature, dim=0)
                idx = torch.multinomial(probs, 1).item()
            else:
                idx = torch.argmax(logits).item()
        
        selected_dec = candidates[idx]
        move = self._decision_to_move(selected_dec)
        self._update_history(game_state)
        
        return move, []
    
    def _update_history(self, game_state: GameState):
        from copy import deepcopy
        self.history.append(deepcopy(game_state.board))
        if len(self.history) > 3:
            self.history.pop(0)
    
    def _decision_to_move(self, dec: Decision) -> Move:
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


def play_one_game(agent1: Agent, agent2: Agent, max_moves: int = 800, verbose: bool = False):
    """进行一局对战，agent1执白，agent2执黑"""
    state = GameState.new_game(14)
    move_count = 0
    
    # 重置历史
    if hasattr(agent1, 'history'):
        agent1.history = []
    if hasattr(agent2, 'history'):
        agent2.history = []
    
    while not state.is_over() and move_count < max_moves:
        current_agent = agent1 if state.next_player == Player.white else agent2
        
        move, _ = current_agent.select_move(state)
        if move is None:
            break
        
        state = state.apply_move(move)
        move_count += 1
        
        if verbose and move_count % 50 == 0:
            w = state.board.get_player_total(Player.white)
            b = state.board.get_player_total(Player.black)
            print(f"  Step {move_count}: 白{w} vs 黑{b}")
    
    winner = state.winner()
    if winner is None and move_count >= max_moves:
        winner = state.winner_by_timeout()

    return winner, move_count


def run_arena(agent1: Agent, agent2: Agent, num_games: int = 10, verbose: bool = True):
    """运行对战评估"""
    results = {'agent1_wins': 0, 'agent2_wins': 0, 'draws': 0}
    game_lengths = []

    print(f"\n{'='*60}")
    print(f"🏟️ Arena Match: {num_games} 局对战")
    print(f"{'='*60}")

    for game_num in tqdm(range(num_games), desc="对战中"):
        # 交替执白
        if game_num % 2 == 0:
            white_agent, black_agent = agent1, agent2
            white_name, black_name = "Agent1", "Agent2"
        else:
            white_agent, black_agent = agent2, agent1
            white_name, black_name = "Agent2", "Agent1"

        start_time = time.time()
        winner, move_count = play_one_game(white_agent, black_agent, verbose=False)
        game_time = time.time() - start_time
        game_lengths.append(move_count)

        if winner == Player.white:
            if white_name == "Agent1":
                results['agent1_wins'] += 1
            else:
                results['agent2_wins'] += 1
            winner_str = f"{white_name}(白)"
        elif winner == Player.black:
            if black_name == "Agent1":
                results['agent1_wins'] += 1
            else:
                results['agent2_wins'] += 1
            winner_str = f"{black_name}(黑)"
        else:
            results['draws'] += 1
            winner_str = "和棋"

        if verbose:
            print(f"  Game {game_num+1}: {winner_str} ({move_count}步, {game_time:.1f}s)")

    # 统计结果
    total = num_games
    agent1_rate = results['agent1_wins'] / total * 100
    agent2_rate = results['agent2_wins'] / total * 100
    draw_rate = results['draws'] / total * 100
    avg_length = np.mean(game_lengths)

    print(f"\n{'='*60}")
    print(f"📊 对战结果")
    print(f"{'='*60}")
    print(f"  Agent1 胜率: {results['agent1_wins']}/{total} = {agent1_rate:.1f}%")
    print(f"  Agent2 胜率: {results['agent2_wins']}/{total} = {agent2_rate:.1f}%")
    print(f"  和棋:        {results['draws']}/{total} = {draw_rate:.1f}%")
    print(f"  平均步数:    {avg_length:.1f}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description='模型对战评估')
    parser.add_argument('--agent1', type=str, required=True, help='Agent1模型路径')
    parser.add_argument('--agent2', type=str, required=True, help='Agent2模型路径')
    parser.add_argument('--agent1-type', type=str, default='jiuqinet', choices=['jiuqinet', 'dyt'])
    parser.add_argument('--agent2-type', type=str, default='dyt', choices=['jiuqinet', 'dyt'])
    parser.add_argument('--num-games', type=int, default=10, help='对战局数')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--temperature', type=float, default=0.1, help='采样温度')
    args = parser.parse_args()

    # 创建Agent
    print(f"\n📦 加载模型...")

    if args.agent1_type == 'jiuqinet':
        agent1 = JiuqiNetAgent(args.agent1, device=args.device, temperature=args.temperature)
    else:
        agent1 = DyTAgent(args.agent1, device=args.device, temperature=args.temperature)

    if args.agent2_type == 'jiuqinet':
        agent2 = JiuqiNetAgent(args.agent2, device=args.device, temperature=args.temperature)
    else:
        agent2 = DyTAgent(args.agent2, device=args.device, temperature=args.temperature)

    print(f"  Agent1: {args.agent1_type} - {args.agent1}")
    print(f"  Agent2: {args.agent2_type} - {args.agent2}")

    # 运行对战
    results = run_arena(agent1, agent2, num_games=args.num_games, verbose=True)

    return results


if __name__ == '__main__':
    main()

