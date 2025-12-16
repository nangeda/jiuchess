import platform, re, subprocess
from jiu.jiutypes import *
from jiu import jiutypes

STONE_TO_CHAR = {
    None: ' . ',
    jiutypes.Player.black.value: ' x ',
    jiutypes.Player.white.value: ' o ',
}

def print_board(board):
    col_labels = [dataset_col_to_letter(i) for i in range(1, board.num_cols + 1)]
    print('   ' + '  '.join(col_labels))
    for row in range(1, board.num_rows + 1):
        row_label = f"{row:2d}"
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(jiutypes.Point(row=row, col=col))
            if stone is not None:
                line.append(STONE_TO_CHAR[stone.value])
            else:
                line.append(STONE_TO_CHAR[stone])
        print(f'{row_label} ' + ''.join(line))

def init_info(FF=None, GM=None, SZ=None, PB=None, PW=None, RE=None):
    s = '(;'
    if FF is not None: s += f"FF[{FF}] "
    if GM is not None: s += f"GM[{GM}] "
    if SZ is not None: s += f"SZ[{SZ}] "
    if PB is not None: s += f"PB[{PB}] "
    if PW is not None: s += f"PW[{PW}] "
    if RE is not None: s += f"RE[{RE}] "
    s += ';'
    return s

def move_str(bot_move, fang_eats, player):
    if fang_eats is None:
        fang_eats = []
    s = ''
    num_chart = {i: chr(ord('a')+i-1) for i in range(1,15)}  # 1->'a'
    c = 'B' if player==Player.black else 'W'
    o = 'W' if c=='B' else 'B'
    if bot_move.is_put:
        put = bot_move.point
        s = '{}[{}{}];'.format(c, num_chart[put.row], num_chart[put.col])
    elif bot_move.is_go or bot_move.is_fly:
        go = bot_move.go_to.go; to = bot_move.go_to.to
        s = '{}[{}{}]-O[{}{}],'.format(c, num_chart[go.row], num_chart[go.col], num_chart[to.row], num_chart[to.col])
        eats = ''
        if len(fang_eats)!=0: eats = 'FC:'
        for p in fang_eats:
            eats += '{}[{}{}],'.format(o, num_chart[p.row], num_chart[p.col])
        s += eats
        s = s[:-1]+';'
    elif bot_move.is_skip_eat:
        go = bot_move.skip_eat_points.go
        to = bot_move.skip_eat_points.to
        eat = bot_move.skip_eat_points.eat
        s = '{}[{}{}]-O[{}{}],TC:{}[{}{}],'.format(c, num_chart[go.row], num_chart[go.col],
                                                   num_chart[to.row], num_chart[to.col],
                                                   o, num_chart[eat.row], num_chart[eat.col])
        eats = ''
        if len(fang_eats)!=0: eats = 'FC:'
        for p in fang_eats:
            eats += '{}[{}{}],'.format(o, num_chart[p.row], num_chart[p.col])
        s += eats
        s = s[:-1]+';'
    elif bot_move.is_skip_eat_seq:
        go = bot_move.skip_eat_points[0].go
        tos = [se.to for se in bot_move.skip_eat_points]
        eats_seq = [se.eat for se in bot_move.skip_eat_points]
        s = '{}[{}{}]'.format(c, num_chart[go.row], num_chart[go.col])
        for t in tos:
            s += '-O[{}{}]'.format(num_chart[t.row], num_chart[t.col])
        s += ',TC:'
        for e in eats_seq:
            s += '{}[{}{}],'.format(o, num_chart[e.row], num_chart[e.col])
        eats = ''
        if len(fang_eats)!=0: eats = 'FC:'
        for p in fang_eats:
            eats += '{}[{}{}],'.format(o, num_chart[p.row], num_chart[p.col])
        s += eats
        s = s[:-1]+';'
    return s

# 为兼容保留（如需把文本解析回 Move，可继续扩展）
def extract_letters_with_bracket_data(input_string):  # 简化版占位
    return {'B': [], 'W': [], 'O': [], 'TC': [], 'FC': []}

def tuple_point(t):
    return Point(t[0], t[1])

def tuple_move(move_tuple):  # 简化版占位（如果你有现成解析需求可替换）
    raise NotImplementedError

def go_to_points(move):
    if move.is_go or move.is_fly:
        return move.go_to.go, move.go_to.to
    elif move.is_skip_eat:
        return move.skip_eat_points.go, move.skip_eat_points.to
    elif move.is_skip_eat_seq:
        return move.skip_eat_points[0].go, move.skip_eat_points[-1].to
