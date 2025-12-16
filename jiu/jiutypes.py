import enum
from collections import namedtuple
from typing import Iterable, Tuple, Union

# ---------------------------------------------------------------------------
# 棋盘常量（对齐公开赛数据集：列用字母 A-N，行用数字 1-14）
# ---------------------------------------------------------------------------
board_size = 14          # 实际棋盘为 14×14
board_gild = 196         # 布子阶段步数

_DATASET_COL_LABELS = tuple(chr(ord('A') + i) for i in range(board_size))
_LETTER_TO_COL = {c: idx + 1 for idx, c in enumerate(_DATASET_COL_LABELS)}
_COL_TO_LETTER = {idx + 1: c for idx, c in enumerate(_DATASET_COL_LABELS)}
dataset_columns = _DATASET_COL_LABELS
dataset_rows = tuple(range(1, board_size + 1))


def dataset_letter_to_col(letter: str) -> int:
    """公开赛棋谱中的列字母（A-N） → 1-based 列号."""
    if not letter:
        raise ValueError("Column letter cannot be empty")
    col = _LETTER_TO_COL.get(letter.strip().upper())
    if col is None:
        raise ValueError(f"Unsupported column letter: {letter!r}")
    return col


def dataset_col_to_letter(col: int) -> str:
    """1-based 列号 → 公开赛棋谱字母."""
    letter = _COL_TO_LETTER.get(int(col))
    if letter is None:
        raise ValueError(f"Unsupported column index: {col!r}")
    return letter


def dataset_row_to_index(row: Union[int, str], *, allow_extended_row: bool = False) -> int:
    """公开赛棋谱的行数字 → 1-based 行号."""
    if isinstance(row, str):
        row = row.strip()
        if not row:
            raise ValueError("Row string cannot be empty")
        idx = int(row)
    else:
        idx = int(row)
    if 1 <= idx <= board_size:
        return idx
    if allow_extended_row and idx == board_size + 1:
        # 部分棋谱在提子列表里用 15 表示“离盘”位置，调用方如需保存可显式允许
        return idx
    raise ValueError(f"Row index {idx} out of range 1..{board_size}")


def dataset_coord_to_point(coord: Union[str, Tuple[str, Union[str, int]]],
                           *,
                           allow_extended_row: bool = False) -> "Point":
    """
    将棋谱坐标（如 'G7' 或 ('G',7)）转换为 Point。
    """
    if isinstance(coord, str):
        raw = coord.strip()
        if not raw:
            raise ValueError("Coordinate string cannot be empty")
        if ',' in raw:
            letter, row = raw.split(',', 1)
        else:
            letter, row = raw[0], raw[1:]
    else:
        if len(coord) != 2:
            raise ValueError(f"Coordinate tuple must be (letter,row), got {coord!r}")
        letter, row = coord
    col_idx = dataset_letter_to_col(letter)
    row_idx = dataset_row_to_index(row, allow_extended_row=allow_extended_row)
    return Point(row_idx, col_idx)


def point_to_dataset_coord(point: "Point") -> Tuple[str, int]:
    """Point → (列字母, 行数字) 的棋谱坐标表示."""
    return dataset_col_to_letter(point.col), int(point.row)


class Player(enum.Enum):
    white = 1  # 规则 6：白先
    black = 2

    @property
    def other(self):
        return Player.white if self == Player.black else Player.black


class Skip_eat(namedtuple('skip_eat_points', 'go eat to')):
    __slots__ = ()


class Go(namedtuple('goto', 'go to')):
    __slots__ = ()


class Decision(namedtuple('Decision', 'act points eats')):
    __slots__ = ()


class Point(namedtuple('Point', 'row col')):
    __slots__ = ()

    # ------------------------------------------------------------------
    # 数据集互操作
    # ------------------------------------------------------------------
    @classmethod
    def from_dataset(cls,
                     letter: str,
                     row: Union[int, str],
                     *,
                     allow_extended_row: bool = False) -> "Point":
        """
        根据公开赛棋谱坐标（列字母、行数字）构造 Point。
        allow_extended_row=True 时允许行号为 15（提子记录用），
        但该坐标超出实盘，需要调用方自行判断是否可用。
        """
        col_idx = dataset_letter_to_col(letter)
        row_idx = dataset_row_to_index(row, allow_extended_row=allow_extended_row)
        return cls(row_idx, col_idx)

    def to_dataset(self) -> Tuple[str, int]:
        """返回 (列字母, 行数字) 形式的棋谱坐标."""
        return point_to_dataset_coord(self)

    # ------------------------------------------------------------------
    # 邻居与判定
    # ------------------------------------------------------------------
    def neighbors_four(self) -> Iterable["Point"]:
        cand = [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
        ]
        return [p for p in cand if 1 <= p.row <= board_size and 1 <= p.col <= board_size]

    def neighbors_eight(self) -> Iterable["Point"]:
        cand = [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
            Point(self.row - 1, self.col - 1),
            Point(self.row - 1, self.col + 1),
            Point(self.row + 1, self.col - 1),
            Point(self.row + 1, self.col + 1),
        ]
        return [p for p in cand if 1 <= p.row <= board_size and 1 <= p.col <= board_size]

    def neighbors_four_far2(self) -> Iterable["Point"]:
        """返回距离为2的4个直线方向邻居（上下左右），用于跳吃"""
        cand = [
            Point(self.row, self.col - 2),      # 左
            Point(self.row, self.col + 2),      # 右
            Point(self.row - 2, self.col),      # 上
            Point(self.row + 2, self.col),      # 下
        ]
        return [p for p in cand if 1 <= p.row <= board_size and 1 <= p.col <= board_size]

    def neighbors_eight_far2(self) -> Iterable["Point"]:
        """返回距离为2的8个方向邻居（包括斜向）"""
        cand = [
            Point(self.row, self.col - 2),
            Point(self.row, self.col + 2),
            Point(self.row - 2, self.col),
            Point(self.row + 2, self.col),
            Point(self.row - 2, self.col - 2),
            Point(self.row - 2, self.col + 2),
            Point(self.row + 2, self.col - 2),
            Point(self.row + 2, self.col + 2),
        ]
        return [p for p in cand if 1 <= p.row <= board_size and 1 <= p.col <= board_size]

    def neighbor_fangs(self) -> Iterable["Point"]:
        s = [
            Point(self.row - 1, self.col - 1),
            Point(self.row - 1, self.col + 1),
            Point(self.row + 1, self.col - 1),
            Point(self.row + 1, self.col + 1),
        ]
        return [p for p in s if 1 <= p.row <= board_size and 1 <= p.col <= board_size]


__all__ = [
    'board_size',
    'board_gild',
    'dataset_columns',
    'dataset_rows',
    'dataset_letter_to_col',
    'dataset_col_to_letter',
    'dataset_row_to_index',
    'dataset_coord_to_point',
    'point_to_dataset_coord',
    'Player',
    'Skip_eat',
    'Go',
    'Decision',
    'Point',
]
