# Constraint Propagation library

Solver for satisfying discrete constraints with a small number of discrete variables, such as in [Sudoku](https://en.wikipedia.org/wiki/Sudoku), the [Zebra Puzzle](https://en.wikipedia.org/wiki/Zebra_Puzzle), or [Nonograms/Picross](https://en.wikipedia.org/wiki/Nonogram).

This library exposes one main function for solving generic constraints, and a couple of functions for easy solving of common puzzles.

## Generic solver

```python
def solve(candidates_by_key, is_pair_valid):
```

- `candidates_by_key`: a mapping from id (key) to a list of candidates. An id can be an index, a string, an (x, y) point, a name, or whatever you prefer. E.g. for Sudoku, using (row, col) tuple as id, an empty board is `{(0, 0): range(1, 10), (0, 1): range(1, 10), ..., (8, 8): range(1, 10)}`.
- `is_pair_valid(key, value, other_key, other_value)`: a local constraint verifier, for pairs of values. A function that given the information about two ids and their values, returns `True` or `False` depending if the pair is allowed. E.g. in Sudoku `is_pair_valid((0, 0), 1, (0, 5), 1)` must return `False` because the two 1's are in the same row.

The solver then picks a random candidate for a random cell, and uses the `is_pair_valid` function to eliminate invalid candidates for every other key. The process is repeated until every cell has a single candidate, yielding a solution (e.g. `{(0, 0): 1, (0, 1): 5, (0, 2): 3, ...}` for Sudoku), or a dead end is found where a cell has no candidates left. Either way, the solver backtracks and tries different candidates, yielding all the different solutions it finds.

## Nonogram/Picross Solver

```python
def solve_nonogram(col_hints, row_hints):
    """
    Solves a black-and-white Nonogram/Picross and returns all solution grids as strings.
    Column hints and row hints should be the numbers representing filled group lengths
    for each column and row.

    For example, this Nonogram:

          1
          1 1 3
         -------
    1 1 | x   x |
      2 |   x x |
    1 1 | x   x |
         -------

    can be encoded as:

    >>> solve_nonogram(
            [[1, 1], [1], [3]],
            [[1, 1], [2], [1, 1]]
        )

        x x
         xx
        x x
        
    """
```

## Sudoku Solver

```python
def solve_sudoku(str_puzzle, extra_pairwise_rules=lambda key, value, other_key, other_value: True):
    """
    Given an string representing a Sudoku board, with cells represented by 1-9 or _ for unknown spots, returns
    all solutions as string grids.

    Characters other than 1-9 and _ in the input string are ignored, so all the
    following inputs are valid and equivalent:

    Shortest:
        1__4__7___2__5__8___3__6__9_1__4__7___2__5__89__3__6__7____8__28__2__9___9__7__1_

    Compact:
        1__4__7__
        _2__5__8_
        __3__6__9
        _1__4__7_
        __2__5__8
        9__3__6__
        7____8__2
        8__2__9__
        _9__7__1_

    Spaces between cells:
        1 _ _  4 _ _  7 _ _
        _ 2 _  _ 5 _  _ 8 _
        _ _ 3  _ _ 6  _ _ 9

        _ 1 _  _ 4 _  _ 7 _
        _ _ 2  _ _ 5  _ _ 8
        9 _ _  3 _ _  6 _ _

        7 _ _  _ _ 8  _ _ 2
        8 _ _  2 _ _  9 _ _
        _ 9 _  _ 7 _  _ 1 _

    Full ASCII table:
        1 _ _ | 4 _ _ | 7 _ _
        _ 2 _ | _ 5 _ | _ 8 _
        _ _ 3 | _ _ 6 | _ _ 9
        ----------------------
        _ 1 _ | _ 4 _ | _ 7 _
        _ _ 2 | _ _ 5 | _ _ 8
        9 _ _ | 3 _ _ | 6 _ _
        ----------------------
        7 _ _ | _ _ 8 | _ _ 2
        8 _ _ | 2 _ _ | 9 _ _
        _ 9 _ | _ 7 _ | _ 1 _
    """
```

## Example on how to use the generic solver for Sudoku

```py
from collections import namedtuple
from constraint_propagation import solve

def print_solution(solution):
    # The solution will be in the form {(row, column): number}, but since the
    # positions are ordered we can ignore them and just print the numbers in
    # sequence.
    print("""
{} {} {} | {} {} {} | {} {} {}
{} {} {} | {} {} {} | {} {} {}
{} {} {} | {} {} {} | {} {} {}
---------------------
{} {} {} | {} {} {} | {} {} {}
{} {} {} | {} {} {} | {} {} {}
{} {} {} | {} {} {} | {} {} {}
---------------------
{} {} {} | {} {} {} | {} {} {}
{} {} {} | {} {} {} | {} {} {}
{} {} {} | {} {} {} | {} {} {}
""".format(*solution.values()))

# Description of Sudoku puzzle.
Pos = namedtuple('Pos', 'row col')
def is_sudoku_pair_valid(pos, value, other_pos, other_value):
    """" Tests if a pair of cells are valid according to Sudoku rules. """
    return value != other_value or (
        pos.row != other_pos.row
        and pos.col != other_pos.col
        and (pos.row // 3, pos.col // 3) != (other_pos.row // 3, other_pos.col // 3)
    )

# Define a empty board and the candidates (numbers 1-9) for each cell.
empty_board = {Pos(row, col): range(1, 10) for row in range(9) for col in range(9)}

# Generate 10 Sudoku solutions by solving an empty board.
for i, solution in zip(range(10), solve(empty_board, is_sudoku_pair_valid)):
    print(f"Example Sudoku board number {i+1}")
    print_solution(solution)


# Diabolical Sudoku, from https://www.youtube.com/watch?v=8C-A7xmBLRU
# Reuse the (row, column) tuples from the empty board with the values from the
# literal string below.
diabolical_sudoku = dict(zip(empty_board, [range(1, 10) if i == "_" else [int(i)] for i in """
1 _ _  4 _ _  7 _ _
_ 2 _  _ 5 _  _ 8 _
_ _ 3  _ _ 6  _ _ 9

_ 1 _  _ 4 _  _ 7 _
_ _ 2  _ _ 5  _ _ 8
9 _ _  3 _ _  6 _ _

7 _ _  _ _ 8  _ _ 2
8 _ _  2 _ _  9 _ _
_ 9 _  _ 7 _  _ 1 _
""".split()]))

for solution in solve(diabolical_sudoku, is_sudoku_pair_valid):
    print('Found the solution for the Diabolical Sudoku!')
    print_solution(solution)


# Miracle Sudoku, has extra restrictions.
# https://www.youtube.com/watch?v=yKf9aUIxdb4
def is_miracle_sudoku_pair_valid(key, value, other_key, other_value):
    if not is_sudoku_pair_valid(key, value, other_key, other_value): return False

    is_sequential = abs(value - other_value) == 1
    d_row, d_col = abs(p.row - q.row), abs(p.col - q.col)
    is_orthogonal = d_row + d_col == 1
    is_kings_move = d_row + d_col <= 2
    is_knights_move = sorted([d_row, d_col]) == [1, 2]

    if is_sequential and is_orthogonal: return False
    if value == other_value and (is_kings_move or is_knights_move): return False
    return True

# Only these 2 given numbers are required to have a unique solution.
miracle_sudoku = {**empty_board, (4, 2): [1], (5, 6): [2]}

for solution in solve(miracle_sudoku, is_miracle_sudoku_pair_valid):
    print('Found the solution for the Miracle Sudoku!')
    print_solution(solution)
```