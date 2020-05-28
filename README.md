# Constraint Propagation library

This is a single-function library for solving assignment problems based on constraint propagation, that is, choosing values to put into cells so that local and global restrictions are met. Classical examples are Sudoku and Zebra puzzles.

```
def solve(cell_candidates, is_pair_valid=None, is_state_valid=None):
```

- `cell_candidates`: a map of cell ids to list of candidates for that cell, e.g. for Sudoku `{(0, 0): range(1, 10), (0, 1): range(1, 10), ..., (8, 8): range(1, 10)}`.
- `is_pair_valid(cell, value, other_cell, other_value)`: optional, a function that given the information about two candidates and their cells, return True or False depending if the pair is allowed. E.g. `is_pair_valid((0, 0), 1, (0, 5), 1) -> False`.
- `is_state_valid(cell, cell_candidates)`: optional, a function that given the global state of all candidates, and the id of a cell that was updated, returns True or False depending if the state is allowed.

## Example

```py
from collections import namedtuple
from constraint_propagation import solve

def print_solution(solution):
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
Cell = namedtuple('Cell', 'row col')
def is_sudoku_pair_valid(cell, value, other_cell, other_value):
    """" Tests if a pair of cells and their values are valid according to Sudoku rules. """
    return value != other_value or (
        cell.row != other_cell.row
        and cell.col != other_cell.col
        and (cell.row // 3, cell.col // 3) != (other_cell.row // 3, other_cell.col // 3)
    )

# Define a completely empty board and the candidates (1-9) for each cell.
empty_board = {Cell(row, col): range(1, 10) for row in range(9) for col in range(9)}

# Generate Sudoku solutions by solving an empty board.
for i, solution in zip(range(10), solve(empty_board, is_sudoku_pair_valid)):
    print(f"Example Sudoku board number {i+1}")
    print_solution(solution)

# Diabolical Sudoku, from https://www.youtube.com/watch?v=8C-A7xmBLRU
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
def is_miracle_sudoku_pair_valid(cell, value, other_cell, other_value):
    if not is_sudoku_pair_valid(cell, value, other_cell, other_value): return False

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