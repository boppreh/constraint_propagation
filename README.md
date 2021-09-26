# Constraint Propagation library

Solver for satisfying discrete constraints with a small number of discrete variables, such as in [Sudoku](https://en.wikipedia.org/wiki/Sudoku) and the [Zebra Puzzle](https://en.wikipedia.org/wiki/Zebra_Puzzle).

This library exposes only one function:

```python
def solve(cell_candidates, is_pair_valid):
```

- `cell_candidates`: a mapping from cell id to a list of candidates for that cell. A cell id can be an index, a string, an (x, y) point, or whatever you prefer. E.g. for Sudoku, using (row, col) tuple as id, an empty board is `{(0, 0): range(1, 10), (0, 1): range(1, 10), ..., (8, 8): range(1, 10)}`.
- `is_pair_valid(cell, value, other_cell, other_value)`: a local constraint verifier, for cell pairs. A function that given the information about two cell ids and their values, returns `True` or `False` depending if the pair is allowed. E.g. in Sudoku `is_pair_valid((0, 0), 1, (0, 5), 1)` must return `False` because the two 1's are in the same column.

The solver then picks a random candidate for a random cell, and uses the `is_pair_valid` function to verify this choice against every other candidate of every other cell. The process is repeated until every cell has a single candidate, yielding a solution (e.g. `{(0, 0): 1, (0, 1): 5, (0, 2): 3, ...}` for Sudoku), or a dead end is found where a cell has no candidates left. Either way, the solver backtracks and tries different candidates, yielding all the different solutions it finds.

## Sudoku example with variants

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
Cell = namedtuple('Cell', 'row col')
def is_sudoku_pair_valid(cell, value, other_cell, other_value):
    """" Tests if a pair of cells and their values are valid according to Sudoku rules. """
    return value != other_value or (
        cell.row != other_cell.row
        and cell.col != other_cell.col
        and (cell.row // 3, cell.col // 3) != (other_cell.row // 3, other_cell.col // 3)
    )

# Define a completely empty board and the candidates (numbers 1-9) for each cell.
empty_board = {Cell(row, col): range(1, 10) for row in range(9) for col in range(9)}

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