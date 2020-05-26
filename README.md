# Constraint Propagation library

This is a single-function library for solving assignment problems based on constraint propagation, that is, choosing values to put into cells so that local and global restrictions are met. Classical examples are Sudoku and Zebra puzzles.

```
def solve(cell_candidates, is_pair_valid=None, is_state_valid=None, unpropagated_cells=None):
```

- `cell_candidates`: a map of cell ids to list of candidates for that cell, e.g. for Sudoku `{(0, 0): range(1, 10), (0, 1): range(1, 10), ..., (8, 8): range(1, 10)}`.
- `is_pair_valid(cell, value, other_cell, other_value)`: optional, a function that given the information about two candidates and their cells, return True or False depending if the pair is allowed. E.g. `is_pair_valid((0, 0), 1, (0, 5), 1) -> False`.
- `is_state_valid(cell, cell_candidates)`: optional, a function that given the global state of all candidates, and the id of a cell that was updated, returns True or False depending if the state is allowed.

## Example

```py
from constraint_propagation import solve

box = lambda p: (p[0] // 3, p[1] // 3) # Compute which of the 9 boxes a point belongs to.
def is_sudoku_pair_valid(cell, value, other_cell, other_value):
    """" Tests if a pair of cells and their values are valid according to Sudoku rules. """
    if value != other_value: return True
    return cell[0] != other_cell[0] and cell[1] != other_cell[1] and box(cell) != box(other_cell)

def print_sudoku(solution):
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

indices = [(i, j) for i in range(9) for j in range(9)]
empty_board = {p: range(1, 10) for p in indices}

print("Generate 10 Sudokus solutions from scratch:")
for i, solution in zip(range(10), solve(empty_board, is_sudoku_pair_valid)):
    print_sudoku(solution)

# Diabolical Sudoku, from https://www.youtube.com/watch?v=8C-A7xmBLRU
cell_candidates = dict(zip(indices, [range(1, 10) if i == "_" else [int(i)] for i in """
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
for solution in solve(cell_candidates, is_sudoku_pair_valid):
    print('Found the solution for the Diabolical Sudoku!')
    print_sudoku(solution)
```