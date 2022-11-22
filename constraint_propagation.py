from random import random
from itertools import product
from collections import namedtuple

def solve(cell_candidates, is_pair_valid=None, is_state_valid=None, unpropagated_cells=None):
    """
    """
    # If not provided, assume that no constraints have been propagated and that every cell must be checked.
    if unpropagated_cells is None: unpropagated_cells = list(cell_candidates.keys())

    # Use a stack instead of recursion so that we can more easily bail out of this guess with a `return`.
    while unpropagated_cells:
        cell = unpropagated_cells.pop(0)
        if is_state_valid and not is_state_valid(cell, cell_candidates):
            return
        if is_pair_valid is not None:
            candidates = cell_candidates[cell]
            for other_cell, other_candidates in cell_candidates.items():
                if other_cell == cell: continue
                is_value_allowed = lambda other_value: any(is_pair_valid(cell, value, other_cell, other_value) for value in candidates)
                new_candidates = tuple(filter(is_value_allowed, other_candidates))
                if not new_candidates: return
                cell_candidates[other_cell] = new_candidates
                # Only propagating solved cells (`len(new_candidates) == 1`) is an unnecessary restriction
                # and leads to extra candidates, but in practice is several times faster than propagating
                # every change in candidates.
                if len(other_candidates) > len(new_candidates) == 1: unpropagated_cells.append(other_cell)

    pending = [(cell, candidates) for cell, candidates in cell_candidates.items() if len(candidates) != 1]
    if pending:
        # Select the most constrained cell to guess. Sacrifices lexical ordering of solutions
        # to achieve orders of magnitude better performance.
        pending_cell, candidates = min(pending, key=lambda p: len(p[1]))
        for value in candidates:
            # Potentially expensive, but cheaper than trying to undo changes to candidate lists.
            cell_candidates_copy = cell_candidates.copy()
            cell_candidates_copy[pending_cell] = (value,)
            yield from solve(cell_candidates_copy, is_pair_valid, is_state_valid, [pending_cell])
    else:
        yield {cell: candidates[0] for cell, candidates in cell_candidates.items()}


GridLoc = namedtuple('GridLoc', 'row col')

def _is_nonogram_pair_valid(cell, value, other_cell, other_value):
    cell_type, i = cell
    other_cell_type, j = other_cell

    # Comparison of row and row, or col and col, are not necessary.
    if cell_type == other_cell_type: return i != j or value == other_value
    # Check if the intersection of the row and col has the same value.
    return value[j] == other_value[i]
def _generate_valid_nonogram_lines(n, filled_group_sizes):
    assert sum(filled_group_sizes) + len(filled_group_sizes) - 1 <= n
    if len(filled_group_sizes) == 0: return [' ' * n]
    if sum(filled_group_sizes) == n: return ['x' * n]

    max_empty_group_size = n - sum(filled_group_sizes)
    lines = []
    # The first and last empty groups can have size 0, the other ones must have at least 1 empty cell in them.
    for empty_group_sizes in product(range(0, max_empty_group_size+1), *([range(1, max_empty_group_size+1)] * (len(filled_group_sizes)-1)), range(0, max_empty_group_size+1)):
        if sum(empty_group_sizes) + sum(filled_group_sizes) != n: continue
        line = []
        for empty_size, filled_size in zip(empty_group_sizes, filled_group_sizes):
            line.extend(' ' * empty_size + 'x' * filled_size)
        line.extend(' ' * empty_group_sizes[-1])
        lines.append(line)
    return lines
def solve_nonogram(col_hints, row_hints):
    """
    Solves a black-and-white Nonogram and returns the grid representation. Column hints and row hints should
    be the list of hints for each column and row. For example, for the following Nonogram:

        1
        1 1 3
    1 1 x   x
      2   x x
    1 1 x   x

    >>> solve_nonogram([[1, 1], [1], [3]], [[1, 1], [2], [1, 1]])
    x x
     xx
    x x

    """
    n_rows = len(row_hints)
    n_cols = len(col_hints)
    assert all(sum(entry)+len(entry)-1 <= n_rows for entry in col_hints)
    assert all(sum(entry)+len(entry)-1 <= n_cols for entry in row_hints)

    cell_candidates = {('row', i): _generate_valid_nonogram_lines(n_cols, hint) for i, hint in enumerate(row_hints)} | {('col', i): _generate_valid_nonogram_lines(n_rows, hint) for i, hint in enumerate(col_hints)}
    solution = next(solve(cell_candidates, is_pair_valid=_is_nonogram_pair_valid))
    return '\n'.join(''.join(values) for (cell_type, _), values in solution.items() if cell_type == 'row')

def _is_sudoku_pair_valid(cell, value, other_cell, other_value):
    """" Tests if a pair of cells and their values are valid according to Sudoku rules. """
    return value != other_value or (
        cell.row != other_cell.row
        and cell.col != other_cell.col
        and (cell.row // 3, cell.col // 3) != (other_cell.row // 3, other_cell.col // 3)
    )
def _sudoku_solution_to_str(solution):
        # The solution will be in the form {(row, column): number}, but since the
        # positions are ordered we can ignore them and just print the numbers in
        # sequence.
        return """
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
        """.format(*solution.values())
def solve_sudoku(str_puzzle):
    """
    Given an ASCII representation of a Sudoku board with 0-9 and _ for unknown cells, returns
    an equivalent ASCII representation of the solved board. Whitespace is ignored.

    Example valid puzzle:

    1 _ _  4 _ _  7 _ _
    _ 2 _  _ 5 _  _ 8 _
    _ _ 3  _ _ 6  _ _ 9

    _ 1 _  _ 4 _  _ 7 _
    _ _ 2  _ _ 5  _ _ 8
    9 _ _  3 _ _  6 _ _

    7 _ _  _ _ 8  _ _ 2
    8 _ _  2 _ _  9 _ _
    _ 9 _  _ 7 _  _ 1 _
    """
    assert isinstance(str_puzzle, str) and all(char in '01234567890_ \n' for char in str_puzzle) and len(str_puzzle.split()) == 9 * 9, 'Invalid puzzle string'


    empty_board = {GridLoc(row, col): range(1, 10) for row in range(9) for col in range(9)}
    puzzle = dict(zip(empty_board, [range(1, 10) if i == "_" else [int(i)] for i in str_puzzle.split()]))

    solutions = solve(puzzle, is_pair_valid=_is_sudoku_pair_valid)

    # The solution will be in the form {(row, column): number}, but since the
    # positions are ordered we can ignore them and just print the numbers in
    # sequence.
    return _sudoku_solution_to_str(next(solutions))

if __name__ == '__main__':
    from pprint import pprint

    # Nonogram from the CSS puzzle
    # https://cohost.org/blackle/post/260204-div-style-width-60
    print(solve_nonogram([[4], [6], [1, 1, 4], [1, 1, 1, 2], [1, 1, 2], [1, 1], [1, 1], [4]], [[4], [1, 1], [4, 1], [2, 1], [5, 1], [3, 1], [4, 1], [4]]))
    print()

    # Magic square
    n = 3
    equal_sum = n * (n**2+1) / 2
    def is_magic_square_valid(cell, cell_candidates):
        row, col = cell
        lines = []
        lines.append([(i, col) for i in range(n)])
        lines.append([(row, i) for i in range(n)])
        if row == col: lines.append([(i, i) for i in range(n)])
        if row == n - col - 1: lines.append([(i, n - i - 1) for i in range(n)])
        for cell_line in lines:
            min_sum = sum(min(cell_candidates[cell]) for cell in cell_line)
            max_sum = sum(max(cell_candidates[cell]) for cell in cell_line)
            if min_sum > equal_sum or max_sum < equal_sum:
                #print([cell_candidates[cell] for cell in cell_line], min_sum, max_sum, equal_sum)
                return False
        return True
    indices = [(i, j) for i in range(n) for j in range(n)]
    def print_magic_square(solution):
        assert sorted(solution.keys()) == indices
        values = [str(v) for k, v in sorted(solution.items())]
        for row in range(n):
            print(' '.join(values[row*n:row*n+n]))
        print()
    cell_candidates = {i: range(1, n**2+1) for i in indices}
    for solution in solve(cell_candidates, is_pair_valid=lambda c, v, cc, vv: v != vv, is_state_valid=is_magic_square_valid):
        print_magic_square(solution)

    ###
    #https://web.stanford.edu/~laurik/fsmbook/examples/Einstein'sPuzzle.html
    BLUE, GREEN, RED, WHITE, YELLOW, BRIT, DANE, GERMAN, NORWEGIAN, SWEDE, BEER, COFFEE, MILK, TEA, WATER, BLENDS, BLUE_MASTER, DUNHILL, PALL_MALL, PRINCE, BIRDS, CATS, DOGS, HORSES, FISH = 'BLUE, GREEN, RED, WHITE, YELLOW, BRIT, DANE, GERMAN, NORWEGIAN, SWEDE, BEER, COFFEE, MILK, TEA, WATER, BLENDS, BLUE_MASTER, DUNHILL, PALL_MALL, PRINCE, BIRDS, CATS, DOGS, HORSES, FISH'.split(', ')

    colors = {BLUE, GREEN, RED, WHITE, YELLOW}
    nationalities = {BRIT, DANE, GERMAN, NORWEGIAN, SWEDE}
    drinks = {BEER, COFFEE, MILK, TEA, WATER}
    cigarettes = {BLENDS, BLUE_MASTER, DUNHILL, PALL_MALL, PRINCE}
    pets = {BIRDS, CATS, DOGS, HORSES, FISH}

    is_neighbor = lambda cell_a, cell_b: abs(cell_a - cell_b) == 1
    def is_zebra_pair_valid(cell, value, other_cell, other_value):
        # Uniqueness.
        for part_a, part_b in zip(value, other_value):
            if part_a == part_b: return False

        if GREEN in value and WHITE in other_value and other_cell != cell + 1: return False
        if BLENDS in value and CATS in other_value and not is_neighbor(cell, other_cell): return False
        if HORSES in value and DUNHILL in other_value and not is_neighbor(cell, other_cell): return False
        if BLENDS in value and WATER in other_value and not is_neighbor(cell, other_cell): return False
        if NORWEGIAN in value and BLUE in other_value and not is_neighbor(cell, other_cell): return False

        return True

    required_pairs = [
        (BRIT, RED),
        (SWEDE, DOGS),
        (DANE, TEA),
        (GREEN, COFFEE),
        (PALL_MALL, BIRDS),
        (YELLOW, DUNHILL),
        (BLUE_MASTER, BEER),
        (GERMAN, PRINCE),
    ]
    forbidden_pairs = [
        (BLENDS, CATS),
        (HORSES, DUNHILL),
        (BLENDS, WATER),
        (BLUE, NORWEGIAN),
    ]

    def is_combination_valid(cell, combination):
        for pair in required_pairs:
            if (pair[0] in combination) != (pair[1] in combination):
                return False
        for pair in forbidden_pairs:
            if pair[0] in combination and pair[1] in combination:
                return False
        if GREEN in combination and cell == 5: return False
        if WHITE in combination and cell == 1: return False
        if (NORWEGIAN in combination) != (cell == 1): return False
        if (MILK in combination) != (cell == 3): return False
        return True

    all_combinations = tuple(product(colors, nationalities, drinks, cigarettes, pets))
    cell_candidates = {i: [c for c in all_combinations if is_combination_valid(i, c)] for i in range(1, 6)}

    for solution in solve(cell_candidates, is_zebra_pair_valid):
        assert all((FISH in value) == (GERMAN in value) for value in solution.values()), solution
        pprint(solution)

    ### Sudoku

    # Solve the second Sudoku from the CSS puzzle at
    # https://cohost.org/blackle/post/260204-div-style-width-60
    print('CSS Puzzle Sudoku #2:')
    print(solve_sudoku("""
    1 _ 7  3 _ _  _ 6 9
    _ 2 6  _ _ 1  4 5 _
    _ _ _  _ _ _  7 _ 2

    6 _ 2  _ 5 9  1 _ 4
    7 _ _  2 1 4  _ _ _
    5 1 _  _ _ 3  2 9 _

    _ _ 1  _ _ 8  5 _ 7
    4 _ 3  _ _ _  _ 8 _
    _ 8 _  6 _ _  3 _ _
    """))
    print()

    # Define a completely empty board and the candidates (numbers 1-9) for each cell.
    empty_board = {GridLoc(row, col): range(1, 10) for row in range(9) for col in range(9)}

    # Generate 10 Sudoku solutions by solving an empty board.
    for i, solution in zip(range(10), solve(empty_board, _is_sudoku_pair_valid)):
        print(f"Example Sudoku board number {i+1}")
        print(_sudoku_solution_to_str(solution))


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

    for solution in solve(diabolical_sudoku, _is_sudoku_pair_valid):
        print('Found the solution for the Diabolical Sudoku!')
        print(_sudoku_solution_to_str(solution))


    # Miracle Sudoku, has extra restrictions.
    # https://www.youtube.com/watch?v=yKf9aUIxdb4
    def is_miracle_sudoku_pair_valid(cell, value, other_cell, other_value):
        if not _is_sudoku_pair_valid(cell, value, other_cell, other_value): return False

        is_sequential = abs(value - other_value) == 1
        d_row, d_col = abs(cell.row - other_cell.row), abs(cell.col - other_cell.col)
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
        print(_sudoku_solution_to_str(solution))


    # No Xing - "Another Sudoku Breakthrough: Man Vs Machine"
    # https://www.youtube.com/watch?v=5q94_FcnYMI
    noxing_sudoku = dict(zip(empty_board, [range(1, 10) if i == "_" else [int(i)] for i in """
    6 _ _  5 _ 4  3 _ _
    _ _ 9  _ _ _  _ _ _
    1 _ _  _ _ _  _ _ 5

    8 _ _  _ 5 3  _ _ 6
    _ 6 5  _ _ 7  _ _ _
    4 _ _  9 _ _  _ _ 7

    _ 1 _  _ 4 _  _ 9 _
    _ _ 2  _ 8 _  _ _ _
    _ _ _  3 _ 5  2 _ 4
    """.split()]))

    for solution in solve(noxing_sudoku, _is_sudoku_pair_valid):
        print('Found the solution for the No Xing Sudoku!')
        print(_sudoku_solution_to_str(solution))