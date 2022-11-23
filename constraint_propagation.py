import re
from random import random
from itertools import product, permutations
from collections import namedtuple

def solve(candidates_by_key, is_pair_valid, unpropagated_keys=None):
    """
    """
    # If not provided, assume that no constraints have been propagated and that every key must be checked.
    if unpropagated_keys is None:
        unpropagated_keys = list(candidates_by_key.keys())

    # Use a stack instead of recursion so that we can more easily bail out of this guess with a `return`.
    while unpropagated_keys:
        key = unpropagated_keys.pop(0)
        candidates = candidates_by_key[key]
        for other_key, other_candidates in candidates_by_key.items():
            if other_key == key: continue
            is_value_allowed = lambda other_value: any(is_pair_valid(key, value, other_key, other_value) for value in candidates)
            new_candidates = tuple(filter(is_value_allowed, other_candidates))
            if not new_candidates: return
            candidates_by_key[other_key] = new_candidates
            # Only propagating solved keys (`len(new_candidates) == 1`) is an unnecessary restriction
            # and leads to extra candidates, but in practice is several times faster than propagating
            # every change in candidates.
            if len(other_candidates) > len(new_candidates) == 1: unpropagated_keys.append(other_key)

    pending = [(key, candidates) for key, candidates in candidates_by_key.items() if len(candidates) != 1]
    if pending:
        # Select the most constrained key to guess. Sacrifices lexical ordering of solutions
        # to achieve orders of magnitude better performance.
        pending_key, candidates = min(pending, key=lambda p: len(p[1]))
        for value in candidates:
            # Potentially expensive, but cheaper than trying to undo changes to candidate lists.
            candidates_by_key_copy = candidates_by_key.copy()
            candidates_by_key_copy[pending_key] = (value,)
            yield from solve(candidates_by_key_copy, is_pair_valid, [pending_key])
    else:
        yield {key: candidates[0] for key, candidates in candidates_by_key.items()}


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
    # Nonograms require more global checks than sudoku, so the encoding is
    # different: instead of encoding each cell in the grid with a constraint,
    # we encode whole rows and columns, overlapping grid cells and all.
    #
    # This requires an extra constraint that a row and a column must have the same
    # cells in their intersection, but it drastically reduces the number of candidates
    # by only allowing rows and columns that are valid according to the hints.
    def is_pair_valid(key, value, other_key, other_value):
        # Key can be for example ('row', 5) or ('col', 0).
        key_type, i = key
        other_key_type, j = other_key
        # We should never have to compare a row or column to itself.
        assert not (key_type == other_key_type and i == j)
        # Check if the intersection of the row and col has the same value.
        return key_type == other_key_type or value[j] == other_value[i]
    def generate_valid_lines(n, filled_group_sizes):
        """
        Given a length and hints, generates all valid lines (rows or columns).
        For example n=5 and filled_group_sizes=[2, 1] should generate:

        [
            ' xx x',
            'xx  x',
            'xx x ',
        ]
        """
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
    def solution_to_str(solution):
        """ Converts a nonogram dict solution to an ASCII grid. """
        return '\n\t' + '\n\t'.join(''.join(values) for (cell_type, _), values in solution.items() if cell_type == 'row') + '\n'

    n_rows = len(row_hints)
    n_cols = len(col_hints)
    # Ensure hints don't have obvious contradictions:
    assert all(sum(entry)+len(entry)-1 <= n_rows for entry in col_hints)
    assert all(sum(entry)+len(entry)-1 <= n_cols for entry in row_hints)

    candidates_by_key = {
        **{('row', i): generate_valid_lines(n_cols, hint) for i, hint in enumerate(row_hints)},
        **{('col', i): generate_valid_lines(n_rows, hint) for i, hint in enumerate(col_hints)},
    }
    solutions = solve(candidates_by_key, is_pair_valid=is_pair_valid)
    return map(solution_to_str, solutions)

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
    Pos = namedtuple('Pos', 'row col')
    def is_pair_valid(key, value, other_key, other_value):
        """" Tests if a pair of cells and their values are valid according to Sudoku rules. """
        return (value != other_value or (
            key.row != other_key.row
            and key.col != other_key.col
            and (key.row // 3, key.col // 3) != (other_key.row // 3, other_key.col // 3)
        )) and extra_pairwise_rules(key, value, other_key, other_value)
    def solution_to_str(solution):
            # The solution will be in the form {(row, column): number}, but since the
            # positions are ordered we can ignore them and just print the numbers in
            # sequence.
            return """
    ? ? ? | ? ? ? | ? ? ?
    ? ? ? | ? ? ? | ? ? ?
    ? ? ? | ? ? ? | ? ? ?
    ---------------------
    ? ? ? | ? ? ? | ? ? ?
    ? ? ? | ? ? ? | ? ? ?
    ? ? ? | ? ? ? | ? ? ?
    ---------------------
    ? ? ? | ? ? ? | ? ? ?
    ? ? ? | ? ? ? | ? ? ?
    ? ? ? | ? ? ? | ? ? ?
    """.replace('?', '{}').format(*solution.values())

    str_cells = re.findall(r'[0-9_]', str_puzzle)
    assert len(str_cells) == 9 * 9, 'Invalid Sudoku string, should contain 0-9 and _ for unknown cells.'


    empty_board = {Pos(row, col): range(1, 10) for row in range(9) for col in range(9)}
    puzzle = dict(zip(empty_board, [range(1, 10) if i == "_" else [int(i)] for i in str_cells]))

    solutions = solve(puzzle, is_pair_valid=is_pair_valid)

    return map(solution_to_str, solutions)

if __name__ == '__main__':
    from pprint import pprint

    # Nonogram from the CSS puzzle
    # https://cohost.org/blackle/post/260204-div-style-width-60
    print('Solution for Nonogram')
    print(solve_nonogram([[4], [6], [1, 1, 4], [1, 1, 1, 2], [1, 1, 2], [1, 1], [1, 1], [4]], [[4], [1, 1], [4, 1], [2, 1], [5, 1], [3, 1], [4, 1], [4]]))

    # Magic Squares
    n = 3
    magic_constant = n * (n**2+1) / 2
    valid_lines = [line for line in permutations(range(1, 10), r=3) if sum(line) == magic_constant]
    candidates_by_key = {
        **{((i, 0), (i, 1), (i, 2)): valid_lines for i in range(3)},
        **{((0, i), (1, i), (2, i)): valid_lines for i in range(3)},
        ((0, 0), (1, 1), (2, 2)): valid_lines,
        ((0, 2), (1, 1), (2, 0)): valid_lines,
    }
    def is_magic_square_pair_valid(key, value, other_key, other_value):
        for pair in key:
            if pair in other_key:
                return value[key.index(pair)] == other_value[other_key.index(pair)]
        return True
    print('Magic square solution')
    solution = next(solve(candidates_by_key, is_pair_valid=is_magic_square_pair_valid))
    grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for pairs, values in solution.items():
        for (row, col), value in zip(pairs, values):
            grid[row][col] = value
    print('\t' + '\n\t'.join(' '.join(map(str, row)) for row in grid))
    print()

    ###
    #https://web.stanford.edu/~laurik/fsmbook/examples/Einstein'sPuzzle.html
    BLUE, GREEN, RED, WHITE, YELLOW, BRIT, DANE, GERMAN, NORWEGIAN, SWEDE, BEER, COFFEE, MILK, TEA, WATER, BLENDS, BLUE_MASTER, DUNHILL, PALL_MALL, PRINCE, BIRDS, CATS, DOGS, HORSES, FISH = 'BLUE, GREEN, RED, WHITE, YELLOW, BRIT, DANE, GERMAN, NORWEGIAN, SWEDE, BEER, COFFEE, MILK, TEA, WATER, BLENDS, BLUE_MASTER, DUNHILL, PALL_MALL, PRINCE, BIRDS, CATS, DOGS, HORSES, FISH'.split(', ')

    colors = {BLUE, GREEN, RED, WHITE, YELLOW}
    nationalities = {BRIT, DANE, GERMAN, NORWEGIAN, SWEDE}
    drinks = {BEER, COFFEE, MILK, TEA, WATER}
    cigarettes = {BLENDS, BLUE_MASTER, DUNHILL, PALL_MALL, PRINCE}
    pets = {BIRDS, CATS, DOGS, HORSES, FISH}

    is_neighbor = lambda key_a, key_b: abs(key_a - key_b) == 1
    def is_zebra_pair_valid(key, value, other_key, other_value):
        # Uniqueness.
        for part_a, part_b in zip(value, other_value):
            if part_a == part_b: return False

        if GREEN in value and WHITE in other_value and other_key != key + 1: return False
        if BLENDS in value and CATS in other_value and not is_neighbor(key, other_key): return False
        if HORSES in value and DUNHILL in other_value and not is_neighbor(key, other_key): return False
        if BLENDS in value and WATER in other_value and not is_neighbor(key, other_key): return False
        if NORWEGIAN in value and BLUE in other_value and not is_neighbor(key, other_key): return False

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
    candidates_by_key = {i: [c for c in all_combinations if is_combination_valid(i, c)] for i in range(1, 6)}

    for solution in solve(candidates_by_key, is_zebra_pair_valid):
        assert all((FISH in value) == (GERMAN in value) for value in solution.values()), solution
        pprint(solution)
    print()

    ### Sudoku

    # Solve the second Sudoku from the CSS puzzle at
    # https://cohost.org/blackle/post/260204-div-style-width-60
    print('CSS Puzzle Sudoku #2:')
    print(next(solve_sudoku("""
    1 _ 7  3 _ _  _ 6 9
    _ 2 6  _ _ 1  4 5 _
    _ _ _  _ _ _  7 _ 2

    6 _ 2  _ 5 9  1 _ 4
    7 _ _  2 1 4  _ _ _
    5 1 _  _ _ 3  2 9 _

    _ _ 1  _ _ 8  5 _ 7
    4 _ 3  _ _ _  _ 8 _
    _ 8 _  6 _ _  3 _ _
    """)))
    print()

    empty_sudoku = """
    _ _ _ | _ _ _ | _ _ _
    _ _ _ | _ _ _ | _ _ _
    _ _ _ | _ _ _ | _ _ _
    ---------------------
    _ _ _ | _ _ _ | _ _ _
    _ _ _ | _ _ _ | _ _ _
    _ _ _ | _ _ _ | _ _ _
    ---------------------
    _ _ _ | _ _ _ | _ _ _
    _ _ _ | _ _ _ | _ _ _
    _ _ _ | _ _ _ | _ _ _
    """
    # Generate 10 Sudoku boards by solving an empty board.
    for i, solution in zip(range(1, 11), solve_sudoku(empty_sudoku)):
        print(f"Example Sudoku board number {i}")
        print(solution)


    # Diabolical Sudoku, from https://www.youtube.com/watch?v=8C-A7xmBLRU
    # Reuse the (row, column) tuples from the empty board with the values from the
    # literal string below.
    diabolical_sudoku = """
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
    print('Solution for the Diabolical Sudoku:')
    print(next(solve_sudoku(diabolical_sudoku)))


    # Miracle Sudoku, has extra restrictions.
    # https://www.youtube.com/watch?v=yKf9aUIxdb4
    def is_miracle_sudoku_pair_valid(key, value, other_key, other_value):
        is_sequential = abs(value - other_value) == 1
        d_row, d_col = abs(key.row - other_key.row), abs(key.col - other_key.col)
        is_orthogonal = d_row + d_col == 1
        is_kings_move = d_row + d_col <= 2
        is_knights_move = sorted([d_row, d_col]) == [1, 2]

        if is_sequential and is_orthogonal: return False
        if value == other_value and (is_kings_move or is_knights_move): return False
        return True
    miracle_sudoku = """
    _ _ _ | _ _ _ | _ _ _
    _ _ _ | _ _ _ | _ _ _
    _ _ _ | _ _ _ | _ _ _
    ---------------------
    _ _ _ | _ _ _ | _ _ _
    _ _ 1 | _ _ _ | _ _ _
    _ _ _ | _ _ _ | 2 _ _
    ---------------------
    _ _ _ | _ _ _ | _ _ _
    _ _ _ | _ _ _ | _ _ _
    _ _ _ | _ _ _ | _ _ _
    """
    print('Solution for the Miracle Sudoku:')
    print(next(solve_sudoku(miracle_sudoku, extra_pairwise_rules=is_miracle_sudoku_pair_valid)))


    # No Xing - "Another Sudoku Breakthrough: Man Vs Machine"
    # https://www.youtube.com/watch?v=5q94_FcnYMI
    noxing_sudoku = """
    6 _ _  5 _ 4  3 _ _
    _ _ 9  _ _ _  _ _ _
    1 _ _  _ _ _  _ _ 5

    8 _ _  _ 5 3  _ _ 6
    _ 6 5  _ _ 7  _ _ _
    4 _ _  9 _ _  _ _ 7

    _ 1 _  _ 4 _  _ 9 _
    _ _ 2  _ 8 _  _ _ _
    _ _ _  3 _ 5  2 _ 4
    """
    print('Solution for the No Xing Sudoku:')
    print(next(solve_sudoku(noxing_sudoku)))