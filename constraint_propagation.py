from random import random

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

if __name__ == '__main__':
    from itertools import product, chain
    from pprint import pprint

    # Magic square
    n = 3
    equal_sum = n * (n**2+1) / 2
    def is_state_valid(cell, cell_candidates):
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
    for solution in solve(cell_candidates, is_pair_valid=lambda c, v, cc, vv: v != vv, is_state_valid=is_state_valid):
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
    
    ###
    box = lambda p: (p[0] // 3, p[1] // 3)
    def is_sudoku_pair_valid(cell, value, other_cell, other_value):
        if value != other_value: return True
        return cell[0] != other_cell[0] and cell[1] != other_cell[1] and box(cell) != box(other_cell)
    indices = [(i, j) for i in range(9) for j in range(9)]
    def print_sudoku(solution):
        assert sorted(solution.keys()) == indices
        values = [str(v) for k, v in sorted(solution.items())]
        for row in range(9):
            print(f'{" ".join(values[row*9:row*9+3])} | {" ".join(values[row*9+3:row*9+6])} | {" ".join(values[row*9+6:row*9+9])}')
            if row % 3 == 2 and row != 8:
                print('-' * 21)
        print()

    cell_candidates = dict(zip(indices, [
        [6],[2],[4],[5],[3],[9],[1],[8],[7],
        [5],[1],range(1,10),[7],[2],[8],[6],[3],[4],
        [8],[3],[7],[6],[1],[4],[2],[9],[5],
        [1],[4],[3],[8],[6],[5],[7],[2],[9],
        [9],range(1,10),[8],range(1,10),[4],[7],[3],[6],[1],
        [7],[6],[2],[3],[9],[1],[4],[5],[8],
        [3],[7],[1],[9],[5],[6],[8],[4],[2],
        [4],range(1,10),[6],[1],[8],[2],range(1,10),[7],[3],
        [2],[8],range(1,10),[4],[7],[3],[9],[1],[6],
    ]))
    empty = {p: range(1, 10) for p in indices}
    for solution in solve(cell_candidates, is_sudoku_pair_valid):
        print_sudoku(solution)

    print("10 Sudokus from scratch:")
    for i, solution in zip(range(10), solve(empty, is_sudoku_pair_valid)):
        print_sudoku(solution)
    #exit()
    # Miracle Sudoku
    # https://www.youtube.com/watch?v=yKf9aUIxdb4
    is_orthogonal = lambda p, q: abs(p[0]-q[0]) + abs(p[1]-q[1]) == 1
    is_kings_move = lambda p, q: abs(p[0]-q[0]) <= 1 and abs(p[1]-q[1]) <= 1
    is_knights_move = lambda p, q: sorted([abs(p[0]-q[0]), abs(p[1]-q[1])]) == [1, 2]
    def is_miracle_sudoku_pair_valid(cell, value, other_cell, other_value):
        if not is_sudoku_pair_valid(cell, value, other_cell, other_value): return False
        if is_orthogonal(cell, other_cell) and abs(value - other_value) == 1: return False
        if value == other_value and (is_kings_move(cell, other_cell) or is_knights_move(cell, other_cell)): return False
        return True
    import re
    canonical_solution = dict(zip(indices, map(int, re.findall(r'\d', """
483 726 159
726 159 483
159 483 726

837 261 594
261 594 837
594 837 261

372 615 948
615 948 372
948 372 615
    """))))
    cell_candidates = dict({index: range(1, 10) for index in indices})
    cell_candidates[(4, 2)] = [1]
    cell_candidates[(5, 6)] = [2]
    for solution in solve(cell_candidates, is_miracle_sudoku_pair_valid):
        print('Miracle found!')
        print_sudoku(solution)
        assert solution == canonical_solution
    # https://www.youtube.com/watch?v=Tv-48b-KuxI
    cell_candidates = dict({index: range(1, 10) for index in indices})
    cell_candidates[(3, 2)] = [3]
    cell_candidates[(2, 4)] = [4]
    for solution in solve(cell_candidates, is_miracle_sudoku_pair_valid):
        print('Miracle 2 found!')
        print_sudoku(solution)
    # https://www.youtube.com/watch?v=8C-A7xmBLRU
    cell_candidates = dict(zip(indices, [range(1, 10) if i == "0" else [int(i)] for i in re.findall(r'\d', """
100 400 700
020 050 080
003 006 009

010 040 070
002 005 008
900 300 600

700 008 002
800 200 900
090 070 010
    """)]))
    for solution in solve(cell_candidates, is_sudoku_pair_valid):
        print('Diabolical found!')
        print_sudoku(solution)