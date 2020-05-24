from random import random

def propagate(cell, cell_candidates, is_pair_valid):
    is_compatible = lambda other_cell, other_value: any(is_pair_valid(cell, value, other_cell, other_value) for value in cell_candidates[cell])
    for other_cell, other_candidates in cell_candidates.items():
        if other_cell == cell: continue
        new_candidates = tuple(other_value for other_value in other_candidates if is_compatible(other_cell, other_value))
        if not new_candidates: return False
        cell_candidates[other_cell] = new_candidates
        if len(other_candidates) != len(new_candidates) and not propagate(other_cell, cell_candidates, is_pair_valid): return False
    return True

def guess_and_backtrack(cell_candidates, is_pair_valid):
    try:
        pending_cell, candidates = next((cell, candidates) for cell, candidates in cell_candidates.items() if len(candidates) != 1)
    except StopIteration:
        yield {cell: candidates[0] for cell, candidates in cell_candidates.items()}
        return

    for value in candidates:
        cell_candidates_copy = cell_candidates.copy()
        cell_candidates_copy[pending_cell] = (value,)
        if propagate(pending_cell, cell_candidates_copy, is_pair_valid):
            yield from guess_and_backtrack(cell_candidates_copy, is_pair_valid)

def solve(cell_candidates, is_pair_valid):
    for cell, candidates in cell_candidates.items():
        assert propagate(cell, cell_candidates, is_pair_valid)

    return guess_and_backtrack(cell_candidates, is_pair_valid)

def validate(solution, is_pair_valid):
    for cell, value in solution.items():
        for other_cell, other_value in solution.items():
            if cell == other_cell: continue
            assert is_pair_valid(cell, value, other_cell, other_value)

if __name__ == '__main__':
    ###
    #https://web.stanford.edu/~laurik/fsmbook/examples/Einstein'sPuzzle.html

    from itertools import product, chain
    from pprint import pprint

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
        values = list(map(str, solution.values()))
        for row in range(9):
            print(f'{" ".join(values[row*9:row*9+3])} | {" ".join(values[row*9+3:row*9+6])} | {" ".join(values[row*9+6:row*9+9])}')
            if row % 3 == 2 and row != 8:
                print('-' * 21)

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


    for i, solution in zip(range(100), solve(empty, is_sudoku_pair_valid)):
        print_sudoku(solution)
    exit()
    # Miracle Sudoku
    # https://www.youtube.com/watch?v=yKf9aUIxdb4
    kings_moves = lambda p: [(p[0]+x, p[1]+y) for x, y in product([-1, 0, 1], [-1, 0, 1]) if x or y]
    knights_moves = lambda p: [(p[0]+x, p[1]+y) for x, y in chain(product([-1, 1], [-2, 2]), product([-2, 2], [-1, 1]))]
    is_orthogonal = lambda p, q: abs(p[0]-q[0]) + abs(p[1]-q[1]) == 1
    def is_miracle_sudoku_pair_valid(cell, value, other_cell, other_value):
        if not is_sudoku_pair_valid(cell, value, other_cell, other_value): return False
        if is_orthogonal(cell, other_cell) and abs(value - other_value) == 1: return False
        if value == other_value and other_cell in chain(kings_moves(cell), knights_moves(cell)): return False
        return True
    indices = [(i, j) for i in range(9) for j in range(9)]
    cell_candidates = dict({index: range(1, 10) for index in indices})
    cell_candidates[(3, 5)] = [1]
    cell_candidates[(7, 6)] = [2]
    for solution in solve(cell_candidates, is_miracle_sudoku_pair_valid):
        print('Miracle found!')
        pprint(solution)
    cell_candidates = dict({index: range(1, 10) for index in indices})
    cell_candidates[(4, 2)] = [4]
    cell_candidates[(2, 3)] = [2]
    for solution in solve(cell_candidates, is_miracle_sudoku_pair_valid):
        print('Miracle 2 found!')
        pprint(solution)