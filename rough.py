from itertools import combinations_with_replacement
from collections import defaultdict

def generate_terms(variables, degree, var_val = [2,2,2]):
    terms = defaultdict(float)
    terms['1'] = 0.0  # Adding the constant term
    for d in range(degree + 1):
        print(list(combinations_with_replacement(range(len(variables)), d)))
        for combo in combinations_with_replacement(range(len(variables)), d):
            exponents = [0] * len(variables)
            for idx in combo:
                exponents[idx] += 1
            print("exponents: ", exponents)
            term = ' '.join([f'{variables[i]}^{exp}' if exp > 1 else variables[i] for i, exp in enumerate(exponents) if exp > 0])
            
            if term == '':
                continue
            print('term:', term)
            print([var_val[i]**exp if exp > 1 else var_val[i] for i, exp in enumerate(exponents) if exp > 0])
            print()
            terms[term] = 0.0
    return terms

from kron_dummy import *

def get_kronrod_n(i):
    n = 2**i
    return eval(f'GAUSS_{n}_XG'), eval(f'GAUSS_{n}_WG'), eval(f'GAUSS_{n}_XK'), eval(f'GAUSS_{n}_WK')

pr
print(get_kronrod_n(15))