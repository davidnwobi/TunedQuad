from itertools import combinations_with_replacement
from collections import Counter

def generate_polynomial_function_c(n, k):

    terms = ""
    variables = [f'var{i}' for i in range(n)]
    coeff_counter = 0
    for degree in range(k + 1):
        for exponents in combinations_with_replacement(range(n), degree):
            exponent_count = Counter(exponents)
            explict_power = lambda var, exp: ' * '.join(f'{var}' for _ in range(exp)) if exp > 1 else var
            term = ' * '.join(f'({explict_power(variables[i],exponent_count[i])})' if exponent_count[i] > 1 else variables[i]
                           for i in range(n) if exponent_count[i] > 0)
            if not term:
                term = '1'
                term += f' * coeffs[{coeff_counter}]'
                terms += f'{term}'
                coeff_counter += 1
            else:
                term += f' * coeffs[{coeff_counter}]'
                coeff_counter += 1
                terms += f' + {term}'

    v = [f'double var{i}' for i in range(n)]
    vars = ', '.join(v)
    func = f'''
        
double eval_poly({vars}){{
    return {terms} + intercept;
}}
    '''
    
    return func


def generate_integration_c(param_names):
    func = f'''

const int lb = 1;
const int ub = 15;
typedef double (*Integrand)(double x, {', '.join(f'double {name}' for name in param_names)});
''' 

    func += f'''
double integrate(Integrand f, double a, double b, {', '.join(f'double {name}' for name in param_names)}){{\n\n'''
    
    func += f'''
    // Determine the number of points for the Gauss quadrature
    int expo = (int)(eval_poly({', '.join(f'log2({name})' for name in param_names)}) + 1);
    int n = (int)(pow(2, max(lb, min(ub, expo))));
    
    double *xg, *wg;
    get_gauss_points(n, &xg, &wg);
    
    // Perform the integration
    double sum = 0;
    for (int i = 0; i < n; i++){{
        sum += f(a + (b - a) * 0.5 * (xg[i] + 1), {', '.join(name for name in param_names)}) * wg[i];
    }}
    sum *= (b - a) * 0.5;
    return sum;
    '''
    
    func += f'''
}}'''

    return func