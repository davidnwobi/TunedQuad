
        
def eval_poly_2_2(vars, coeffs, intercept):
    return 1 * coeffs[0] + vars[0] * coeffs[1] + vars[1] * coeffs[2] + vars[0] ** 2 * coeffs[3] + vars[0] * vars[1] * coeffs[4] + vars[1] ** 2 * coeffs[5] + intercept
    