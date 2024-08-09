from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import typing as tp
import numpy as np


def get_shift(pred, orig, threshold=.99):
    '''
    Get the shift needed to make the predicted values greater than the original values (threshold)% of the time.
    '''
    def percentage_greater():
        return (pred >= orig).mean() > threshold
    
    l = 0
    r = pred.max()-orig.min() # unless there is some weird edge case, this should be positive and a sufficient starting point
    eps = 1e-6
    while(abs(l-r) > eps): # probably not the best way to do this but I know binary search works
        m = (l+r)/2
        if percentage_greater(pred+m, orig):
            r = m
        else:
            l = m


# reg_param_list: tp.List[np.ndarray] = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
tuned_quad_dict: tp.Dict[tp.Tuple[int, ...], int] = {(1, 2): 3, (4, 5): 6}

def fit_dict(tuned_quad_dict: tp.Dict[tp.Tuple[int, ...], int], degree, threshold=.99) -> tp.Tuple[int, int, np.ndarray, np.float64]:
    '''
    Fit a polynomial to the tuned quadrature dictionary.
    
    Parameters
    ----------
    tuned_quad_dict : `DictType(Tuple(int32, ...), int32)`
        The tuned matrix.
    
    Returns
    -------
    `Tuple(int, int, np.ndarray, np.float64)`
    '''

    n_params = len(tuned_quad_dict.keys()[0])
    if n_params > degree:
        raise ValueError("Number of variables is greater than polynomial degree")
    
    # Fit a polynomial to the tuned matrix
    features = np.vstack(np.array([np.array(k) for k in tuned_quad_dict.keys()]))
    values = np.array(tuned_quad_dict.values())

    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(features, values)
    prediction = model.predict(features)

    # Get the shift needed to make the predicted values greater than the original values (threshold)% of the time.
    shift = get_shift(prediction, values, threshold)

    coeff = model.named_steps['linearregression'].coef_
    intercept = model.named_steps['linearregression'].intercept_+shift

    return n_params, degree, coeff, intercept