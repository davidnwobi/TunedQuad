try:
    import sasmodels
except ImportError:
    raise ImportError("sasmodels is not installed. Please install it using 'pip install sasmodels'.")

from numpy import logspace, sqrt
from matplotlib import pyplot as plt
from sasmodels.core import load_model
from sasmodels.direct_model import call_kernel, call_Fq
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot

import pandas
import numpy as np
from itertools import product
from timeit import timeit

model_name = 'cylinder_accurate'
try:
    model_cylinder_accurate = load_model(model_name)
except Exception as e:
    print(f"Error: {e}")
    print(f"Model {model_name} not found. Please check if the model exists in sasmodels.")
    raise e

q = np.linspace(0.001, 0.1, 200)
kernel_cylinder = model_cylinder_accurate.make_kernel([q])

lengths = [0.000001, 0.0001, 0.001, 0.01, 0.01, 1, 10, 100, 10000, 100000, 1000000]
# lengths = lengths[::4]
rtols = [0.1, 0.01, 0.001, 0.0001, 0.00001]
# rtols = [0.00001]

Iq_data = {(rtol, length): {} for rtol in rtols for length in lengths}

radii = [0.000001, 0.0001, 0.001, 0.01, 0.01, 1, 10, 100, 10000, 100000, 1000000]
# radii = radii[::4]

res = dict()
n_runs = 10
for rtol, length, radius in product(rtols, lengths, radii):
    pars_cylinder = {'radius': radius, 'length': length, 'scale': 1, 'background': 0.001, 'rtol': rtol}
    Iq_data[(rtol, length)][radius] = call_kernel(kernel_cylinder, pars_cylinder)
    res[(rtol, length, radius)] = timeit(lambda: call_kernel(kernel_cylinder, pars_cylinder), number=n_runs) / n_runs

index = pandas.MultiIndex.from_tuples(res.keys(), names=['rtol', 'length', 'radius'])

df = pandas.DataFrame(res.values(), index=index, columns=['time'])
df.to_excel('cylinder_accurate_benchmark.xlsx')
