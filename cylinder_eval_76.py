"""
Minimal example of calling a kernel for a specific set of q values.
"""

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


model_cylinder = load_model('cylinder')
q = np.linspace(0.001, 0.1, 20000)
kernel_cylinder = model_cylinder.make_kernel([q])


lengths = [0.000001, 0.0001, 0.001, 0.01, 0.01, 1, 10, 100, 10000, 100000, 1000000]
# lengths = [10, 100, 10000, 100000, 1000000]

Iq_data = {length: {} for length in lengths}


radii = [0.000001, 0.0001, 0.001, 0.01, 1, 10, 100, 10000, 100000, 1000000]
# radii =  [1, 10, 100, 10000, 100000, 1000000]
for length in lengths:
    for r in radii:
        pars_cylinder = {'radius': r, 'length': length, 'scale': 1, 'background': 0.001}
        Iq_data[length][r] = call_kernel(kernel_cylinder, pars_cylinder)

fig = make_subplots(rows=len(radii), cols=1, vertical_spacing=0.02)

initial_length = lengths[0]
for i, radius in enumerate(radii):
    fig.add_trace(go.Scatter(x=q, y=Iq_data[initial_length][radius], mode='lines', name=f'radius: {radius}'), row=i + 1, col=1)

for i in range(1, len(radii) + 1):
    fig.update_xaxes(type='log', title_text='q (1/A)', row=i, col=1)
    fig.update_yaxes(type='log', title_text='I(q) (1/cm)', row=i, col=1)

fig.update_layout(
    title_text=f"Cylinder with different radius (Length: {initial_length})",
    height=3000,
    width=1200
)

steps = []
for length in lengths:
    step = {
        'method': 'update',
        'label': f'Length: {length}',
        'args': [
            {'y': [Iq_data[length][radius] for radius in radii]},
            {'title': f"Cylinder with different radius (Length: {length})"}
        ]
    }
    steps.append(step)

# Add slider to the layout
fig.update_layout(
    sliders=[{
        'active': 0,
        'pad': {"t": 50},
        'steps': steps
    }]
)

filename = 'cylinder76.html'
plot(fig, filename=filename)


