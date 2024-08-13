"""
Minimal example of calling a kernel for a specific set of q values.
"""

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

model_name = 'cylinder_accurate'
try:
    model_cylinder_accurate = load_model(model_name)
except Exception as e:
    print(f"Error: {e}")
    print(f"Model {model_name} not found. Please check if the model exists in sasmodels.")
    raise e

q = np.linspace(0.001, 0.1, 20000)
kernel_cylinder = model_cylinder_accurate.make_kernel([q])

lengths = [0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 10000, 100000, 1000000]
rtols = [0.1, 0.01, 0.001, 0.0001, 0.00001]

# rtols = [0.00001]
Iq_data = {(length, rtol): {} for length in lengths for rtol in rtols}

radii = [0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 10000, 100000, 1000000]
for length in lengths:
    for rtol in rtols:
        for r in radii:
            pars_cylinder = {'radius': r, 'length': length, 'scale': 1, 'background': 0.001, 'rtol': rtol}
            Iq_data[(length, rtol)][r] = call_kernel(kernel_cylinder, pars_cylinder)

for initial_rtol in rtols:
    fig = make_subplots(rows=len(radii), cols=1, vertical_spacing=0.02)
    initial_length = lengths[0]

    for i, radius in enumerate(radii):
        fig.add_trace(
            go.Scatter(x=q, y=Iq_data[(initial_length, initial_rtol)][radius], mode='lines', name=f'radius: {radius}'),
            row=i + 1, col=1)

    # Update layout for the initial plot
    aspect_ratio = 30
    for i in range(1, len(radii) + 1):
        fig.update_xaxes(type='log', title_text='q (1/A)', row=i, col=1)
        fig.update_yaxes(type='log', title_text='I(q) (1/cm)', row=i, col=1)

    fig.update_layout(
        title_text=f"Cylinder with different radius (Length: {initial_length}, Rtol: {initial_rtol})",
        height=3000,
        width=1200
    )

    # Create steps for the length slider
    length_steps = []
    for length in lengths:
        step = {
            'method': 'update',
            'label': f'Length: {length}',
            'args': [
                {'y': [Iq_data[(length, initial_rtol)][radius] for radius in radii]},
                {'title': f"Cylinder with different radius (Length: {length}, Rtol: {initial_rtol})"}
            ]
        }
        length_steps.append(step)

    # Add slider and dropdown to the layout
    fig.update_layout(
        sliders=[{
            'active': 0,
            'pad': {"t": 50},
            'steps': length_steps
        }]
    )
    filename = f"cylinder_plot_rtol_{initial_rtol}.html"
    plot(fig, filename=filename)



