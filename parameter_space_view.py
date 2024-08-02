import plotly
import plotly.express as px
from kronrod import kronrod_points_dict
from tuned_quad import RegisteredParametersDictType, ParametersDictType
from fixed_quad import fixed_quad_init, fixed_quad_integrate
from numba.typed import Dict
from numba import njit, vectorize, float64, prange
from j1 import j1
import numpy as np
import plotly.graph_objects as go
import pandas as pd

a = 0
b = np.pi/2
n_test = 40 # Number of Kronrod points to test. Please keep this number even
n = 10 # Dimension of the parameter space so A and B produce an n^2 grid

@vectorize([float64(float64)])
def J1x_nb(x):
    return j1(x)/x if x != 0 else 0.5


@njit
def integrand_2param(
    x, 
    params):
    A = params['A']
    B = params['B']

    return (np.sinc(A * np.cos(x)/np.pi) * J1x_nb(B * np.sin(x)))**2*np.sin(x)

# @njit(parallel=True)
def compute_param_space(integrator, integrand, a, b, Ag, Bg, Valg):
    
    for i in prange(len(Ag)):
        params = Dict.empty(*ParametersDictType)
        params['A'] = Ag[i]
        params['B'] = Bg[i]
        Valg[i] = fixed_quad_integrate(integrator, integrand, a, b, params)


param_space = Dict.empty(*RegisteredParametersDictType)
param_space['A'] = np.geomspace(1, 1000000, n)
param_space['B'] = np.geomspace(1, 1000000, n)
Ag, Bg = np.meshgrid(param_space['A'], param_space['B'], indexing='ij')

Ag = Ag.ravel()
Bg = Bg.ravel()

kronrod_set = np.array(list(kronrod_points_dict.keys()))
kronrod_set1 = kronrod_set[kronrod_set < 1000]
kronrod_set2 = kronrod_set[kronrod_set >= 1000]

mask1 = np.geomspace(1, len(kronrod_set1)-1, n_test//2, dtype=int)
mask2 = np.geomspace(1, len(kronrod_set2)-1, n_test//2, dtype=int)

kronrod_set1 = kronrod_set1[mask1]
kronrod_set2 = kronrod_set2[mask2]

kronrod_set = np.concatenate([kronrod_set1, kronrod_set2])

ValgL = np.empty((n_test, len(Ag)))
for i in range(n_test):
    Valg = np.empty_like(Ag)
    n_kronrod = kronrod_set[i]
    integrator = fixed_quad_init(n_kronrod, kronrod_points_dict)
    compute_param_space(integrator, integrand_2param, a, b, Ag, Bg, Valg)
    pd.DataFrame(Valg.reshape((n,n)), columns=param_space['B'], index=param_space['A']).to_csv(f"Integral_{n_kronrod}_Sampling_Points.csv")
    Valg = np.log10(Valg)
    ValgL[i] = Valg



Ag = Ag.reshape((n, n))
Bg = Bg.reshape((n, n))
ValgL = ValgL.reshape((n_test, n, n))

global_min = np.min(ValgL)
global_max = np.max(ValgL)

fig = go.Figure()

x = Bg[0].round(2).astype(str)
y = Ag[:, 0].round(2).astype(str)

# Add the initial frame
fig.add_trace(go.Heatmap(
    z=ValgL[0],
    x=x,
    y=y,
    colorscale='Viridis',
    zmin=global_min,
    zmax=global_max
))

# Add frames for each row in ValgL
for i in range(1, n_test):
    fig.add_trace(go.Heatmap(
        z=ValgL[i],
        x=x,
        y=y,
        colorscale='Viridis',
        visible=False,
        zmin=global_min,
        zmax=global_max
    ))

# Create the steps for the slider
steps = []
for i in range(n_test):
    step = dict(
        method="update",
        args=[{"visible": [False] * n_test},
              {"title": f"Value of Log10[Integral] using No. Sampling Points Points = {kronrod_set[i]}, Cylinder Model"}],
        label=f"{kronrod_set[i]}"
    )
    step["args"][0]["visible"][i] = True
    steps.append(step)

# Create sliders
sliders = [dict(
    active=0,
    currentvalue={"prefix": "No. Sampling Points: "},
    pad={"t": 50},
    steps=steps
)]

# Update layout with sliders
fig.update_layout(
    sliders=sliders,
    xaxis=dict(title='B'),
    yaxis=dict(title='A'),
    coloraxis=dict(cmin=global_min, cmax=global_max)
)

ValgL = np.transpose(ValgL, (1, 0, 2)) # Transpose to (n, n_test, n) so we can plot graphs showing how the values change with the number of Kronrod points

global_min = np.log10(np.abs(ValgL)).min()-.01
global_max = np.log10(np.abs(ValgL)).max()+.01

df = pd.DataFrame(ValgL[0], columns=param_space['B'].round(2).astype(str), index=kronrod_set)
df = df.abs()
df = np.log10(df)
fig2 = px.line(df, x=df.index, y=df.columns, title=f"Log10[Abs[Log10[Integral]]] vs Number of Sampling Points for A = {param_space['A'][0]}, Cylinder Model", markers=True)
fig2.update_layout(xaxis_title="Number of Sampling Points (Log Scaled)", yaxis_title="Log10[Abs[Log10[Integral]]]", yaxis_range=[global_min, global_max])
# fig2.update_yaxes(type="log")
fig2.update_xaxes(type="log")
for i, col in enumerate(df.columns):
    fig2.data[i].name = f'B = {col}'
steps = []
for i in range(ValgL.shape[0]):
    df = pd.DataFrame(ValgL[i], columns=param_space['B'].round(2).astype(str), index=kronrod_set)
    df = df.abs()
    df = np.log10(df)
    step = dict(
        method="update",
        args=[
            {"y": [df[col].values for col in df.columns]},
            {"title": f"Log10[Abs[Log10[Integral]]] vs Number of Sampling Points for A = {param_space['A'][i]}"}
        ],
        label=f"{np.round(param_space['A'][i],2)}"
    )
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "A = "},
    pad={"t": 50},
    steps=steps
)]

fig2.update_layout(
    sliders=sliders
)

fig2.show()
fig.show()

import plotly

plotly.offline.plot(fig, filename='Variation_with_sampling_points.html')
plotly.offline.plot(fig2, filename='Convergence_with_sampling_points.html')







