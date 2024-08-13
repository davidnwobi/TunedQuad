import numpy as np
import pandas as pd
import plotly.express as px
import typing as tp
import os

curr_dir = os.path.dirname(__file__)

cylinder_accurate_benchmark_loc = curr_dir + '\cylinder_accurate_benchmark.xlsx'
cylinder76_benchmark_loc = curr_dir + '\cylinder76_benchmark.xlsx'

if not os.path.exists(cylinder_accurate_benchmark_loc):
    import cylinder_benchmark
cylinder_accurate_benchmark_df = pd.read_excel(cylinder_accurate_benchmark_loc, index_col=[0, 1, 2])

if not os.path.exists(cylinder76_benchmark_loc):
    import cylinder76_benchmark
cylinder76_benchmark_df = pd.read_excel(cylinder76_benchmark_loc, index_col=[0, 1])

rtols = [0.1, 0.01, 0.001, 0.0001, 0.00001]

ratios: tp.List[pd.DataFrame] = [(cylinder_accurate_benchmark_df.loc[rtol] / cylinder76_benchmark_df).reset_index().pivot(index='length', columns='radius', values='time') for rtol in rtols]
for i in range(len(ratios)):

    ratios[i] = ratios[i].round(2)
    ratios[i].columns = ratios[i].columns.round(6).astype(str)
    ratios[i].index = ratios[i].index.round(6).astype(str)
    print(ratios[i])


fig = px.imshow(ratios[0], labels=dict(x="Length", y="Radius", color="Time Ratio"), title="Time Ratios for Cylinder Model", text_auto=True)

fig.update_layout(autosize=True)

steps = []
for i, rtol in enumerate(rtols):
    step = {
        'method': 'update',
        'label': f'Rtol: {rtol}',
        'args': [
            {'z': [ratios[i].values]},
            {'title': f"Time Ratios for New Cylinder Model vs Old Model (Rtol: {rtol})"}
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


fig.show()
