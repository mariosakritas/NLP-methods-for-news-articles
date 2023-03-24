import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import functools
import itertools
import operator
import os
import os.path as op
import numpy as np
import pytrends
from pytrends.request import TrendReq as UTrendReq
from datetime import date
import datetime as d
from collections import Counter
import click
import pdb
import sys, traceback

from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px

def get_interest_over_time_plot(google_df):
    fig = px.line(google_df)
    plt.show()
    return fig

# # Opening JSON file
# f = open('CMS_2010_to_June_2022_ENGLISH.json')
  
# # returns JSON object as a dictionary
# data = json.load(f)

# # convert to data frame
# df = pd.DataFrame.from_dict(data)

sys.path.append('..')

from models.MA_dw_to_google_pipeline import get_interest_over_time

app = Dash()

keyword_dropdown = dcc.Dropdown(options=['Germany', 'Donald Trump'], value='Germany')

app.layout = html.Div(children=[
    html.H1(children='Google Search Interest'),
    keyword_dropdown,
    dcc.Graph(id='search-graph')
])

@app.callback(
    Output(component_id='search-graph', component_property='figure'),
    Input(component_id=keyword_dropdown, component_property='value')
)
def update_graph(selected_keyword):
    google_df = get_interest_over_time(selected_keyword, start_date = '2023-01-01', end_date=f'{date.today()}')
    line_fig = get_interest_over_time_plot(google_df)
    return line_fig

# Run local server
app.run_server(debug=True, port=8049, use_reloader=False)