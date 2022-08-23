from datetime import datetime as dt

import dash
import dash_bootstrap_components as dbc
import git
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import MATCH, Dash, Input, Output, callback, dcc, html

from vehicle_stream_pipeline import utils

repo = git.Repo(".", search_parent_directories=True).git.rev_parse("--show-toplevel")

# Code from: https://github.com/plotly/dash-labs/tree/main/docs/demos/multi_page_example1
dash.register_page(__name__)

# fetch data (here we can automate it)
# df_stops = pd.read_excel(
#     f"{repo}/data/other/MoDstops+Preismodell.xlsx", sheet_name="MoDstops"
# )

# df_edges = pd.read_excel(
#     f"{repo}/data/other/MoDstops+Preismodell.xlsx", sheet_name="Liste 2022"
# )

# df_edges.rename(columns={"Start #": "start_id", "Ende #": "end_id"}, inplace=True)

# rides_df = pd.read_csv(f"{repo}/data/cleaning/data_cleaned.csv")
rides_df = pd.read_excel(f"{repo}/data/cleaning/data_cleaned_1808.xlsx")
rides_df = rides_df[(rides_df["state"] == "completed")]
rides_df["scheduled_to"] = pd.to_datetime(rides_df["scheduled_to"])

sim_df_small = pd.read_csv(f"{repo}/data/sim_rides_1808_9.csv")
sim_df_large = pd.read_csv(f"{repo}/data/sim_rides_600k.csv")

# dataframes for Distplots
dist_df = utils.transformForDist(rides_df, "Original Rides")
dist_df_sim_s = utils.transformForDist(sim_df_small, "Simulated Rides small")
dist_df_sim_l = utils.transformForDist(sim_df_large, "Simulated Rides large")

# dataframe for Boxplot
boxplot_df = pd.concat([dist_df, dist_df_sim_s, dist_df_sim_l])

# dataframe for Piechart Route Visualization
df_value_counts_rides = utils.transformForRoute(dist_df, "Original Rides")
df_value_counts_sim_s = utils.transformForRoute(dist_df_sim_s, "Simulated Rides small")
df_value_counts_sim_l = utils.transformForRoute(dist_df_sim_l, "Simulated Rides large")

print(df_value_counts_rides.shape)
# dataframe for Barchart Route Visualization
top_df = utils.transformForBar(
    10, df_value_counts_rides, df_value_counts_sim_s, df_value_counts_sim_l
)

# common colour for graphs
color_rides = "forestgreen"
color_rides_sim_s = "lightsteelblue"
color_rides_sim_s_2 = "cornflowerblue"
color_rides_sim_l = "navy"


layout = dbc.Container(
    [
        html.H1("Ride Simulation", style={"textAlign": "center"}),
        dcc.Dropdown(
            id="dropdown",
            options={"label": "lavel", "value": "vaslue"},
            value="x",
            clearable=False,
        ),
        dcc.Graph(id="bar-chart"),
    ]
)


@callback(Output("bar-chart", "figure"), Input("dropdown", "value"))
def update_bar_chart(value):

    fig = px.bar(
        top_df, x="route", y=["rel_counts", "dataset"], barmode="group", orientation="v"
    )
    return fig
