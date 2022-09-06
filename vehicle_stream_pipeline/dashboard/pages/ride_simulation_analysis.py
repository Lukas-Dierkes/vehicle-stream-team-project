from datetime import datetime as dt

import dash
import dash_bootstrap_components as dbc
import git
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from dash import MATCH, Dash, Input, Output, callback, dcc, html

from vehicle_stream_pipeline import utils

repo = git.Repo(".", search_parent_directories=True).git.rev_parse("--show-toplevel")
graph_metrics_df = pd.read_csv(f"{repo}/data/regression/graph_metrics_5Ksteps.csv")

# Code from: https://github.com/plotly/dash-labs/tree/main/docs/demos/multi_page_example1
dash.register_page(__name__)


controls = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Label("Pick date range"),
                dcc.DatePickerRange(
                    id="date_calendar",  # ID to be used for callback
                    calendar_orientation="horizontal",  # vertical or horizontal
                    day_size=39,  # size of calendar image. Default is 39
                    # text that appears when no end date chosen
                    end_date_placeholder_text="End-date",
                    with_portal=False,  # if True calendar will open in a full screen overlay portal
                    # Display of calendar when open (0 = Sunday)
                    first_day_of_week=0,
                    reopen_calendar_on_clear=True,
                    is_RTL=False,  # True or False for direction of calendar
                    clearable=True,  # whether or not the user can clear the dropdown
                    number_of_months_shown=1,  # number of months shown when calendar is open
                    min_date_allowed=dt(
                        2021, 1, 1
                    ),  # minimum date allowed on the DatePickerRange component
                    max_date_allowed=dt(
                        2022, 12, 31
                    ),  # maximum date allowed on the DatePickerRange component
                    initial_visible_month=dt(
                        2021, 12, 1
                    ),  # the month initially presented when the user opens the calendar
                    start_date=dt(2022, 2, 1).date(),
                    end_date=dt(2022, 2, 28).date(),
                    # how selected dates are displayed in the DatePickerRange component.
                    display_format="MMM Do, YY",
                    # how calendar headers are displayed when the calendar is opened.
                    month_format="MMMM, YYYY",
                    minimum_nights=0,  # minimum number of days between start and end date
                    persistence=True,
                    persisted_props=["start_date"],
                    persistence_type="session",  # session, local, or memory. Default is 'local'
                    # singledate or bothdates. Determines when callback is triggered
                    updatemode="singledate",
                ),
            ],
        ),
        html.Hr(),
        dbc.Row(
            [
                html.Div(
                    dcc.RadioItems(
                        id={
                            "type": "dynmaic-dpn-drones",
                        },
                        options=[
                            {"label": "Drones", "value": "1"},
                            {"label": "No Drones", "value": "0"},
                        ],
                        value="0",
                        labelStyle={"display": "block"},
                    )
                )
            ]
        ),
        html.Hr(),
        dbc.Row(
            [
                html.Label("Enter drone radius (meter)"),
                html.Div(
                    [
                        dcc.Input(
                            id="input_number_drones",
                            type="number",
                            value=500,
                        )
                    ]
                ),
            ]
        ),
        dbc.Row(
            [
                html.Label("Add simulated rides"),
                html.Div(
                    [
                        dcc.Input(
                            id="input_number_simulated_ride",
                            type="number",
                            value=0,
                        )
                    ]
                ),
            ]
        ),
    ]
)


layout = dbc.Container(
    [
        html.H1("Ride Simulation Analysis", style={"textAlign": "center"}),
        html.Div(
            [
                html.Label("Maximum Delivery Days"),
                dcc.Input(
                    id="input_number_max_days",
                    type="number",
                    value=5,
                ),
            ],
            className="d-grid gap-2 col-6 mx-auto",
        ),
        html.Div(
            [
                html.Label("Graph Metric"),
                dcc.Dropdown(
                    id="dropdown_graph_metric",
                    options=[
                        {
                            "label": "Maximum Delivery Days without Drones",
                            "value": "diameter_w/o_drones",
                        },
                        {
                            "label": "Maximum Delivery Days with Drones",
                            "value": "diameter_with_drones",
                        },
                        {
                            "label": "Average Delivery Days without Drones",
                            "value": "avg_w/o_drones",
                        },
                        {
                            "label": "Average Delivery Days with Drones",
                            "value": "avg_with_drones",
                        },
                    ],
                    value="avg_w/o_drones",
                    clearable=False,
                ),
            ],
            className="d-grid gap-2 col-6 mx-auto",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id="needed_rides_plot"),
                    # style={"height": "820px"},
                ),
            ]
        ),
    ]
)


@callback(
    [
        Output("needed_rides_plot", "figure"),
    ],
    [
        Input("input_number_max_days", "value"),
        Input("dropdown_graph_metric", "value"),
    ],
)
def update_charts(max_days=5, current_metric="avg_w/o_drones"):

    global graph_metrics_df

    graph_metrics_df_1 = graph_metrics_df.copy()
    max_days = max_days
    current_metric = current_metric
    needed_rides = utils.get_rides_num(max_days, graph_metrics_df_1, current_metric)

    # Update Figure diplaying Orginal Data Scatter Plot, Regression and max_days_line
    # Scatter Plot of Orginal Rides Data
    needed_rides_fig1 = px.scatter(
        graph_metrics_df_1,
        x=current_metric,
        y="#_simulated_rides",
        color_discrete_sequence=["DarkKhaki"],
        title="Break Even of Rides",
    )
    needed_rides_fig1["data"][0]["name"] = "Simulated Rides Data"
    needed_rides_fig1["data"][0]["showlegend"] = True
    # Line Plot of Regressed Data
    needed_rides_fig2 = px.line(
        x=graph_metrics_df_1[current_metric],
        y=utils.regression_function(
            graph_metrics_df_1[current_metric],
            *utils.get_opt_parameter(graph_metrics_df_1, current_metric),
        ),
        color_discrete_sequence=["DarkCyan"],
    )
    needed_rides_fig2["data"][0]["name"] = "Regression of Rides Data"
    needed_rides_fig2["data"][0]["showlegend"] = True
    # Line Plot working as cursor for current max days
    needed_rides_fig3 = px.line(
        x=[max_days, max_days], y=[0, needed_rides], color_discrete_sequence=["tomato"]
    )
    needed_rides_fig3["data"][0]["name"] = "Max Days for Delivery"
    needed_rides_fig3["data"][0]["showlegend"] = True
    needed_rides_fig4 = px.line(
        x=[0, max_days],
        y=[needed_rides, needed_rides],
        color_discrete_sequence=["tomato"],
    )

    needed_rides_fig = go.Figure(
        data=needed_rides_fig1.data
        + needed_rides_fig2.data
        + needed_rides_fig3.data
        + needed_rides_fig4.data,
        layout=needed_rides_fig1.layout,
    )

    return [
        needed_rides_fig,
    ]  # output fig needs to be list in order work
