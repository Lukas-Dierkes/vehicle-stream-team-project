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
        html.H1("MoD Stop Analysis", style={"textAlign": "center"}),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col([controls], width=3),
            ]
        ),
    ]
)
