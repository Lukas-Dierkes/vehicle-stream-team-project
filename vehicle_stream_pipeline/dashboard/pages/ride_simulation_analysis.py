from datetime import datetime as dt
from glob import glob

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

rides_df = pd.read_csv(f"{repo}/data/cleaning/data_cleaned.csv")
rides_df = rides_df[(rides_df["state"] == "completed")]
rides_df["scheduled_to"] = pd.to_datetime(rides_df["scheduled_to"])
rides_df["dropoff_at"] = pd.to_datetime(rides_df["dropoff_at"])
rides_df["dispatched_at"] = pd.to_datetime(rides_df["dispatched_at"])


simulated_rides = pd.read_csv(f"{repo}/data/sim_rides_500k.csv")
simulated_rides["scheduled_to"] = pd.to_datetime(simulated_rides["scheduled_to"])
simulated_rides["dropoff_at"] = pd.to_datetime(simulated_rides["dropoff_at"])
simulated_rides["dispatched_at"] = pd.to_datetime(simulated_rides["dispatched_at"])

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
                dbc.Label("Select combined_rides_factor"),
                dcc.Slider(0, 1, 0.1, value=0.3, id="combined_rides_factor"),
            ]
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Label("Average days need for package arrival"),
                dcc.Slider(0, 10, 0.5, value=2.0, id="avg_package_arrival"),
            ]
        ),
        html.Hr(),
        dbc.Row(
            [
                html.Div(
                    dcc.RadioItems(
                        id={
                            "type": "dynmaic-main_routes",
                        },
                        options=[
                            {"label": "Only main routes", "value": "1"},
                            {"label": "All routes", "value": "0"},
                        ],
                        value="0",
                        labelStyle={"display": "block"},
                    )
                )
            ]
        ),
    ]
)


layout = dbc.Container(
    [
        html.H1("Ride Simulation Analysis", style={"textAlign": "center"}),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col([controls], width=3),
                dbc.Col(
                    dcc.Graph(id="fig_drivers"),
                    # style={"height": "820px"},
                ),
                dbc.Col(
                    dcc.Graph(id="fig_drives"),
                    # style={"height": "820px"},
                ),
            ]
        ),
    ]
)


@callback(
    Output("fig_drivers", "figure"),
    Output("fig_drives", "figure"),
    [
        Input("date_calendar", "start_date"),
        Input("date_calendar", "end_date"),
        Input(component_id="combined_rides_factor", component_property="value"),
        Input(component_id="avg_package_arrival", component_property="value"),
        Input(
            component_id={"type": "dynmaic-main_routes"},
            component_property="value",
        ),
    ],
)
# add parameters with default None
def create_dashboard_page(
    start_date,
    end_date,
    combined_rides_factor=0.3,
    avg_package_arrival=2,
    only_main_routes="0",
):
    global rides_df
    global simulated_rides

    start_date = dt.strptime(start_date, "%Y-%m-%d")
    end_date = dt.strptime(end_date, "%Y-%m-%d")

    rides_df_filterd = rides_df[
        (rides_df["scheduled_to"] > start_date) & (rides_df["scheduled_to"] < end_date)
    ]

    hours = list(range(0, 24))
    numb_drivers_per_hour = []
    avg_drives_per_hour = []
    for i in hours:
        numb_drivers_per_hour.append(
            utils.calculate_number_drivers(rides_df_filterd, i)[0]
        )
        avg_drives_per_hour.append(
            utils.calculate_number_drivers(rides_df_filterd, i)[1]
        )

    df_drivers_per_hour = pd.DataFrame(
        list(zip(hours, numb_drivers_per_hour)), columns=["hour", "drivers"]
    )

    df_drives_per_hour = pd.DataFrame(
        list(zip(hours, avg_drives_per_hour)), columns=["hour", "drives"]
    )

    fig_drivers = px.bar(df_drivers_per_hour, x="hour", y="drivers")
    fig_drives = px.bar(df_drives_per_hour, x="hour", y="drives")

    return fig_drivers, fig_drives
