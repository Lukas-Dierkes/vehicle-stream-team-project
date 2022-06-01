from datetime import datetime as dt
from tkinter import CENTER

import dash
import dash_bootstrap_components as dbc
import git
import pandas as pd
import plotly.express as px
from dash import MATCH, Dash, Input, Output, callback, dcc, html

from vehicle_stream_pipeline.connect_to_mysql import Database
from vehicle_stream_pipeline.utils import get_shortest_ride

# Code from: https://github.com/plotly/dash-labs/tree/main/docs/demos/multi_page_example1
dash.register_page(__name__)


# fetch data (here we can automate it)
rides_df = Database.query_dataframe(
    "rides", "Select * From rides where state = 'completed'"
)
df_stops = Database.query_dataframe("mod_stops")

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
                    start_date=dt(2020, 8, 1).date(),
                    end_date=dt(2022, 3, 31).date(),
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
        dbc.Row(
            [
                dbc.Label("Choose feature for circle size"),
                dcc.Dropdown(
                    id={
                        "type": "dynamic-dpn-ctg",
                    },
                    options=[
                        {"label": c, "value": c}
                        for c in ["number_of_pickups", "number_of_dropoffs"]
                    ],
                    value="number_of_pickups",
                    clearable=False,
                ),
            ]
        ),
        # insert dbc row with two columns for input text fields
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Dropdown(
                            id={
                                "type": "dynmaic-dpn-pickup_address",
                            },
                            options=rides_df["pickup_address"],
                            clearable=False,
                        )
                    ]
                ),
                dbc.Col(
                    [
                        dcc.Dropdown(
                            id={
                                "type": "dynmaic-dpn-dropoff_address",
                            },
                            options=rides_df["pickup_address"],
                            clearable=False,
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
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(id="geo_stops"),
                                    style={"height": "820px"},
                                ),
                            ]
                        )
                    ]
                ),
            ]
        ),
    ],
    fluid=True,
)


@callback(
    Output("geo_stops", "figure"),
    [
        Input("date_calendar", "start_date"),
        Input("date_calendar", "end_date"),
        # insert input start and endpoint
        Input(
            component_id={"type": "dynamic-dpn-ctg"},
            component_property="value",
        ),
        Input(
            component_id={"type": "dynmaic-dpn-pickup_address"},
            component_property="value",
        ),
        Input(
            component_id={"type": "dynmaic-dpn-dropoff_address"},
            component_property="value",
        ),
    ],
)
# add parameters with default None
def create_geo_graph(
    start_date, end_date, ctg_value, pickup_address=None, dropoff_address=None
):
    df_routes = df_routes[
        (df_routes["scheduled_to"] > start_date)
        & (df_routes["scheduled_to"] < end_date)
    ]

    # if default parameters None, do nothing else get shortest ride of function call
    if pickup_address is not None or dropoff_address is not None:
        path, length = get_shortest_ride(
            df_routes, pickup_address, dropoff_address, start_date, end_date
        )

    # get lat and lon for traces for each stop in the returend list
    for stop in path:
        pass

    pickup_counts = (
        df_routes.groupby("pickup_address")
        .size()
        .to_frame("number_of_pickups")
        .reset_index()
    )
    dropoff_counts = (
        df_routes.groupby("dropoff_address")
        .size()
        .to_frame("number_of_dropoffs")
        .reset_index()
    )

    pickup_counts["pickup_address"] = pickup_counts["pickup_address"].astype(int)
    dropoff_counts["dropoff_address"] = dropoff_counts["dropoff_address"].astype(int)

    df_stops_1 = pd.merge(
        df_stops, pickup_counts, left_on="MoDStop Id", right_on="pickup_address"
    ).drop("pickup_address", axis=1)
    df_stops_1 = pd.merge(
        df_stops_1, dropoff_counts, left_on="MoDStop Id", right_on="dropoff_address"
    ).drop("dropoff_address", axis=1)

    fig = px.scatter_mapbox(
        df_stops_1,
        lat="MoDStop Lat",
        lon="MoDStop Long",
        hover_name="MoDStop Adresse",
        hover_data=["number_of_pickups", "number_of_dropoffs"],
        color_discrete_sequence=["fuchsia"],
        size=ctg_value,
        zoom=13,
    )
    #
    fig = fig.update_layout(mapbox_style="open-street-map")
    fig = fig.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=20, r=20, t=20, b=20),
    )

    return fig
