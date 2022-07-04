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
df_stops = pd.read_excel(
    f"{repo}/data/other/MoDstops+Preismodell.xlsx", sheet_name="MoDstops"
)

df_edges = pd.read_excel(
    f"{repo}/data/other/MoDstops+Preismodell.xlsx", sheet_name="Liste 2022"
)

rides_df = pd.read_csv(f"{repo}/data/cleaning/data_cleaned.csv")
rides_df = rides_df[(rides_df["state"] == "completed")]

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
                    end_date=dt(2022, 8, 31).date(),
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
        # insert dbc row with two columns for input text fields
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Pickup address"),
                        dcc.Dropdown(
                            id={
                                "type": "dynmaic-dpn-pickup_address",
                            },
                            options=rides_df["pickup_address"],
                            value=1001,
                            clearable=False,
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        html.Label("Dropoff address"),
                        dcc.Dropdown(
                            id={
                                "type": "dynmaic-dpn-dropoff_address",
                            },
                            options=rides_df["pickup_address"],
                            value=10001,
                            clearable=False,
                        ),
                    ]
                ),
            ]
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
                dcc.Input(
                    id="input_number",
                    type="number",
                    value=500,
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
                                    # style={"height": "820px"},
                                ),
                            ]
                        )
                    ]
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [html.Div(id="ridetime", style={"whiteSpace": "pre-line"})]
                        ),
                        html.Hr(),
                        dbc.Row(
                            [
                                html.Div(
                                    id="route_information",
                                    style={"whiteSpace": "pre-line"},
                                )
                            ]
                        ),
                    ],
                    width=3,
                ),
            ]
        ),
    ],
    fluid=True,
)


@callback(
    [
        Output("geo_stops", "figure"),
        Output("ridetime", "children"),
        Output("route_information", "children"),
    ],
    [
        Input("date_calendar", "start_date"),
        Input("date_calendar", "end_date"),
        Input(
            component_id={"type": "dynmaic-dpn-drones"},
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
        Input("input_number", "value"),
    ],
)
# add parameters with default None
def create_geo_graph(
    start_date,
    end_date,
    drones_activated="0",
    pickup_address=1001,
    dropoff_address=10001,
    radius=500,
):
    global rides_df
    global df_edges
    global df_stops
    route_information = ""
    ridetime = "Average time to destination: 77 days"

    start_date = dt.strptime(start_date, "%Y-%m-%d")
    end_date = dt.strptime(end_date, "%Y-%m-%d")

    rides_df = rides_df[(rides_df["state"] == "completed")]
    rides_df["scheduled_to"] = pd.to_datetime(rides_df["scheduled_to"])

    rides_df = rides_df[
        (rides_df["scheduled_to"] > start_date) & (rides_df["scheduled_to"] < end_date)
    ]

    df_edges.rename(columns={"Start #": "start_id", "Ende #": "end_id"}, inplace=True)

    # if default parameters None, do nothing else get shortest ride of function call
    if pickup_address is not None or dropoff_address is not None:

        drives = utils.calculate_drives(rides_df, start_date, end_date)

        # hotspots = utils.get_hotspots(df_edges, aggregated_drives)
        # hotspots = [spot[0] for spot in hotspots]
        hotspots = [4025, 1009, 11017, 7001, 6002, 12007]

        df_stops_hotspots = df_stops[df_stops["MoDStop Id"].isin(hotspots)]
        if drones_activated == "0":
            layers = []
        else:
            layers = utils.create_circles_around_drone_spots(df_stops_hotspots, radius)

        if drones_activated == "1":
            drives = utils.add_drone_flights(
                df_edges, drives, drone_spots=hotspots, radius=radius
            )
            graph = utils.calculate_graph(drives)
        else:
            graph = utils.calculate_graph(drives)

        path, shortest_time = utils.get_shortest_ride(
            pickup_address, dropoff_address, graph
        )

        route_information = utils.get_route_information(drives, path)

        ridetime = f"Average time to destination: {round(shortest_time, 2)} days"
        # get lat and lon for traces for each stop in the returend list
        latitudes, longitudes = utils.get_geo_cordinates_for_path(df_stops, path)

    pickup_counts = (
        rides_df.groupby("pickup_address")
        .size()
        .to_frame("number_of_pickups")
        .reset_index()
    )
    dropoff_counts = (
        rides_df.groupby("dropoff_address")
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

    df_stops_1["is_drone_spot"] = np.where(
        df_stops_1["MoDStop Id"].isin(hotspots), "orange", "blue"
    )
    fig = go.Figure(
        go.Scattermapbox(
            mode="markers",
            lat=df_stops_1["MoDStop Lat"],
            lon=df_stops_1["MoDStop Long"],
            # [df_stops_1['MoDStop Name'], df_stops_1['MoDStop Id']]
            hovertext=df_stops_1["MoDStop Name"],
            hoverinfo="text",
            marker=dict(color=df_stops_1["is_drone_spot"]),
            showlegend=False,
        )
    )
    #
    fig.add_trace(
        go.Scattermapbox(
            lon=longitudes,
            lat=latitudes,
            mode="markers+lines",
            line=dict(width=1, color="red"),
            # [df_stops_1['MoDStop Name'], df_stops_1['MoDStop Id']]
            hovertext=df_stops_1["MoDStop Name"],
            hoverinfo="text",
            opacity=1,
            showlegend=False,
        )
    )
    fig = fig.update_layout(
        mapbox_layers=layers,
        mapbox_style="open-street-map",
        mapbox_zoom=11,
        mapbox_center={"lat": 49.3517, "lon": 8.13664},
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=20, r=20, t=20, b=20),
    )

    return fig, ridetime, route_information
