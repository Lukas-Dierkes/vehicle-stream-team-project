import math
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
dash.register_page(__name__, path="/")


# fetch data (here we can automate it)
df_stops = pd.read_excel(
    f"{repo}/data/other/MoDstops+Preismodell.xlsx", sheet_name="MoDstops"
)

df_edges = pd.read_excel(
    f"{repo}/data/other/MoDstops+Preismodell.xlsx", sheet_name="Liste 2022"
)

df_edges.rename(columns={"Start #": "start_id", "Ende #": "end_id"}, inplace=True)

rides_df = pd.read_csv(f"{repo}/data/cleaning/data_cleaned.csv")
rides_df = rides_df[(rides_df["state"] == "completed")]
rides_df["scheduled_to"] = pd.to_datetime(rides_df["scheduled_to"])

# simulate high number of rides depending on start and end date of rides_df
start_date = min(rides_df["scheduled_to"])
end_date = max(rides_df["scheduled_to"])

date_range = utils.get_date_range(start_date, end_date)
data_range_len = len(date_range)


sim_rides_all = pd.read_csv(f"{repo}/data/simulated/sim_rides_500k.csv")
sim_rides_all["simulated"] = True  # will be filtered later
sim_rides_all["scheduled_to"] = pd.to_datetime(sim_rides_all["scheduled_to"])

rides_df["simulated"] = False


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
                            options=df_stops["MoDStop Name"],
                            value="Rathaus",
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
                            options=df_stops["MoDStop Name"],
                            value="Hauptbahnhof",
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
        Input("input_number_drones", "value"),
        Input("input_number_simulated_ride", "value"),
    ],
)
# add parameters with default None
def create_geo_graph(
    start_date,
    end_date,
    drones_activated="0",
    pickup_address="Rathaus",
    dropoff_address="Hauptbahnhof",
    radius=500,
    sim_rides=0,
):
    global rides_df
    global df_edges
    global df_stops
    global sim_rides_all

    rides_df_1 = rides_df.copy()
    sim_rides_all_1 = sim_rides_all.copy()
    route_information = ""
    ridetime = "Average time to destination: 77 days"

    pickup_address = utils.find_id_for_name(pickup_address, df_stops)
    dropoff_address = utils.find_id_for_name(dropoff_address, df_stops)

    start_date = dt.strptime(start_date, "%Y-%m-%d")
    end_date = dt.strptime(end_date, "%Y-%m-%d")
    if sim_rides != 0:
        sim_rides_all_1 = sim_rides_all_1[
            (sim_rides_all_1["scheduled_to"] > start_date)
            & (sim_rides_all_1["scheduled_to"] < end_date)
        ]
        if sim_rides > len(sim_rides_all_1):
            sim_rides_sample = sim_rides_all_1.sample(n=sim_rides, replace=True)
        else:
            sim_rides_sample = sim_rides_all_1.sample(n=sim_rides)

        new_rides_all = pd.concat([rides_df_1, sim_rides_sample])

    else:
        new_rides_all = rides_df_1

    rides_df_filterd = new_rides_all[
        (new_rides_all["scheduled_to"] > start_date)
        & (new_rides_all["scheduled_to"] < end_date)
    ]

    # print(len(rides_df_filterd))
    # if default parameters None, do nothing else get shortest ride of function call
    if pickup_address is not None or dropoff_address is not None:

        drives_without_drones = utils.calculate_drives(
            rides_df_filterd, start_date, end_date
        )

        # hotspots = utils.get_hotspots(df_edges, drives_without_drones)
        # hotspots = [spot[0] for spot in hotspots]
        hotspots = [1008, 4025, 1005, 1009, 1007, 12007, 7001, 6004, 1010, 11017]

        drone_spots = [15011, 13001, 2002, 11007, 4016, 1002, 3020, 9019, 9005]

        df_stops_drones = df_stops[df_stops["MoDStop Id"].isin(drone_spots)]

        if drones_activated == "0":
            layers = []

            graph_without_drones = utils.calculate_graph(drives_without_drones)
            path, shortest_time = utils.get_shortest_ride(
                pickup_address, dropoff_address, graph_without_drones
            )
            route_information = utils.get_route_information(
                drives_without_drones, path, df_stops
            )
        else:
            layers = utils.create_circles_around_drone_spots(df_stops_drones, radius)

            drives_with_drones = utils.add_drone_flights(
                df_edges, drives_without_drones, drone_spots=drone_spots, radius=radius
            )
            graph_with_drones = utils.calculate_graph(drives_with_drones)
            path, shortest_time = utils.get_shortest_ride(
                pickup_address, dropoff_address, graph_with_drones
            )
            route_information = utils.get_route_information(
                drives_with_drones, path, df_stops
            )

        ridetime = f"Average time to destination: {round(shortest_time, 2)} days"
        # get lat and lon for traces for each stop in the returend list
        latitudes, longitudes = utils.get_geo_cordinates_for_path(df_stops, path)

    pickup_counts = (
        rides_df_filterd.groupby("pickup_address")
        .size()
        .to_frame("number_of_pickups")
        .reset_index()
    )
    dropoff_counts = (
        rides_df_filterd.groupby("dropoff_address")
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

    if drones_activated == "1":
        df_stops_1["is_drone_spot_color"] = np.where(
            df_stops_1["MoDStop Id"].isin(drone_spots), "purple", "blue"
        )

        df_stops_1["is_drone_spot_size"] = np.where(
            (df_stops_1["MoDStop Id"].isin(hotspots))
            | (df_stops_1["MoDStop Id"].isin(drone_spots)),
            12,
            6,
        )

    else:
        df_stops_1["is_drone_spot_color"] = np.where(
            df_stops_1["MoDStop Id"].isin(drone_spots), "blue", "blue"
        )

        df_stops_1["is_drone_spot_size"] = np.where(
            (df_stops_1["MoDStop Id"].isin(hotspots)), 12, 6
        )

    fig = go.Figure(
        go.Scattermapbox(
            mode="markers",
            lat=df_stops_1["MoDStop Lat"],
            lon=df_stops_1["MoDStop Long"],
            # [df_stops_1['MoDStop Name'], df_stops_1['MoDStop Id']]
            hovertext=df_stops_1["MoDStop Name"],
            hoverinfo="text",
            marker={
                "color": df_stops_1["is_drone_spot_color"],
                "size": df_stops_1["is_drone_spot_size"],
            },
            showlegend=False,
        )
    )
    #
    fig.add_trace(
        go.Scattermapbox(
            lon=longitudes,
            lat=latitudes,
            mode="markers+lines",
            line=dict(width=2, color="red"),
            # [df_stops_1['MoDStop Name'], df_stops_1['MoDStop Id']]
            # hovertext=df_stops_1["MoDStop Name"],
            hoverinfo="skip",
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
