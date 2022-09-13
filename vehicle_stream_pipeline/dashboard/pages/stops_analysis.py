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

from vehicle_stream_pipeline.utils import prob_model as pm
from vehicle_stream_pipeline.utils import ride_simulation as rs

repo = git.Repo(".", search_parent_directories=True).git.rev_parse("--show-toplevel")

# Code from: https://github.com/plotly/dash-labs/tree/main/docs/demos/multi_page_example1
dash.register_page(__name__, path="/")


# Read in stops data
df_stops = pd.read_excel(
    f"{repo}/data/other/MoDstops+Preismodell.xlsx", sheet_name="MoDstops"
)

# Read in all stop combinations
df_edges = pd.read_excel(
    f"{repo}/data/other/MoDstops+Preismodell.xlsx", sheet_name="Liste 2022"
)

df_edges.rename(columns={"Start #": "start_id", "Ende #": "end_id"}, inplace=True)

# Read in cleaned rides
rides_df = pd.read_csv(f"{repo}/data/cleaning/data_cleaned.csv")
rides_df = rides_df[(rides_df["state"] == "completed")]
rides_df["scheduled_to"] = pd.to_datetime(rides_df["scheduled_to"])

# simulate high number of rides depending on start and end date of rides_df
start_date = min(rides_df["scheduled_to"])
end_date = max(rides_df["scheduled_to"])

date_range = rs.get_date_range(start_date, end_date)
data_range_len = len(date_range)

# Read in simulated rides
sim_rides_all = pd.read_csv(f"{repo}/data/simulated/ride_simulation.csv")
sim_rides_all = sim_rides_all.sample(15000)

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
                    start_date=dt(2021, 8, 1).date(),
                    end_date=dt(2022, 6, 30).date(),
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
        dbc.Row(
            [
                html.Div(
                    dcc.RadioItems(
                        id={
                            "type": "dynmaic-dpn-main-spots",
                        },
                        options=[
                            {"label": "Show main spots", "value": "1"},
                            {"label": "Show hotspots", "value": "0"},
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
        Input(
            component_id={"type": "dynmaic-dpn-main-spots"},
            component_property="value",
        ),
    ],
)
def create_geo_graph(
    start_date,
    end_date,
    drones_activated="0",
    pickup_address="Rathaus",
    dropoff_address="Hauptbahnhof",
    radius=500,
    sim_rides=0,
    main_spots="0",
):
    # Use dataframes from global name space so you don't need to read them every time you cange an input
    global rides_df
    global df_edges
    global df_stops
    global sim_rides_all

    # Copy data so it won't be overwritten by following manipulations
    rides_df_1 = rides_df.copy()
    sim_rides_all_1 = sim_rides_all.copy()

    # Set placeholder for routeinformation output
    route_information = ""
    ridetime = "Average time to destination: 77 days"

    # Find id for the dropoff and pickupp address
    pickup_address = pm.find_id_for_name(pickup_address, df_stops)
    dropoff_address = pm.find_id_for_name(dropoff_address, df_stops)

    # Convert start_date and end_date input to timestamps
    start_date = dt.strptime(start_date, "%Y-%m-%d")
    end_date = dt.strptime(end_date, "%Y-%m-%d")

    # Check whether simulated rides should be added or not
    if sim_rides != 0:
        sim_rides_all_1 = sim_rides_all_1[
            (sim_rides_all_1["scheduled_to"] > start_date)
            & (sim_rides_all_1["scheduled_to"] < end_date)
        ]
        # If number of needed simulation rides is bigger than our sample then use sample with replacement
        if sim_rides > len(sim_rides_all_1):
            sim_rides_sample = sim_rides_all_1.sample(n=sim_rides, replace=True)
        else:
            sim_rides_sample = sim_rides_all_1.sample(n=sim_rides)

        # Combine original rides with simulated rides
        new_rides_all = pd.concat([rides_df_1, sim_rides_sample])

    else:
        # Only original rides
        new_rides_all = rides_df_1

    # Filter data for given start_date and end_date
    rides_df_filterd = new_rides_all[
        (new_rides_all["scheduled_to"] > start_date)
        & (new_rides_all["scheduled_to"] < end_date)
    ]

    # if default parameters None, do nothing else get shortest ride of function call
    if pickup_address is not None or dropoff_address is not None:

        drives_without_drones = pm.calculate_drives(
            rides_df_filterd, start_date, end_date
        )

        # Check whether hotsspots or main_spots should be shown in the graph
        if main_spots == "1":
            # Set main_spots and drone_spots (both are same for this case)
            hotspots = [
                1008,
                4025,
                6004,
                12007,
                11017,
                15013,
                3021,
                8001,
                5001,
                11003,
                4016,
            ]

            # Manually define drone spots
            drone_spots = []
        else:
            # Hotspots are hardcoded because calculation takes to long for dashbaord updating
            hotspots = [1008, 4025, 1005, 1009, 1007, 12007, 7001, 6004, 1010, 11017]

            # Manually define drone spots
            drone_spots = [15011, 13001, 2002, 11007, 4016, 1002, 3020, 9019, 9005]

        # Filter stops for only drone stops
        df_stops_drones = df_stops[df_stops["MoDStop Id"].isin(drone_spots)]

        # Check whether drone flights should be added or not
        if drones_activated == "0":
            layers = []

            # Calculate graph without drones
            graph_without_drones = pm.calculate_graph(drives_without_drones)

            # Calculate shortest path for given input pickup address and dropoff address
            path, shortest_time = pm.get_shortest_ride(
                pickup_address, dropoff_address, graph_without_drones
            )

            # Get the route information for the shortest path
            route_information = pm.get_route_information(
                drives_without_drones, path, df_stops
            )
        else:
            # Create circle layer for the graph which show the radius of drones
            layers = pm.create_circles_around_drone_spots(df_stops_drones, radius)

            # Add drone flights
            drives_with_drones = pm.add_drone_flights(
                df_edges, drives_without_drones, drone_spots=drone_spots, radius=radius
            )

            # Calculate graph with drones flights included
            graph_with_drones = pm.calculate_graph(drives_with_drones)

            # Calculate shortest path with drone flights
            path, shortest_time = pm.get_shortest_ride(
                pickup_address, dropoff_address, graph_with_drones
            )

            # Get route information for shortest path
            route_information = pm.get_route_information(
                drives_with_drones, path, df_stops
            )

        ridetime = f"Average time to destination: {round(shortest_time, 2)} days"
        # get lat and lon for traces for each stop in the returend list

        # Get geo coordinates for each stop in the graph
        latitudes, longitudes = pm.get_geo_cordinates_for_path(df_stops, path)

    # Calculate the number of pickups for each spot
    pickup_counts = (
        rides_df_filterd.groupby("pickup_address")
        .size()
        .to_frame("number_of_pickups")
        .reset_index()
    )

    # Calculate the number of dropoffs for each spot
    dropoff_counts = (
        rides_df_filterd.groupby("dropoff_address")
        .size()
        .to_frame("number_of_dropoffs")
        .reset_index()
    )

    # Convert counts to integer
    pickup_counts["pickup_address"] = pickup_counts["pickup_address"].astype(int)
    dropoff_counts["dropoff_address"] = dropoff_counts["dropoff_address"].astype(int)

    # Join the pickup counts and dropff counts to the stop dataframe
    df_stops_1 = pd.merge(
        df_stops, pickup_counts, left_on="MoDStop Id", right_on="pickup_address"
    ).drop("pickup_address", axis=1)
    df_stops_1 = pd.merge(
        df_stops_1, dropoff_counts, left_on="MoDStop Id", right_on="dropoff_address"
    ).drop("dropoff_address", axis=1)

    # Coloring of drone spots in the graph so we can distinguish between them in the graph
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

    # Create Map with spots and shortest path
    fig = go.Figure(
        go.Scattermapbox(
            mode="markers",
            lat=df_stops_1["MoDStop Lat"],
            lon=df_stops_1["MoDStop Long"],
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
