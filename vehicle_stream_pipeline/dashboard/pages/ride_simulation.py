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


rides_df = pd.read_csv(f"{repo}/data/cleaning/data_cleaned.csv")
rides_df = rides_df[(rides_df["state"] == "completed")]
rides_df["scheduled_to"] = pd.to_datetime(rides_df["scheduled_to"])

# sim_df_small = pd.read_csv(f"{repo}/data/simulated/sim_rides_1808_9.csv")
sim_df_large = pd.read_csv(f"{repo}/data/simulated/sim_rides_500k.csv")
sim_df_large["scheduled_to"] = pd.to_datetime(sim_df_large["scheduled_to"])


# common colour for graphs
color_rides = "forestgreen"
color_rides_sim_s = "lightsteelblue"
color_rides_sim_s_2 = "cornflowerblue"
color_rides_sim_l = "navy"
color_sequence = [color_rides, color_rides_sim_s, color_rides_sim_l]
color_sequence_2 = [color_rides_sim_s, color_rides]


layout = dbc.Container(
    [
        html.H1("Ride Simulation", style={"textAlign": "center"}),
        html.Div(
            [
                dcc.Input(
                    id="input_number_simulated_ride",
                    type="number",
                    value=100,
                )
            ],
            className="d-grid gap-2 col-6 mx-auto",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(id="routes_bar"),
                                    # style={"height": "820px"},
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(id="routes_pie_1"),
                                    # style={"height": "820px"},
                                ),
                                dbc.Col(
                                    dcc.Graph(id="routes_pie_2"),
                                    # style={"height": "820px"},
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(id="arrival_deviation"),
                                    # style={"height": "820px"},
                                ),
                                dbc.Col(
                                    dcc.Graph(id="waiting_time"),
                                    # style={"height": "820px"},
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(id="boarding_time"),
                                    # style={"height": "820px"},
                                ),
                                dbc.Col(
                                    dcc.Graph(id="delay"),
                                    # style={"height": "820px"},
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(id="ride_time"),
                                    # style={"height": "820px"},
                                ),
                                dbc.Col(
                                    dcc.Graph(id="trip_time"),
                                    # style={"height": "820px"},
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(id="density_day"),
                                    # style={"height": "820px"},
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(id="density_week"),
                                    # style={"height": "820px"},
                                ),
                            ]
                        ),
                    ],
                ),
            ]
        ),
    ],
    fluid=True,
)


@callback(
    [
        Output("routes_bar", "figure"),
        Output("routes_pie_1", "figure"),
        Output("routes_pie_2", "figure"),
        Output("arrival_deviation", "figure"),
        Output("waiting_time", "figure"),
        Output("boarding_time", "figure"),
        Output("delay", "figure"),
        Output("ride_time", "figure"),
        Output("trip_time", "figure"),
        Output("density_day", "figure"),
        Output("density_week", "figure"),
    ],
    [
        Input("input_number_simulated_ride", "value"),
    ],
)
def update_charts(number_simulations=100):
    global sim_df_large
    sim_df_large_1 = sim_df_large.sample(number_simulations)
    sim_df_small = sim_df_large_1.sample(100)

    # dataframes for Distplots
    dist_df = utils.transformForDist(rides_df, "Original Rides")
    dist_df_sim_s = utils.transformForDist(sim_df_small, "Simulated Rides small")
    dist_df_sim_l = utils.transformForDist(sim_df_large_1, "Simulated Rides large")

    # dataframe for Boxplot
    boxplot_df = pd.concat([dist_df, dist_df_sim_s, dist_df_sim_l])

    # dataframe for Piechart Route Visualization
    df_value_counts_rides = utils.transformForRoute(dist_df, "Original Rides")
    df_value_counts_sim_s = utils.transformForRoute(
        dist_df_sim_s, "Simulated Rides small"
    )
    df_value_counts_sim_l = utils.transformForRoute(
        dist_df_sim_l, "Simulated Rides large"
    )
    known_route_s = (
        df_value_counts_sim_s["route"]
        .loc[df_value_counts_sim_s["route"].isin(df_value_counts_rides["route"])]
        .count()
    )
    unknown_route_s = (
        df_value_counts_sim_s["route"]
        .loc[~df_value_counts_sim_s["route"].isin(df_value_counts_rides["route"])]
        .count()
    )
    known_route_l = (
        df_value_counts_sim_l["route"]
        .loc[df_value_counts_sim_l["route"].isin(df_value_counts_rides["route"])]
        .count()
    )
    unknown_route_l = (
        df_value_counts_sim_l["route"]
        .loc[~df_value_counts_sim_l["route"].isin(df_value_counts_rides["route"])]
        .count()
    )

    # print(df_value_counts_rides.shape)
    # dataframe for Barchart Route Visualization
    top_df = utils.transformForBar(
        10, df_value_counts_rides, df_value_counts_sim_s, df_value_counts_sim_l
    )
    # figure bar chart for routes
    fig_routes_bar = px.bar(
        data_frame=top_df,
        x="route",
        y="rel_counts",
        color="dataset",
        color_discrete_sequence=color_sequence,
        orientation="v",
        barmode="group",
        title="Top 10 routes frequency of Original Rides",
    )

    # figures for pie charts for routes
    labels1 = ["Known Route", "Unknown Route"]
    values1 = [known_route_s, unknown_route_s]
    labels2 = ["Known Route", "Unknown Route"]
    values2 = [known_route_l, unknown_route_l]
    fig_routes_pie_1 = px.pie(
        values=values1,
        names=labels1,
        color_discrete_sequence=color_sequence_2,
        title="Route selection 9k simulated rides",
    )
    fig_routes_pie_2 = px.pie(
        values=values2,
        names=labels2,
        color_discrete_sequence=color_sequence,
        title="Route selection 600k simulated rides",
    )

    # figures for boxplots of times
    current_attribute = "arrival_deviation"
    fig_box_arrival_deviation = px.box(
        boxplot_df,
        x="dataset",
        y=current_attribute,
        color="dataset",
        color_discrete_sequence=color_sequence,
        title=f"Boxplot {current_attribute}",
    )

    current_attribute = "waiting_time"
    fig_box_waiting_time = px.box(
        boxplot_df,
        x="dataset",
        y=current_attribute,
        color="dataset",
        color_discrete_sequence=color_sequence,
        title=f"Boxplot {current_attribute}",
    )

    current_attribute = "boarding_time"
    fig_box_boarding_time = px.box(
        boxplot_df,
        x="dataset",
        y=current_attribute,
        color="dataset",
        color_discrete_sequence=color_sequence,
        title=f"Boxplot {current_attribute}",
    )

    current_attribute = "delay"
    fig_box_delay = px.box(
        boxplot_df,
        x="dataset",
        y=current_attribute,
        color="dataset",
        color_discrete_sequence=color_sequence,
        title=f"Boxplot {current_attribute}",
    )

    current_attribute = "ride_time"
    fig_box_ride_time = px.box(
        boxplot_df,
        x="dataset",
        y=current_attribute,
        color="dataset",
        color_discrete_sequence=color_sequence,
        title=f"Boxplot {current_attribute}",
    )

    current_attribute = "trip_time"
    fig_box_trip_time = px.box(
        boxplot_df,
        x="dataset",
        y=current_attribute,
        color="dataset",
        color_discrete_sequence=color_sequence,
        title=f"Boxplot {current_attribute}",
    )

    # figures for distplots of rel_frequency over day and week

    group_labels = ["Original Rides", "Simulated Rides small", "Simulated Rides large"]

    current_attribute = "hour"
    hist_data = [
        dist_df[current_attribute],
        dist_df_sim_s[current_attribute],
        dist_df_sim_l[current_attribute],
    ]
    fig_dist_hour = ff.create_distplot(
        hist_data, group_labels, colors=color_sequence, show_rug=False
    )
    fig_dist_hour.update_layout(title="Rides distribution over day")

    current_attribute = "day_of_week"
    hist_data = [
        dist_df[current_attribute],
        dist_df_sim_s[current_attribute],
        dist_df_sim_l[current_attribute],
    ]
    fig_dist_week = ff.create_distplot(
        hist_data, group_labels, colors=color_sequence, show_rug=False
    )
    fig_dist_week.update_layout(title="Rides distribution over week")

    return (
        fig_routes_bar,
        fig_routes_pie_1,
        fig_routes_pie_2,
        fig_box_arrival_deviation,
        fig_box_waiting_time,
        fig_box_boarding_time,
        fig_box_delay,
        fig_box_ride_time,
        fig_box_trip_time,
        fig_dist_hour,
        fig_dist_week,
    )
