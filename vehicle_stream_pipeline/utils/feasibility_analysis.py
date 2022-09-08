# noqa
import collections
import json
import os
import time
import warnings
from datetime import datetime as dt

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats as stats
import shapely.geometry
from scipy.optimize import curve_fit

from vehicle_stream_pipeline.utils.prob_model import (
    add_drone_flights,
    calculate_drives,
    calculate_graph,
)


def getDeliveryTimes(
    rides_simulated, df_edges, month_diff, drone_radius=500, only_main_routes=False
):
    """Takes an input dateframe of rides (simulated or not simulated), transforms the data into graphs and applies diameter and average shortest path calculations.
        The output is used for regression of diameter and average shortest path
    Args:
        rides_simulated (DataFrame): DataFrame containing Rides in format provided by MoD
        df_edges (DataFrame): DataFrame containing stop combinations in format provided by MoD
        month_diff (int): No of original data, to divide # of rides
        drone_radius (int, optional): Radius in meter where drone can connect stops directly. Defaults to 500.
    Returns:
        List: A list with the values ["#_simulated_rides", "diameter_w/o_drones", "avg_w/o_drones", "diameter_with_drones", "avg_with_drones"] for the given dataframe
    """

    rides_simulated["scheduled_to"] = pd.to_datetime(rides_simulated["scheduled_to"])
    start_date = rides_simulated["scheduled_to"].min()
    end_date = rides_simulated["scheduled_to"].max()

    # rides without drones: calculate graph, diameter and average_shortest_path
    drives_without_drones = calculate_drives(rides_simulated, start_date, end_date)
    graph_without_drones = calculate_graph(drives_without_drones)
    # graph needs to be strongly connected to calcutate diameter
    if nx.is_strongly_connected(graph_without_drones):
        diameter = nx.diameter(graph_without_drones, weight="avg_time_to_destination")
    else:
        diameter = 0

    if nx.is_weakly_connected(graph_without_drones):
        avg = nx.average_shortest_path_length(
            graph_without_drones, weight="avg_time_to_destination"
        )
    else:
        avg = 0

    # rides with drones: calculate graph, diameter and average_shortest_path
    if only_main_routes:
        drone_spots = []
    else:
        drone_spots = [15011, 13001, 2002, 11007, 4016, 1009, 3020, 9019, 9005, 4025]

    drives_with_drones = add_drone_flights(
        df_edges, drives_without_drones, drone_spots=drone_spots, radius=drone_radius
    )
    # graph needs to be strongly connected to calcutate diameter
    graph_with_drones = calculate_graph(drives_with_drones)
    if nx.is_strongly_connected(graph_with_drones):
        diameter_with_drones = nx.diameter(
            graph_with_drones, weight="avg_time_to_destination"
        )
    else:
        diameter = 0

    if nx.is_weakly_connected(graph_without_drones):
        avg_with_drones = nx.average_shortest_path_length(
            graph_with_drones, weight="avg_time_to_destination"
        )

    return [
        len(rides_simulated) / month_diff,
        diameter,
        avg,
        diameter_with_drones,
        avg_with_drones,
    ]


def getRegressionMetrics(
    rides_simulated,
    df_edges,
    stepsize=15000,
    lower_boundary=10000,
    only_main_routes=False,
):
    """Takes an input dateframe of rides (simulated or not simulated), applies getDeliveryTimes() for increasing sample sizes of simulated rides to built one dataframe for regression
        The output is a dataframe of increasing sample sizes of simulated rides and used for regression of diameter and average shortest path
    Args:
        rides_simulated (DataFrame): DataFrame containing Rides in format provided by MoD
        df_edges (DataFrame): DataFrame containing stop combinations in format provided by MoD
        stepsize (int, optional): Stepsize determining in which increasing order samples of the original df will be created. Defaults to 15000.
    Returns:
        DataFrame: DataFrame containing the metrics ["#_simulated_rides", "diameter_w/o_drones", "avg_w/o_drones", "diameter_with_drones", "avg_with_drones"] for increasing number of simulated rides
    """
    # month_diff used to nomalize number of rides simulated to 1 month
    rides_simulated["scheduled_to"] = pd.to_datetime(rides_simulated["scheduled_to"])
    start_date = rides_simulated["scheduled_to"].min()
    end_date = rides_simulated["scheduled_to"].max()
    month_diff = (
        (end_date.year - start_date.year) * 12 + end_date.month - start_date.month
    )

    upper_boundary = len(rides_simulated)
    results_df = pd.DataFrame(
        columns=[
            "#_simulated_rides",
            "diameter_w/o_drones",
            "avg_w/o_drones",
            "diameter_with_drones",
            "avg_with_drones",
        ]
    )

    for n in list(range(lower_boundary, upper_boundary, stepsize)):
        current_sample_df = rides_simulated.sample(n=n)
        results_df.loc[len(results_df)] = getDeliveryTimes(
            current_sample_df, df_edges, month_diff, only_main_routes
        )

    return results_df


def regression_function(x, a, b, c):
    """Exponential Decay regression. Will be used for regression curve fitting and to calculate simulated rides number for optimzed parameters/ to calculate regression line

    Args:
        x (list/float): can be a list/list like of floats to calculate y values for x
        a (float): optimized parameter from function get_opt_parameter
        b (float): optimized parameter from function get_opt_parameter
        c (float): optimized parameter from function get_opt_parameter

    Returns:
        list/float: regressed y values for given x - equals the required number of rides for given maximum delivery days
    """
    return a * np.exp(-b / x) + c


def get_opt_parameter(graph_metrics_df, metric="avg_w/o_drones"):
    """Function takes upfront generated graph metrics dataframe as input to execute an exponential decay regression with the data.
       Output are the optimized parameters which can be used input for the function regression_function

    Args:
        graph_metrics_df (DataFrame): DataFrame containing graph metrics for increasing number of rides. Calculated upfront.
        metric (str, optional): graph metric for which regression will be done. ['diameter_w/o_drones',
       'avg_w/o_drones', 'diameter_with_drones', 'avg_with_drones'] are possible. Defaults to 'avg_w/o_drones'.

    Returns:
        list: list of parameters [a,b,c] which can be used as input for the regression_function
    """

    # gather data
    y = graph_metrics_df["#_simulated_rides"].to_numpy()
    x = graph_metrics_df[metric].to_numpy()

    # curve fitting
    popt, pcov = curve_fit(regression_function, x, y, maxfev=5000)

    return popt


def get_rides_num(max_days, graph_metrics_df, metric="avg_w/o_drones"):
    """Function takes upfront generated graph metrics dataframe as input to execute an exponential decay regression with the data.
       The regressed function is used to calculate the needed rides for a given max_days delivery threshold
    Args:
        max_days (int): Threshold of maximum delivery days
        graph_metrics_df (DataFrame): DataFrame containing graph metrics for increasing number of rides. Calculated upfront.
        metric (str, optional): graph metric for which regression will be done. ['diameter_w/o_drones',
       'avg_w/o_drones', 'diameter_with_drones', 'avg_with_drones'] are possible. Defaults to 'avg_w/o_drones'.
    Returns:
        float: minimum number of rides needed to deliver package within max_days threshold for given graph metric.
    """
    # gather data
    y = graph_metrics_df["#_simulated_rides"].to_numpy()
    x = graph_metrics_df[metric].to_numpy()

    # curve fitting
    popt, pcov = curve_fit(regression_function, x, y, maxfev=5000)

    return regression_function(max_days, *popt)


def calculate_number_drivers(df, hour, comined_rides_factor=0.3):
    """Calculates the number of drivers needed for the given rides in the dataframe in the given hour.

    Args:
        df (pandas DataFrame): Dataframe containing the rides for that we calculate how many drivers we need
        hour (int): The hour where we calculate for how many drivers we need
        comined_rides_factor (float, optional): Gives the percentage of how many drives can be combined with another drive. Defaults to 0.3.

    Returns:
        tuple(float, float): Tuple containing the number of drivers needed and how many parallel drives are there in the dataframe for the given hour.
    """
    df_drivers = df.copy()

    df_drivers["scheduled_to_same_month"] = dt.strptime(f"2022-01", "%Y-%m")
    df_drivers["scheduled_to_same_month"] += pd.to_timedelta(
        df_drivers["scheduled_to"].dt.hour, unit="h"
    )
    df_drivers["scheduled_to_same_month"] += pd.to_timedelta(
        df_drivers["scheduled_to"].dt.minute, unit="m"
    )
    df_drivers["scheduled_to_same_month"] += pd.to_timedelta(
        df_drivers["scheduled_to"].dt.day - 1, unit="d"
    )

    """
    # Calculate how many drives a driver can to be hour
    df_drivers["driver_time"] = df_drivers["dropoff_at"] - \
        df_drivers["dispatched_at"]
    df_drivers["driver_time"] = df_drivers["driver_time"].dt.seconds / 60
    avg_driver_time = df["driver_time"].mean()
    drives_per_hour = 60 / avg_driver_time
    """

    # Calculate how many parallel drives per hour exists
    parallel_drives = (
        df_drivers.resample("H", on="scheduled_to_same_month").id.count().reset_index()
    )
    parallel_drives.rename(columns={"id": "parallel_drives"}, inplace=True)
    parallel_drives = parallel_drives[
        parallel_drives["scheduled_to_same_month"].dt.hour == hour
    ]

    # Calculate average drives for given hour

    average_drives = parallel_drives["parallel_drives"].mean()

    average_drives = average_drives - average_drives * comined_rides_factor

    # Calculate max drives for given hour
    max_drives = parallel_drives["parallel_drives"].max()
    max_drives = max_drives - max_drives * comined_rides_factor

    return (average_drives / 3, average_drives)
