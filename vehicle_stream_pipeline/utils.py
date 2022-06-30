import collections

import networkx as nx
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)


def get_geo_cordinates_for_path(df_stops, path):
    stop_latitudes = []
    stop_longitudes = []
    for stop in path:
        stop_latitudes.append(float(df_stops[df_stops["id"] == stop]["latitude"]))
        stop_longitudes.append(float(df_stops[df_stops["id"] == stop]["longitude"]))

    return (stop_latitudes, stop_longitudes)


def calculate_graph(drives):
    G = nx.from_pandas_edgelist(
        drives,
        source="pickup_address",
        target="dropoff_address",
        edge_attr="avg_time_to_destination",
    )
    return G


def calculate_drives(df, start_date, end_date):
    days = (end_date - start_date).days + 1
    df["scheduled_to"] = pd.to_datetime(df["scheduled_to"])
    df = df[(df["scheduled_to"] > start_date) & (df["scheduled_to"] < end_date)]

    drives = pd.DataFrame(
        df.groupby(["pickup_address", "dropoff_address"], group_keys=False)
        .size()
        .to_frame("number_of_drives")
    ).reset_index()

    drives["waiting_time"] = days / drives["number_of_drives"]

    drives["avg_ride_time"] = (
        df.groupby(
            ["pickup_address", "dropoff_address"], as_index=False, group_keys=False
        )["ride_time"].mean()["ride_time"]
        / 60
        / 60
        / 24
    )

    drives["avg_time_to_destination"] = drives["waiting_time"] + drives["avg_ride_time"]

    return drives


def get_shortest_ride(startpoint, endpoint, graph):
    if startpoint not in graph:
        return "Not in graph"
    elif endpoint not in graph:
        return "Not in graph"
    else:
        path = nx.shortest_path(
            graph, source=startpoint, target=endpoint, weight="avg_time_to_destination"
        )
        shortest_time = nx.shortest_path_length(
            graph,
            source=startpoint,
            target=endpoint,
            weight="avg_time_to_destination",
            method="dijkstra",
        )

        return (path, shortest_time)


def get_hotspots(df_edges, drives, n=10):
    graph = calculate_graph(drives)
    df_edges.rename(columns={"Start #": "start_id", "Ende #": "end_id"}, inplace=True)
    df_edges["Spots"] = df_edges.apply(
        lambda x: get_shortest_ride(x.start_id, x.end_id, graph)[0], axis=1
    )

    df_edges = df_edges[df_edges.Spots != "Not in graph"]
    hotspots = list(df_edges["Spots"])
    hotspots = [x for xs in hotspots for x in xs]
    counter = collections.Counter(hotspots)
    return counter.most_common(n)


def add_drone_flights(df_edges, drives, drone_spots=[1008], radius=500):
    drone_flights = df_edges.iloc[:, :6]
    drone_flights.drop(["Start Name", "Ende Name", "Route [m]"], axis=1, inplace=True)
    drone_flights.rename(columns={"Luftlinie [m]": "Luftlinie"}, inplace=True)
    drone_flights = drone_flights[drone_flights.Luftlinie <= radius]

    drone_flights["avg_time_to_destination"] = (
        (drone_flights.Luftlinie * 0.12) / 60 / 60 / 24
    )

    drone_flights = drone_flights[
        (drone_flights["start_id"].isin(drone_spots))
        | (drone_flights["end_id"].isin(drone_spots))
    ]

    drives_w_flights = pd.merge(
        drives,
        drone_flights,
        left_on=["pickup_address", "dropoff_address"],
        right_on=["start_id", "end_id"],
        how="left",
    )

    drives_w_flights["avg_time_to_destination"] = np.where(
        (
            drives_w_flights["avg_time_to_destination_x"]
            > drives_w_flights["avg_time_to_destination_y"]
        ),
        drives_w_flights.avg_time_to_destination_y,
        drives_w_flights.avg_time_to_destination_x,
    )

    return drives_w_flights[
        [
            "pickup_address",
            "dropoff_address",
            "number_of_drives",
            "waiting_time",
            "avg_ride_time",
            "avg_time_to_destination",
        ]
    ]
