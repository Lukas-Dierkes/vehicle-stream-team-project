from datetime import datetime as dt

import networkx as nx
import pandas as pd

pd.set_option("display.max_columns", None)


def get_shortest_ride(df, startpoint, endpoint, start_date, end_date):
    start_date = dt.strptime(start_date, "%m/%d/%Y")
    end_date = dt.strptime(end_date, "%m/%d/%Y")
    days = (end_date - start_date).days + 1

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

    G = nx.from_pandas_edgelist(
        drives,
        source="pickup_address",
        target="dropoff_address",
        edge_attr="waiting_time",
    )

    path = nx.shortest_path(
        G, source=startpoint, target=endpoint, weight="avg_time_to_destination"
    )
    shortest_time = nx.shortest_path_length(
        G,
        source=startpoint,
        target=endpoint,
        weight="avg_time_to_destination",
        method="dijkstra",
    )

    return (path, shortest_time)
