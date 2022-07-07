import collections
import json

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import shapely.geometry

pd.set_option("display.max_columns", None)


def get_geo_cordinates_for_path(df_stops, path):
    stop_latitudes = []
    stop_longitudes = []
    for stop in path:
        stop_latitudes.append(
            float(df_stops[df_stops["MoDStop Id"] == stop]["MoDStop Lat"])
        )
        stop_longitudes.append(
            float(df_stops[df_stops["MoDStop Id"] == stop]["MoDStop Long"])
        )

    return (stop_latitudes, stop_longitudes)


def calculate_graph(drives):
    G = nx.from_pandas_edgelist(
        drives,
        source="pickup_address",
        target="dropoff_address",
        edge_attr="avg_time_to_destination",
        create_using=nx.DiGraph(),
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
        return ("Not in graph", -1)
    elif endpoint not in graph:
        return ("Not in graph", -1)
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

    df_edges["include"] = df_edges["Spots"] = df_edges.apply(
        lambda x: graph.has_edge(x.start_id, x.end_id), axis=1
    )

    df_edges_filtered = df_edges[df_edges["include"] == True]

    df_edges_filtered["Spots"] = df_edges_filtered.apply(
        lambda x: get_shortest_ride(x.start_id, x.end_id, graph)[0], axis=1
    )

    df_edges_filtered = df_edges_filtered[df_edges_filtered.Spots != "Not in graph"]
    hotspots = list(df_edges_filtered["Spots"])
    hotspots = [x for xs in hotspots for x in xs]
    counter = collections.Counter(hotspots)
    return counter.most_common(n)


def add_drone_flights(df_edges, drives, drone_spots=[1008], radius=500):
    drone_flights = df_edges.iloc[:, :6]
    drone_flights.drop(["Start Name", "Ende Name", "Route [m]"], axis=1, inplace=True)
    drone_flights.rename(columns={"Luftlinie [m]": "Luftlinie"}, inplace=True)

    drone_flights = drone_flights[drone_flights["Luftlinie"] <= radius]

    drone_flights["number_of_drives"] = 1
    drone_flights["waiting_time"] = 0
    drone_flights["avg_ride_time"] = (drone_flights.Luftlinie * 0.12) / 60 / 60 / 24
    drone_flights["avg_time_to_destination"] = (
        (drone_flights.Luftlinie * 0.12) / 60 / 60 / 24
    )

    drone_flights = drone_flights[
        (drone_flights["start_id"].isin(drone_spots))
        | (drone_flights["end_id"].isin(drone_spots))
    ]

    drone_flights.rename(
        columns={"start_id": "pickup_address", "end_id": "dropoff_address"},
        inplace=True,
    )
    drone_flights.drop(["Luftlinie"], axis=1, inplace=True)

    drives_w_flights = pd.merge(
        drives,
        drone_flights,
        left_on=["pickup_address", "dropoff_address"],
        right_on=["pickup_address", "dropoff_address"],
        how="left",
    )

    drives_w_flights["start_end"] = (
        drives_w_flights["pickup_address"] + drives_w_flights["dropoff_address"]
    )
    drone_flights["start_end"] = (
        drone_flights["pickup_address"] + drone_flights["dropoff_address"]
    )

    drone_flights_added = drone_flights[
        ~(drone_flights["start_end"].isin(drives_w_flights["start_end"]))
    ]

    drives_w_flights["avg_time_to_destination"] = np.where(
        (
            drives_w_flights["avg_time_to_destination_x"]
            > drives_w_flights["avg_time_to_destination_y"]
        ),
        drives_w_flights.avg_time_to_destination_y,
        drives_w_flights.avg_time_to_destination_x,
    )

    drives_w_flights = drives_w_flights[
        [
            "pickup_address",
            "dropoff_address",
            "avg_time_to_destination",
        ]
    ]

    drone_flights_added = drone_flights_added[
        [
            "pickup_address",
            "dropoff_address",
            "avg_time_to_destination",
        ]
    ]

    drives_w_flights = pd.concat([drives_w_flights, drone_flights_added])

    return drives_w_flights


# copied from https://stackoverflow.com/questions/68946831/draw-a-polygon-around-point-in-scattermapbox-using-python
def poi_poly(
    df,
    radius=500,
    # ,{"lat": 49.3517, "lon": 8.13664}
    poi={"Longitude": 8.13664, "Latitude": 49.3517},
    lon_col="MoDStop Long",
    lat_col="MoDStop Lat",
    include_radius_poly=False,
):

    # generate a geopandas data frame of the POI
    gdfpoi = gpd.GeoDataFrame(
        geometry=[shapely.geometry.Point(poi["Longitude"], poi["Latitude"])],
        crs="EPSG:4326",
    )
    # extend point to radius defined (a polygon).  Use UTM so that distances work, then back to WSG84
    gdfpoi = (
        gdfpoi.to_crs(gdfpoi.estimate_utm_crs())
        .geometry.buffer(radius)
        .to_crs("EPSG:4326")
    )

    gdf = gpd.GeoDataFrame(geometry=gdfpoi)

    # create a polygon around the edges of the markers that are within POI polygon
    return pd.concat(
        [
            gpd.GeoDataFrame(
                geometry=[
                    gpd.sjoin(
                        gdf, gpd.GeoDataFrame(geometry=gdfpoi), how="inner"
                    ).unary_union.convex_hull
                ]
            ),
            gpd.GeoDataFrame(geometry=gdfpoi if include_radius_poly else None),
        ]
    )


def create_circles_around_drone_spots(df, radius=500):
    layers = []
    for index, row in df.iterrows():
        current_spot = {
            "Longitude": row["MoDStop Long"],
            "Latitude": row["MoDStop Lat"],
        }

        layers.append(
            {
                "source": json.loads(
                    poi_poly(None, poi=current_spot, radius=radius).to_json()
                ),
                "below": "traces",
                "type": "line",
                "color": "purple",
                "line": {"width": 1.5},
            },
        )

    return layers


def get_route_information(drives, path, df_stops):
    times_and_path = []
    text = ""
    for i in range(len(path) - 1):
        current_stop = path[i]
        next_stop = path[i + 1]

        current_time = round(
            float(
                drives[
                    (drives["pickup_address"] == current_stop)
                    & (drives["dropoff_address"] == next_stop)
                ]["avg_time_to_destination"]
            ),
            5,
        )
        current_name = find_name_for_id(current_stop, df_stops)
        next_name = find_name_for_id(next_stop, df_stops)

        times_and_path.append(f"{current_name} - {next_name}: {current_time} days")

    nl = "\n \n"
    text = f"The shortest path is :{nl}{nl.join(times_and_path)}"

    return text


def find_name_for_id(id, df_stops):
    return df_stops[df_stops["MoDStop Id"] == id]["MoDStop Name"].reset_index(
        drop=True
    )[0]


def find_id_for_name(name, df_stops):
    return df_stops[df_stops["MoDStop Name"] == name]["MoDStop Id"].reset_index(
        drop=True
    )[0]
