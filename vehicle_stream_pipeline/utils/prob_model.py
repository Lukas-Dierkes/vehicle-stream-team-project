import collections
import json

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely.geometry
from scipy.optimize import curve_fit


def find_name_for_id(id, df_stops):
    return df_stops[df_stops["MoDStop Id"] == id]["MoDStop Name"].reset_index(
        drop=True
    )[0]


def find_id_for_name(name, df_stops):
    return df_stops[df_stops["MoDStop Name"] == name]["MoDStop Id"].reset_index(
        drop=True
    )[0]


def calculate_drives(df, start_date, end_date):
    """Calculates for each combination of spots the average ride time and and the average waiting time and stores it in a dataframe
    This dataframe is the bases for our graph.

    Args:
        df (pandas DataFrame): Dataframe containing all drives from MoD
        start_date (Timestamp): Used for filtering the dataframe so that only drives are stored that are later than the start date
        end_date (Timestamp): Used for filtering the dataframe so that only drives are stored that are earlier than the end date

    Returns:
        Pandas DataFrame: Containing the drives which are calculated.
    """
    # Calculate the number of days
    days = (end_date - start_date).days + 1

    # Get number of driver per pickup_address, droppoff_address combination
    drives = pd.DataFrame(
        df.groupby(["pickup_address", "dropoff_address"], group_keys=False)
        .size()
        .to_frame("number_of_drives")
    ).reset_index()

    # Calculate how many days you need to wait for one drive for each pickup dropoff combination
    drives["waiting_time"] = days / drives["number_of_drives"]

    # Calculate the average ride time for each combination
    drives["avg_ride_time"] = (
        df.groupby(
            ["pickup_address", "dropoff_address"], as_index=False, group_keys=False
        )["ride_time"].mean()["ride_time"]
        / 60
        / 60
        / 24
    )

    # Fill missing ride time values (measurement in days)
    drives["avg_ride_time"].fillna(0.001738, inplace=True)

    # Calculate the average time to destination by adding waiting time and ride time
    drives["avg_time_to_destination"] = drives["waiting_time"] + drives["avg_ride_time"]

    return drives


def calculate_graph(drives):
    """Creates the graph which is the basis for our model.

    Args:
        Drives (pandas DataFrame): Dataframe contains the avgeage time to destination from each spot to each other spot.

    Returns:
        Networkx graph: Return the weihted graph where each node is one spot and an edge shows whether there is an existing drive between these two routes.
        The edge weight is based on how often this route was driven before.
    """
    G = nx.from_pandas_edgelist(
        drives,
        source="pickup_address",
        target="dropoff_address",
        edge_attr="avg_time_to_destination",
        create_using=nx.DiGraph(),
    )
    return G


def get_shortest_ride(startpoint, endpoint, graph):
    """Calculates the shortest path from a given startpoint to a given endpoint
    and further calculates the shortest time based on the edge weight.

    Args:
        startpoint (str): Startpoint of the shortest path.
        endpoint (str): Endpoint of the shortest path.
        graph (networkx graph): Graph which is used to calculate the shortest path

    Returns:
        tuple(list[str], float): Returns a tuple which first item is a list conataining all intermediate spots to get to the endpoint from the startpoint.
        The second item it the shortest time which is needed for that path.
    """
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
    """Calculates the hotspots based on all combinations of shortest paths between two spots.
    Therefore we count how often a spot is an intermediate, starting or final spot in a all shortest paths.

    Args:
        df_edges (pandas DataFrame): Dataframe containing all combinations of spots.
        drives (pandas DataFrame): Dataframe containing all shortest rides between given spots.
        n (int, optional): Define how many spots will be returned. Defaults to 10.

    Returns:
        list(str): List of the calculated hotspts.
    """
    # Calculate graph
    graph = calculate_graph(drives)

    # Rename edges variables
    df_edges.rename(columns={"Start #": "start_id", "Ende #": "end_id"}, inplace=True)

    # Create a variable that contains information about whether the graph contains that edge or not
    df_edges["include"] = df_edges["Spots"] = df_edges.apply(
        lambda x: graph.has_edge(x.start_id, x.end_id), axis=1
    )

    # Filter edge combinations for only combination that has an edge in the graph
    df_edges_filtered = df_edges[df_edges["include"] == True]

    # Get shortest path for each combination
    df_edges_filtered["Spots"] = df_edges_filtered.apply(
        lambda x: get_shortest_ride(x.start_id, x.end_id, graph)[0], axis=1
    )

    # Create a list with all shortest paths
    df_edges_filtered = df_edges_filtered[df_edges_filtered.Spots != "Not in graph"]
    hotspots = list(df_edges_filtered["Spots"])

    # Flatten the list so we obtain not considering different paths anymore, only the stops
    hotspots = [x for xs in hotspots for x in xs]

    # Take the most n common spots from the list --> these are the hotspots so the spots that occur most often in a shortest path
    counter = collections.Counter(hotspots)
    hotspots = [i[0] for i in counter.most_common(n)]

    return hotspots


def add_drone_flights(df_edges, drives, drone_spots=[1008], radius=500):
    """Adding drone flights to the current drives. Therefore drones will be placed on given spots so that they can reach other spots in a given radius.
    We assume that drone flights have no waiting time.

    Args:
        df_edges (pandas DataFrame): Dataframe containing all combinations of spots.
        drives (pandas DataFrame): Dataframe containing all shortest rides between given spost.
        drone_spots (list, optional): List where drones should be placed. Defaults to [1008].
        radius (int, optional): Radius which drones are allowed to fly in. Defaults to 500.

    Returns:
        pandas DataFrame: returns a dataframe containing the normal drives plus the added drone flights
    """

    # Prepare edges dataframe and filter not needed columns
    drone_flights = df_edges.iloc[:, :6]
    drone_flights.drop(["Start Name", "Ende Name", "Route [m]"], axis=1, inplace=True)
    drone_flights.rename(columns={"Luftlinie [m]": "Luftlinie"}, inplace=True)

    # Filter every route which have a bigger radius than the given one because we only allow drone flight in a certrain radius
    drone_flights = drone_flights[drone_flights["Luftlinie"] <= radius]

    # We assume there is no waiting time for drones
    drone_flights["waiting_time"] = 0
    drone_flights["number_of_drives"] = 1

    # Calculate the ride time of drones
    drone_flights["avg_ride_time"] = (drone_flights.Luftlinie / 7) / 60 / 60 / 24
    drone_flights["avg_time_to_destination"] = (
        (drone_flights.Luftlinie / 7) / 60 / 60 / 24
    )

    # Filter for routes that either has the dropoff address or the pickup address at at drone spot
    # So we only allow drone flight from a drone spot to another spot in a radius or a drone flight from another spot in 500 m radius to the drone spot
    drone_flights = drone_flights[
        (drone_flights["start_id"].isin(drone_spots))
        | (drone_flights["end_id"].isin(drone_spots))
    ]

    drone_flights.rename(
        columns={"start_id": "pickup_address", "end_id": "dropoff_address"},
        inplace=True,
    )
    drone_flights.drop(["Luftlinie"], axis=1, inplace=True)

    # Combine drives and drone flights
    drives_w_flights = pd.merge(
        drives,
        drone_flights,
        left_on=["pickup_address", "dropoff_address"],
        right_on=["pickup_address", "dropoff_address"],
        how="left",
    )

    drives_w_flights["start_end"] = (
        drives_w_flights["pickup_address"].astype(str)
        + "_"
        + drives_w_flights["dropoff_address"].astype(str)
    )
    drone_flights["start_end"] = (
        drone_flights["pickup_address"].astype(str)
        + "_"
        + drone_flights["dropoff_address"].astype(str)
    )

    drone_flights_added = drone_flights[
        ~(drone_flights["start_end"].isin(drives_w_flights["start_end"]))
    ]

    # Change of exising times (with cars) by the faster times that we have now because of the drones.
    # Only for drives to a drone spot or from a drone spot (both in 500m radius)
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


# copied and from https://stackoverflow.com/questions/68946831/draw-a-polygon-around-point-in-scattermapbox-using-python
def create_circle(
    radius=500,
    position={"Longitude": 8.13664, "Latitude": 49.3517},
):
    """Creates a circle around a spot. So we can visualize the radius of a dronespot.

    Args:
        radius (int, optional): Length of the radius. Defaults to 500.
        position (dict, optional): position of the dronepot define in latitude and longitude. Defaults to {"Longitude": 8.13664, "Latitude": 49.3517}.

    Returns:
        pandas GeoDataFrame: returns a geo dataframe containing information on how to calculate the circles for the given spot
    """

    # generate a geopandas data frame of the POI
    gdfpoi = gpd.GeoDataFrame(
        geometry=[shapely.geometry.Point(position["Longitude"], position["Latitude"])],
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
            gpd.GeoDataFrame(geometry=gdfpoi if False else None),
        ]
    )


def create_circles_around_drone_spots(df, radius=500):
    """Creates a list containing information for the radius which is drawn around the dronespot for each dronepot.

    Args:
        df (pandas DataFrame): dataframe containing all spots where drones are placed.
        radius (int, optional): Radius which drones are allowed to fly in.. Defaults to 500.

    Returns:
        list[dicts]: Containing meta information for the circles to be created in the visualization
    """
    layers = []
    for index, row in df.iterrows():
        current_spot = {
            "Longitude": row["MoDStop Long"],
            "Latitude": row["MoDStop Lat"],
        }

        layers.append(
            {
                "source": json.loads(
                    create_circle(position=current_spot, radius=radius).to_json()
                ),
                "below": "traces",
                "type": "line",
                "color": "purple",
                "line": {"width": 1.5},
            },
        )

    return layers


def get_route_information(drives, path, df_stops):
    """Creates a text for the dashbaord containing the path including all intermediate spots and for each edge on the path the shortest time.

    Args:
        drives (_type_): Dataframe containing all the aggregated drives.
        path (list): List containing all stops for the given route.
        df_stops (pandas DataFrame): Dataframe with each stop.

    Returns:
        str: Text containing information about the path taken and how long to get to each of the intermediate stops
    """
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


def get_geo_cordinates_for_path(df_stops, path):
    """Returns the geo coordinates (long, lat) for each spot in the given dataframe.

    Args:
        df_stops (pandas DataFrame): Dataframe with each stop
        path (str): Path where to find the excel sheet which stores each stop.

    Returns:
        tuple(list, list): Returns two lists, one for all the latitudes of the spots and one for the longitutes of the spots.
    """
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
