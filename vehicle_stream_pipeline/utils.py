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
from scipy.stats import truncnorm


def create_overall_dataframes(path):
    """This function creates three big dataframes out of the given excel files from Mod so we combine the data from all months.
    1: kpi_combined.csv: That is the monthly kpi-stats combined. We rarely use this data
    2. mtd_combined.csv: That (should) contain all the rides combined for each day of the month according to excel sheet.
    3. rides_combined: Here we iterated over each day (excel sheet) and collected the data for each day on our own. Suprisingly this is different to the mtd_combined.csv and seems like that this data is more accurate. So we will use this dataframe for further analysis.

    Args:
        path (str): Path to the folder which stores the excel sheets.

    Returns:
        dict: The combined three dataframes.
    """
    directory = os.fsencode(path)
    df_rides = pd.DataFrame()
    df_kpi = pd.DataFrame()
    df_mtd = pd.DataFrame()
    for filename in os.listdir(directory):
        if "Rides" not in str(filename):
            continue

        # Get correct filename
        current_file = os.path.join(directory, filename).decode("utf-8")
        current_file = current_file.replace("$", "")
        current_file = current_file.replace("~", "")

        # Read current excel file
        df_dict = pd.read_excel(current_file, sheet_name=None)

        # Extract dataframes from current excel file
        df_kpi_temp = df_dict["KPI"]
        df_dict.pop("KPI")
        df_mtd_temp = df_dict["MTD"]
        df_dict.pop("MTD")
        # union all rides from all days in current excel file
        df_rides_temp = pd.concat(df_dict, ignore_index=True)

        # Create big dataframes over all excel files (all months combined)
        df_kpi = pd.concat([df_kpi, df_kpi_temp], axis=0, ignore_index=True)
        df_mtd = pd.concat([df_mtd, df_mtd_temp], axis=0, ignore_index=True)
        df_rides = pd.concat([df_rides, df_rides_temp], axis=0, ignore_index=True)

    return {"df_kpi": df_kpi, "df_mtd": df_mtd, "df_rides": df_rides}


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

    days = (end_date - start_date).days + 1
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

    drives["avg_ride_time"].fillna(0.001738, inplace=True)

    drives["avg_time_to_destination"] = drives["waiting_time"] + drives["avg_ride_time"]

    return drives


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
    drone_flights = df_edges.iloc[:, :6]
    drone_flights.drop(["Start Name", "Ende Name", "Route [m]"], axis=1, inplace=True)
    drone_flights.rename(columns={"Luftlinie [m]": "Luftlinie"}, inplace=True)

    drone_flights = drone_flights[drone_flights["Luftlinie"] <= radius]

    drone_flights["number_of_drives"] = 1
    drone_flights["waiting_time"] = 0
    drone_flights["avg_ride_time"] = (drone_flights.Luftlinie / 7) / 60 / 60 / 24
    drone_flights["avg_time_to_destination"] = (
        (drone_flights.Luftlinie / 7) / 60 / 60 / 24
    )

    # TODO: find all ids that are in the radius of the drone spot and allow rides for each combination of rides within this points
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


def find_name_for_id(id, df_stops):
    return df_stops[df_stops["MoDStop Id"] == id]["MoDStop Name"].reset_index(
        drop=True
    )[0]


def find_id_for_name(name, df_stops):
    return df_stops[df_stops["MoDStop Name"] == name]["MoDStop Id"].reset_index(
        drop=True
    )[0]


def get_date_range(start_date, end_date):
    start_month = start_date.month
    start_year = start_date.year
    end_month = end_date.month
    end_year = end_date.year

    years = list(range(start_year, end_year + 1))
    i = len(years)
    j = 0
    months = []
    years_all = []
    if i == 1:
        months.extend(range(start_month, end_month + 1))
        years_all.extend([start_year] * (end_month - start_month + 1))
    else:
        while i > 1:
            if j == 0:
                months.extend(range(start_month, 13))
                years_all.extend([start_year] * (12 - start_month + 1))
                j = j + 1
            else:
                months.extend(range(1, 13))
                years_all.extend([start_year + j] * 12)
            i = i - 1
        months.extend(range(1, end_month + 1))
        years_all.extend([end_year] * (end_month))
        i = i - 1
    date_range = list(zip(years_all, months))
    return date_range


# help function that returns a probability distribution for continous variables based on mean & standard deviation
def getdistribution(data, column, min=None, max=None):
    """This function generates a normal distribution object for the contious values in a specified column in a DataFrame. If needed the distribution can be truncated between a min & max value.

    Args:
        data (DataFrame): Pandas DataFrame that contains the column for which a distribution is to be created
        column (String): Name of the column for which a distribution is to be created
        min (Float, optional): Minimum value of the distribution / left border. Defaults to None.
        max (Float, optional): Maximum value of the distribution / right border. Defaults to None.

    Returns:
        truncnorm: The normal distribution of the specified column - can be truncated to the range [min, max].
    """

    # distribution over scheduled rides
    mean = data[column].median()  # use median to better deal with outlier
    std = data[column].std()
    if min == None:
        a = data[column].min()  # min value
    else:
        a = min
    if max == None:
        b = data[column].max()  # max value
    else:
        b = max
    return stats.truncnorm((a - mean) / std, (b - mean) / std, loc=mean, scale=std)


def generateRoute(oldRides, newRides, ridestops, routes):
    """This function selects a route (pickup_addres + dropoff_address) for every new ride based on the route distribution depending on the scheduled_to timestamp of the new ride.

    Args:
        oldRides (DataFrame): DataFrame containing all origianl past rides - basis for building probability distributions.
        newRides (DataFrame): DataFrame containing an intermediate result within the process of ride simulation
        ridestops (DataFrame): DataFrame containing all MoDStops
        routes (DataFrame): DataFrame containing all routes between all MoDStops

    Returns:
        DataFrame: Returns the following DataFrame columns ["pickup_address", "dropoff_address", "distance", "shortest_ridetime"] for the new rides.
    """

    # add route identifier to routes dataframe
    allRoutes = routes[
        routes["Route [m]"] > 500
    ]  # Assumption: real rides are at least 500 m long
    allRoutes["route"] = (
        allRoutes["start_id"].astype(str) + "-" + allRoutes["end_id"].astype(str)
    )

    # based on analysis of rides we distinguish between workdays (Monday till Friday noon) and weekend (Friday noon till Sunday)
    newRideStops = pd.DataFrame(
        newRides[["created_at", "scheduled_to", "pickup_address", "dropoff_address"]],
        columns=["created_at", "scheduled_to", "pickup_address", "dropoff_address"],
    )
    newRideStops["route"] = ""
    newRideStops["day"] = newRideStops["scheduled_to"].apply(lambda x: dt.weekday(x))
    newRideStops["hour"] = newRideStops["scheduled_to"].apply(lambda x: x.hour)
    newRideStops["workday"] = np.where(
        (
            newRideStops["day"].isin([0, 1, 2, 3, 4])  # 0 = Monday, 6 = Sunday
            & ~((newRideStops["day"] == 4) & (newRideStops["hour"] > 13))
        ),
        True,
        False,
    )

    oldRidestops = pd.DataFrame(
        oldRides[["state", "scheduled_to", "pickup_address", "dropoff_address"]],
        columns=["state", "scheduled_to", "pickup_address", "dropoff_address"],
    )
    oldRidestops["route"] = (
        oldRidestops["pickup_address"].astype(str)
        + "-"
        + oldRidestops["dropoff_address"].astype(str)
    )
    oldRidestops["scheduled_to"] = pd.to_datetime(oldRidestops["scheduled_to"])
    oldRidestops["day"] = oldRidestops["scheduled_to"].apply(lambda x: dt.weekday(x))
    oldRidestops["hour"] = oldRidestops["scheduled_to"].apply(lambda x: x.hour)
    oldRidestops["workday"] = np.where(
        (
            oldRidestops["day"].isin([0, 1, 2, 3, 4])  # 0 = Monday, 6 = Sunday
            & ~((oldRidestops["day"] == 4) & (oldRidestops["hour"] > 13))
        ),
        True,
        False,
    )
    workdayOldRides = oldRidestops[(oldRidestops["workday"] == True)]
    weekendOldRides = oldRidestops[(oldRidestops["workday"] == False)]

    # generate ridestops
    for h in [0] + list(range(7, 24)):  # rides start between 7:00 and 0:59
        # timeframe used to get ridestop distribution
        if h in [23, 0]:
            timeframe = [22, 23, 0]
        elif h == 7:
            timeframe = [7, 8, 9]
        else:
            timeframe = list(range(h - 1, h + 2))

        ##### workday ridestop distribution #####
        distWorkday = (
            workdayOldRides[(workdayOldRides["hour"].isin(timeframe))]["route"]
            .value_counts()
            .rename_axis("route")
            .reset_index(name="counts")
        )
        numberOfNoise = distWorkday["counts"].sum() / 80 * 20  # 20% noise
        allRoutes["counts"] = distWorkday[
            "counts"
        ].min()  # noise is weighted similar to least frequent real driven route
        distWorkday = pd.concat(
            [
                distWorkday,
                allRoutes[~allRoutes["route"].isin(distWorkday["route"])].sample(
                    frac=1
                )[: int(numberOfNoise)][["route", "counts"]],
            ]
        )
        distWorkday["probabilities"] = distWorkday.counts / distWorkday.counts.sum()

        ##### weekend ridestop distribution #####
        distWeekend = (
            weekendOldRides[(weekendOldRides["hour"].isin(timeframe))]["route"]
            .value_counts()
            .rename_axis("route")
            .reset_index(name="counts")
        )
        numberOfNoise = distWeekend["counts"].sum() / 80 * 20  # 20% noise
        allRoutes["counts"] = distWeekend[
            "counts"
        ].min()  # noise is weighted similar to least frequent real driven route
        distWeekend = pd.concat(
            [
                distWeekend,
                allRoutes[~allRoutes["route"].isin(distWeekend["route"])].sample(
                    frac=1
                )[: int(numberOfNoise)][["route", "counts"]],
            ]
        )
        distWeekend["probabilities"] = distWeekend.counts / distWeekend.counts.sum()

        # split newRideStops dataframe in 1. ride-hour=h & weekend, 2. ride-hour=h & workday, 3. rest
        newRideStops_h_wend = newRideStops[
            (newRideStops["hour"] == h) & (newRideStops["workday"] == False)
        ]
        newRideStops_h_work = newRideStops[
            (newRideStops["hour"] == h) & (newRideStops["workday"] == True)
        ]
        newRideStops_not_h = newRideStops[
            ~((newRideStops["hour"] == h) & (newRideStops["workday"] == False))
            & ~((newRideStops["hour"] == h) & (newRideStops["workday"] == True))
        ]

        # generate routes based on distributions
        newRideStops_h_wend["route"] = np.random.choice(
            distWeekend["route"],
            p=distWeekend["probabilities"],
            size=newRideStops_h_wend.shape[0],
        )
        newRideStops_h_work["route"] = np.random.choice(
            distWorkday["route"],
            p=distWorkday["probabilities"],
            size=newRideStops_h_work.shape[0],
        )

        # concat 3 pieces back together
        newRideStops = pd.concat(
            [newRideStops_not_h, newRideStops_h_wend, newRideStops_h_work]
        )

    # Extract pickup & dropoff address from route column
    newRideStops[["pickup_address", "dropoff_address"]] = newRideStops[
        "route"
    ].str.split("-", expand=True)
    newRideStops["pickup_address"] = pd.to_numeric(newRideStops["pickup_address"])
    newRideStops["dropoff_address"] = pd.to_numeric(newRideStops["dropoff_address"])

    # Extract 'distance' and 'shortest_ridetime' based on generated routes
    newRideStops["distance"] = newRideStops.merge(
        routes,
        left_on=["pickup_address", "dropoff_address"],
        right_on=["start_id", "end_id"],
        how="left",
    )["Route [m]"]
    newRideStops["shortest_ridetime"] = (
        1 / (30 / (newRideStops["distance"] / 1000)) * 60 * 60
    )  # calculate shortest_ridetime in seconds with average speed of 30 km/h
    newRideStops.sort_values(by=["created_at"])
    return newRideStops[
        ["pickup_address", "dropoff_address", "distance", "shortest_ridetime"]
    ]


# function that returns n random 'created_at' timestamps over a period of one specified month based on the probability distribution in original data
# first step: choose a date from the month based on the probability distribution of rides over the weekdays (Monday-Sunday)
# second step: choose a timestamp based on the probability distribution of rides that are on the same weekday
def generateCreatedAt(oldRides, newRides, m, y):
    """This function returns n random 'created_at' timestamps over a period of one specified month based on the probability distribution in original data with respect to the weekdays (Monday-Sunday) as well as the hour

    Args:
        oldRides (DataFrame): DataFrame containing all origianl past rides - basis for building probability
        newRides (DataFrame): DataFrame containing an intermediate result within the process of ride simulation
        m (Integer): Defines the month for which rides should be simulated. January=1 .. December=12
        y (Integer): Defines the year for which rides should be simulated

    Returns:
        Series: Returns a Series object with simulated timestamps for the attribut created_at for all new rides
    """

    # creat list with all days of the month to build up the probability distribution
    if m == 12:
        m1 = 1
        y1 = y + 1
    else:
        m1 = m + 1
        y1 = y
    daydist = pd.DataFrame(
        pd.date_range(
            start=str(m) + "/01/" + str(y),
            end=str(m1) + "/01/" + str(y1),
        )
        .to_pydatetime()
        .tolist()[:-1],
        columns=["date"],
    )
    daydist["weekday"] = daydist["date"].apply(
        lambda x: dt.weekday(x)
    )  # use the weekday distribution to represent real occurrences of rides

    # extract all dates and their weekday, hour and minute
    created = pd.DataFrame(
        pd.to_datetime(oldRides["created_at"]), columns=["created_at"]
    )
    created["day"] = created["created_at"].apply(lambda x: dt.weekday(x))
    created["hour"] = created["created_at"].apply(lambda x: x.hour)
    created["minute"] = created["created_at"].apply(lambda x: x.minute)

    # get the weekday distribution of old rides
    dist_day = (
        created["day"].value_counts().rename_axis("day").reset_index(name="counts")
    )
    dist_day["probabilities"] = dist_day.counts / dist_day.counts.sum()
    dist_day = dist_day.sort_values("day")

    # get the hour distribution of old rides per weekday
    dist_hour = []
    for i in range(0, 7):
        dist_hour.append(
            created[created["day"] == i]["hour"]
            .value_counts()
            .rename_axis("hour")
            .reset_index(name="counts")
        )
        dist_hour[i]["probabilities"] = dist_hour[i].counts / dist_hour[i].counts.sum()
        dist_hour[i] = dist_hour[i].sort_values("hour")

    # get the minute distribution of old rides
    dist_minute = (
        created["minute"]
        .value_counts()
        .rename_axis("minute")
        .reset_index(name="counts")
    )
    dist_minute["probabilities"] = dist_minute.counts / dist_minute.counts.sum()
    dist_minute = dist_minute.sort_values("minute")

    # match probability that a ride is on that weekday to all dates in the simulated month
    daydist["probabilities"] = daydist["weekday"].apply(
        lambda x: dist_day[dist_day["day"] == x]["probabilities"].values[0]
    )
    daydist["probabilities"] = daydist["probabilities"] / (
        daydist["probabilities"].sum()
    )  # normalization neccessary to get probability distribution (sum of odds is 1)

    # generate list of values
    values = pd.DataFrame(
        np.random.choice(
            daydist["date"], p=daydist["probabilities"], size=newRides.shape[0]
        ),
        columns=["created_at"],
    )
    values = values.sort_values("created_at")
    values = values.reset_index()
    values["day"] = values["created_at"].apply(lambda x: dt.weekday(x))
    values["created_at"] = values["created_at"] + values["day"].apply(
        lambda x: pd.Timedelta(
            hours=np.random.choice(
                dist_hour[x]["hour"], p=dist_hour[x]["probabilities"]
            ),  # choose hour based on distribution of that weekday
            minutes=np.random.choice(
                dist_minute["minute"], p=dist_minute["probabilities"]
            ),  # choose minute based on distribution of that hour
            # random choice of seconds
            seconds=np.random.choice(list(range(0, 60))),
        )
    )
    values.sort_values(by=["created_at"])
    return values["created_at"]


# function that returns n random 'scheduled_to' timestamps based on the probability distribution in original data
# first, consider distribution of scheduled & immediate
# second, for a scheduled ride add a random prebooking time (based on probability distribution of the prebooking time in original data) to created_at
def generateScheduledTo(oldRides, newRides):
    """This function returns n random 'scheduled_to' timestamps based on the probability distribution in original data. Distribution of the amount of prebooked & immediate rides as well as of the prebooking time is considered

    Args:
        oldRides (DataFrame): DataFrame containing all origianl past rides - basis for building probability
        newRides (DataFrame): DataFrame containing an intermediate result within the process of ride simulation

    Returns:
        Series: Returns a Series object with simulated timestamps for the attribut created_at for all new rides
    """

    scheduledNew = pd.DataFrame(columns=["hour"])
    scheduledNew["created_at"] = newRides["created_at"]
    scheduledNew["hour"] = scheduledNew["created_at"].apply(lambda x: x.hour)

    # get prebooking time
    scheduled = pd.DataFrame(
        oldRides[["created_at", "scheduled_to"]], columns=["created_at", "scheduled_to"]
    )
    scheduled["isScheduled"] = scheduled.created_at != scheduled.scheduled_to
    scheduled["created_at"] = pd.to_datetime(scheduled["created_at"])
    scheduled["scheduled_to"] = pd.to_datetime(scheduled["scheduled_to"])
    scheduled["prebook_time"] = scheduled.scheduled_to - scheduled.created_at
    scheduled["prebook_time"] = scheduled["prebook_time"].apply(
        lambda x: x.total_seconds()
    )

    # distribution of prebooked and non-prebooked rides
    dist = (
        scheduled["isScheduled"]
        .value_counts()
        .rename_axis("isScheduled")
        .reset_index(name="counts")
    )
    dist["probabilities"] = dist.counts / dist.counts.sum()

    # distribution of average prebook time
    left_border = (
        8 * 60
    )  # min value of 8 min -> assumption: scheduled ride must be at least 8 min in the future
    dist_avg_prebook_time = getdistribution(
        scheduled[scheduled["isScheduled"] == True], "prebook_time", min=left_border
    )

    scheduledNew["scheduled_to"] = [
        (i + pd.Timedelta(dist_avg_prebook_time.rvs(1)[0], unit="seconds")).round(
            freq="10min"
        )
        if np.random.choice(dist["isScheduled"], p=dist["probabilities"])
        else i
        for i, j in zip(scheduledNew.created_at, scheduledNew.hour)
    ]
    # we have no rides before 7 but don't want to many rides starting at 7:00am
    # get target share of rides starting at hour=7
    oldRides["scheduled_to"] = pd.to_datetime(oldRides["scheduled_to"])
    oldRides["hour"] = oldRides["scheduled_to"].apply(lambda x: x.hour)
    dist_hours_oldRides = (
        oldRides["hour"].value_counts().rename_axis("hour").reset_index(name="counts")
    )
    dist_hours_oldRides["probabilities"] = (
        dist_hours_oldRides.counts / dist_hours_oldRides.counts.sum()
    )
    targetRidesAtSeven = dist_hours_oldRides[dist_hours_oldRides["hour"] == 7][
        "probabilities"
    ].values[0]
    # get missing amount of rides starting at hour = 7
    scheduledNew["hour"] = scheduledNew["scheduled_to"].apply(lambda x: x.hour)
    invalidRides = (
        scheduledNew[(scheduledNew["hour"].isin([1, 2, 3, 4, 5, 6]))].count().values[0]
    )
    ridesAtSeven = scheduledNew[(scheduledNew["hour"] == 7)].count().values[0] / len(
        newRides
    )
    missingRidesAtSeven = (
        (targetRidesAtSeven - ridesAtSeven) * len(newRides) / invalidRides
    )  # share of invalid rides that should start at 7 o'clock
    # if not enough rides starting at 7, then add missing rides starting at 7
    if missingRidesAtSeven > 0:
        scheduledNew["scheduled_to"] = [
            dt(scheduled.year, scheduled.month, scheduled.day, 7, 0)
            if (h in [2, 3, 4, 5, 6])
            & (np.random.uniform(0.0, 1.0) <= missingRidesAtSeven)
            else scheduled
            for scheduled, h in zip(scheduledNew.scheduled_to, scheduledNew.hour)
        ]
    # distribute rest over all hours of the day on the next day after created_at & don't consider 7am rides anymore
    dist_hours_oldRides = dist_hours_oldRides[
        (dist_hours_oldRides.hour != 7) & (dist_hours_oldRides.hour != 6)
    ]
    dist_hours_oldRides["probabilities"] = (
        dist_hours_oldRides.counts / dist_hours_oldRides.counts.sum()
    )
    scheduledNew["hour"] = scheduledNew["scheduled_to"].apply(lambda x: x.hour)
    scheduledNew["scheduled_to"] = [
        dt(
            (created + pd.Timedelta(1, unit="days")).year,
            (created + pd.Timedelta(1, unit="days")).month,
            (created + pd.Timedelta(1, unit="days")).day,
            np.random.choice(
                dist_hours_oldRides["hour"], p=dist_hours_oldRides["probabilities"]
            ),
            np.random.choice(list(range(0, 60, 10))),
        )
        if h in [1, 2, 3, 4, 5, 6]
        else scheduled
        for created, scheduled, h in zip(
            scheduledNew.created_at, scheduledNew.scheduled_to, scheduledNew.hour
        )
    ]
    return scheduledNew["scheduled_to"]


# function that returns n random 'dispatched_at' timestamps
# case 1: scheduled ride -> dispatched_at = scheduled_at - 8 min
# case 2: immediate ride -> dispatched_at = scheduled_at
def generateDispatchedAt(oldRides, newRides):
    """This function returns n 'dispatched_at' timestamps based on the logic: scheduled ride => scheduled_to - 8 min.; immediate ride => scheduled_to

    Args:
        oldRides (DataFrame): DataFrame containing all origianl past rides - basis for building probability
        newRides (DataFrame): DataFrame containing an intermediate result within the process of ride simulation

    Returns:
        Series: Returns a Series object with simulated timestamps for the attribut dispatched_at for all new rides
    """

    scheduled = pd.DataFrame(
        newRides[["created_at", "scheduled_to"]], columns=["created_at", "scheduled_to"]
    )
    scheduled["isScheduled"] = scheduled.created_at != scheduled.scheduled_to
    scheduled["created_at"] = pd.to_datetime(scheduled["created_at"])
    scheduled["scheduled_to"] = pd.to_datetime(scheduled["scheduled_to"])
    scheduled["dispatched_at"] = np.where(
        (scheduled["isScheduled"] == True),
        np.where(
            (
                scheduled["scheduled_to"] - pd.Timedelta(minutes=8)
                > scheduled["created_at"]
            ),
            scheduled["scheduled_to"] - pd.Timedelta(minutes=8),
            scheduled["created_at"],
        ),
        scheduled["scheduled_to"],
    )
    return scheduled["dispatched_at"]


# function that returns n random timestamps for 'arriving_push', vehicle_arrived_at', 'pickup_arrival_time'
# first, generate 'vehicle_arrived_at'
# for scheduled rides: scheduled_to + random scheduling deviation (based on probability distribution of scheduling_deviation=vehicle_arrived_at-scheduled_to in original data)
# for immediatie rides: scheduled_to + random pickup_arrival_time (based on probability distribution of pickup_arrival_time in original data)
# second, calculate pickup_arrival_time=vehicle_arrived_at-dispatched_at etc.
def generateArrival(oldRides, newRides):
    """This function returns n random 'vehicle_arrived_at' timestamps & 'pickup_arrival_time' time periods based on probability distributions in original data. Distributions of time needed to arrive at the pickup_address for both cases prebooked & immediate rides are considered

    Args:
        oldRides (DataFrame): DataFrame containing all origianl past rides - basis for building probability
        newRides (DataFrame): DataFrame containing an intermediate result within the process of ride simulation

    Returns:
        DataFrame: Returns the following DataFrame columns ["vehicle_arrived_at", "pickup_arrival_time"] for the new rides.
    """

    # get needed information regarding the vehicle arrival in old data
    arrivalOld = pd.DataFrame(
        oldRides[
            [
                "created_at",
                "scheduled_to",
                "dispatched_at",
                "vehicle_arrived_at",
                "pickup_arrival_time",
            ]
        ],
        columns=[
            "created_at",
            "scheduled_to",
            "dispatched_at",
            "vehicle_arrived_at",
            "pickup_arrival_time",
        ],
    )
    arrivalOld["isScheduled"] = arrivalOld.created_at != arrivalOld.scheduled_to
    arrivalOld["created_at"] = pd.to_datetime(arrivalOld["created_at"])
    arrivalOld["scheduled_to"] = pd.to_datetime(arrivalOld["scheduled_to"])
    arrivalOld["vehicle_arrived_at"] = pd.to_datetime(arrivalOld["vehicle_arrived_at"])
    # arrivalOld['arriving_push'] = pd.to_datetime(arrivalOld['arriving_push'])

    # create dataframe with needed attributes to determine 'vehicle_arrived_at'
    arrivalNew = pd.DataFrame(
        newRides[["created_at", "scheduled_to", "dispatched_at"]],
        columns=["created_at", "scheduled_to", "dispatched_at"],
    )
    arrivalNew["isScheduled"] = arrivalNew.created_at != arrivalNew.scheduled_to
    arrivalNew["created_at"] = pd.to_datetime(arrivalNew["created_at"])
    arrivalNew["scheduled_to"] = pd.to_datetime(arrivalNew["scheduled_to"])
    arrivalNew["dispatched_at"] = pd.to_datetime(arrivalNew["dispatched_at"])

    # generate timestamp 'vehicle_arrived_at'
    arrivalOld["schedule_deviation"] = arrivalOld.apply(
        lambda row: (
            (row["vehicle_arrived_at"] - row["scheduled_to"]).round(freq="s")
        ).total_seconds(),
        axis=1,
    )

    # distribution over scheduled rides
    left_border = -8 * 60  # min value of -8 minutes -> earliest arrived_at
    dist_scheduledRides = getdistribution(
        arrivalOld[arrivalOld["isScheduled"] == True],
        "schedule_deviation",
        min=left_border,
    )

    # distribution over instant rides - based on pickup_arrival_times distribution
    left_border = 1  # min value of 1 second -> earliest arrived_at
    dist_instantRides = getdistribution(
        arrivalOld[arrivalOld["isScheduled"] == False],
        "pickup_arrival_time",
        min=left_border,
    )

    # determine timestamp 'vehicle_arrived_at'
    arrivalNew["vehicle_arrived_at"] = arrivalNew.apply(
        lambda row: (
            row["scheduled_to"]
            + pd.Timedelta(dist_scheduledRides.rvs(1)[0], unit="seconds").round(
                freq="s"
            )
        )
        if (row["isScheduled"] == True)
        else (
            row["scheduled_to"]
            + pd.Timedelta(dist_instantRides.rvs(1)[0], unit="seconds").round(freq="s")
        ),
        axis=1,
    )
    # check that vehicle_arrived_at is after dispatched_at
    arrivalNew["vehicle_arrived_at"] = np.where(
        arrivalNew["dispatched_at"] > arrivalNew["vehicle_arrived_at"],
        arrivalNew["dispatched_at"]
        + (arrivalNew["scheduled_to"] - arrivalNew["dispatched_at"])
        * np.random.uniform(0.1, 0.9),
        arrivalNew["vehicle_arrived_at"],
    )

    # calculate 'pickup_arrival_time'
    arrivalNew["pickup_arrival_time"] = (
        arrivalNew["vehicle_arrived_at"] - arrivalNew["dispatched_at"]
    ).dt.seconds

    return arrivalNew[["vehicle_arrived_at", "pickup_arrival_time"]]


def generatePickup(oldRides, newRides):
    """This function returns for all new rides: 'earliest_pickup_expectation'(= dispatched_at + 3 min), 'pickup_at' (= arrived_at + random boarding time based on probability distribution in origional data) and the following 3 timestamps 'arriving_push', 'pickup_eta' and 'pickup_first_eta' based on probability distributions of their deviation around pickup_at in original data.

    Args:
        oldRides (DataFrame): DataFrame containing all origianl past rides - basis for building probability
        newRides (DataFrame): DataFrame containing an intermediate result within the process of ride simulation

    Returns:
        DataFrame: Returns the following DataFrame columns ["arriving_push", "earliest_pickup_expectation",  "pickup_at", "pickup_eta", "pickup_first_eta"] for the new rides.
    """

    # get needed information regarding the pickup in old data
    pickupOld = pd.DataFrame(
        oldRides[
            [
                "arriving_push",
                "created_at",
                "scheduled_to",
                "dispatched_at",
                "vehicle_arrived_at",
                "pickup_at",
                "pickup_first_eta",
                "pickup_eta",
            ]
        ],
        columns=[
            "arriving_push",
            "created_at",
            "scheduled_to",
            "dispatched_at",
            "vehicle_arrived_at",
            "pickup_at",
            "pickup_first_eta",
            "pickup_eta",
        ],
    )
    pickupOld[
        [
            "arriving_push",
            "created_at",
            "scheduled_to",
            "dispatched_at",
            "vehicle_arrived_at",
            "pickup_at",
            "pickup_first_eta",
            "pickup_eta",
        ]
    ] = pickupOld[
        [
            "arriving_push",
            "created_at",
            "scheduled_to",
            "dispatched_at",
            "vehicle_arrived_at",
            "pickup_at",
            "pickup_first_eta",
            "pickup_eta",
        ]
    ].apply(
        pd.to_datetime
    )
    pickupOld["isScheduled"] = pickupOld.created_at != pickupOld.scheduled_to
    pickupOld["arriving_push"] = pd.to_datetime(pickupOld["arriving_push"])

    # create dataframe with needed attributes to determine pickup attributes
    pickupNew = pd.DataFrame(
        newRides[["created_at", "scheduled_to", "dispatched_at", "vehicle_arrived_at"]],
        columns=["created_at", "scheduled_to", "dispatched_at", "vehicle_arrived_at"],
    )
    pickupNew["isScheduled"] = pickupNew.created_at != pickupNew.scheduled_to
    pickupNew[
        ["created_at", "scheduled_to", "dispatched_at", "vehicle_arrived_at"]
    ] = pickupNew[
        ["created_at", "scheduled_to", "dispatched_at", "vehicle_arrived_at"]
    ].apply(
        pd.to_datetime
    )

    # generate earliest_pickup_expectation
    pickupNew["earliest_pickup_expectation"] = pickupNew[
        "dispatched_at"
    ] + pd.Timedelta(minutes=3)

    # genrate pickup_at
    pickupOld["time_until_pickup"] = pickupOld.apply(
        lambda row: (
            (row["pickup_at"] - row["vehicle_arrived_at"]).round(freq="s")
        ).total_seconds(),
        axis=1,
    )
    # distribution of the time a driver waits until pickup over scheduled rides
    left_border = 1  # min value of 1 second -> earliest arrived_at
    dist_scheduledRides = getdistribution(
        pickupOld[pickupOld["isScheduled"] == True],
        "time_until_pickup",
        min=left_border,
    )

    # distribution of the time a driver waits until pickup over instant rides
    left_border = 1  # min value of 1 second -> earliest arrived_at
    dist_instantRides = getdistribution(
        pickupOld[pickupOld["isScheduled"] == False],
        "time_until_pickup",
        min=left_border,
    )

    # determine timestamp 'pickup_at'
    pickupNew["pickup_at"] = pickupNew.apply(
        lambda row: (
            row["vehicle_arrived_at"]
            + pd.Timedelta(dist_scheduledRides.rvs(1)[0], unit="seconds").round(
                freq="s"
            )
        )
        if (row["isScheduled"] == True)
        else (
            row["vehicle_arrived_at"]
            + pd.Timedelta(dist_instantRides.rvs(1)[0], unit="seconds").round(freq="s")
        ),
        axis=1,
    )

    # generate pickup_eta
    # distribution of the time between pickup_at and pickup_eta
    pickupOld["deviation_of_pickup_eta"] = pickupOld.apply(
        lambda row: (
            (row["pickup_eta"] - row["pickup_at"]).round(freq="s")
        ).total_seconds(),
        axis=1,
    )
    dist = getdistribution(pickupOld, "deviation_of_pickup_eta")

    # determine timestamp 'pickup_eta'
    pickupNew["pickup_eta"] = pickupNew.apply(
        lambda row: (
            row["pickup_at"]
            + pd.Timedelta(dist.rvs(1)[0], unit="seconds").round(freq="s")
        ),
        axis=1,
    )

    # check that pickup_eta is after dispatched_at
    pickupNew["pickup_eta"] = np.where(
        pickupNew["dispatched_at"] > pickupNew["pickup_eta"],
        pickupNew["dispatched_at"] + pd.Timedelta(minutes=3),  # TODO: mehr randomness
        pickupNew["pickup_eta"],
    )

    # generate pickup_first_eta
    # distribution of the time between pickup_at and pickup_first_eta
    pickupOld["deviation_of_pickup_first_eta"] = pickupOld.apply(
        lambda row: (
            (row["pickup_first_eta"] - row["pickup_at"]).round(freq="s")
        ).total_seconds(),
        axis=1,
    )
    dist = getdistribution(pickupOld, "deviation_of_pickup_first_eta")

    # determine timestamp 'pickup_first_eta'
    pickupNew["pickup_first_eta"] = pickupNew.apply(
        lambda row: (
            row["pickup_at"]
            + pd.Timedelta(dist.rvs(1)[0], unit="seconds").round(freq="s")
        ),
        axis=1,
    )

    # check that pickup_first_eta is at least 3 min. after created_at
    pickupNew["pickup_first_eta"] = np.where(
        (pickupNew["created_at"] + pd.Timedelta(minutes=3))
        > pickupNew["pickup_first_eta"],
        pickupNew["created_at"]
        + pd.Timedelta(minutes=3),  # created_at + 3 min. is minimum
        pickupNew["pickup_first_eta"],
    )
    # check that pickup_first_eta is after dispatched_at
    pickupNew["pickup_first_eta"] = np.where(
        pickupNew["dispatched_at"] > pickupNew["pickup_first_eta"],
        pickupNew["dispatched_at"] + pd.Timedelta(minutes=3),  # TODO: mehr randomness?
        pickupNew["pickup_first_eta"],
    )

    # check that pickup_first_eta before pickup_eta
    pickupNew["pickup_first_eta"] = np.where(
        pickupNew["pickup_first_eta"] > pickupNew["pickup_eta"],
        pickupNew["pickup_eta"],
        pickupNew["pickup_first_eta"],
    )

    # generate arriving_push
    # distribution of the time between arriving_push and vehicle_arrived_at
    pickupOld["deviation_of_arriving_push"] = pickupOld.apply(
        lambda row: (
            (row["arriving_push"] - row["pickup_eta"]).round(freq="s")
        ).total_seconds(),
        axis=1,
    )
    dist = getdistribution(pickupOld, "deviation_of_arriving_push")

    # determine timestamp 'arriving_push'
    pickupNew["arriving_push"] = pickupNew.apply(
        lambda row: (
            row["pickup_eta"]
            + pd.Timedelta(dist.rvs(1)[0], unit="seconds").round(freq="s")
        ),
        axis=1,
    )

    # check that arriving_push is after dispatched_at
    pickupNew["arriving_push"] = np.where(
        pickupNew["dispatched_at"] > pickupNew["arriving_push"],
        pickupNew["dispatched_at"]
        + (pickupNew["pickup_eta"] - pickupNew["dispatched_at"])
        * np.random.uniform(0.1, 0.9),
        pickupNew["arriving_push"],
    )

    return pickupNew[
        [
            "arriving_push",
            "earliest_pickup_expectation",
            "pickup_at",
            "pickup_eta",
            "pickup_first_eta",
        ]
    ]


def generateDropoff(oldRides, newRides, routes):
    """This function returns for all new rides: 'dropoff_at'(= pickup_at + average ridetime (+/- random deviation up to 20% possible) of the most similar rides, otherwise shortest_ridetime) and the following 2 timestamps "dropoff_eta" and "dropoff_first_eta" based on probability distributions of their deviation around dropoff_at in original data.

    Args:
        oldRides (DataFrame): DataFrame containing all origianl past rides - basis for building probability
        newRides (DataFrame): DataFrame containing an intermediate result within the process of ride simulation
        routes (DataFrame): DataFrame containing all routes between all MoDStops

    Returns:
        DataFrame: Returns the following DataFrame columns ["dropoff_at", "dropoff_eta", "dropoff_first_eta"] for the new rides.
    """

    # get needed information regarding the dropoff in old data
    dropoffOld = pd.DataFrame(
        oldRides[
            [
                "pickup_address",
                "dropoff_address",
                "scheduled_to",
                "dropoff_at",
                "dropoff_first_eta",
                "dropoff_eta",
                "ride_time",
            ]
        ],
        columns=[
            "pickup_address",
            "dropoff_address",
            "scheduled_to",
            "dropoff_at",
            "dropoff_first_eta",
            "dropoff_eta",
            "ride_time",
        ],
    )
    dropoffOld[
        ["scheduled_to", "dropoff_at", "dropoff_first_eta", "dropoff_eta"]
    ] = dropoffOld[
        ["scheduled_to", "dropoff_at", "dropoff_first_eta", "dropoff_eta"]
    ].apply(
        pd.to_datetime
    )
    dropoffOld["day"] = dropoffOld["scheduled_to"].apply(lambda x: dt.weekday(x))
    dropoffOld["hour"] = dropoffOld["scheduled_to"].apply(lambda x: x.hour)
    dropoffOld["workday"] = np.where(
        (
            dropoffOld["day"].isin([0, 1, 2, 3, 4])  # 0 = Monday, 6 = Sunday
            & ~((dropoffOld["day"] == 4) & (dropoffOld["hour"] > 13))
        ),
        True,
        False,
    )

    # create dataframe with needed attributes to determine dropoff attributes
    dropoffNew = pd.DataFrame(
        newRides[
            [
                "pickup_address",
                "dropoff_address",
                "scheduled_to",
                "pickup_at",
                "pickup_first_eta",
                "pickup_eta",
                "shortest_ridetime",
            ]
        ],
        columns=[
            "pickup_address",
            "dropoff_address",
            "scheduled_to",
            "pickup_at",
            "pickup_first_eta",
            "pickup_eta",
            "shortest_ridetime",
        ],
    )
    dropoffNew[
        ["scheduled_to", "pickup_at", "pickup_first_eta", "pickup_eta"]
    ] = dropoffNew[
        ["scheduled_to", "pickup_at", "pickup_first_eta", "pickup_eta"]
    ].apply(
        pd.to_datetime
    )
    dropoffNew["day"] = dropoffNew["scheduled_to"].apply(lambda x: dt.weekday(x))
    dropoffNew["hour"] = dropoffNew["scheduled_to"].apply(lambda x: x.hour)
    dropoffNew["timeframe"] = dropoffNew["hour"].apply(
        lambda h: (
            [22, 23, 0]
            if h in [23, 0]
            else ([7, 8, 9] if h == 7 else list(range(h - 1, h + 2)))
        )
    )
    dropoffNew["workday"] = np.where(
        (
            dropoffNew["day"].isin([0, 1, 2, 3, 4])  # 0 = Monday, 6 = Sunday
            & ~((dropoffNew["day"] == 4) & (dropoffNew["hour"] > 13))
        ),
        True,
        False,
    )

    # generate ride_time based on ride_time of most similar rides
    dropoffNew["ride_time"] = dropoffNew.apply(
        lambda row:
        # if rides exist with same route & workday/weekend flag & in a timeframe of +/-1 hour
        round(
            dropoffOld[
                (dropoffOld["pickup_address"] == row["pickup_address"])
                & (dropoffOld["dropoff_address"] == row["dropoff_address"])
                & (dropoffOld["workday"] == row["workday"])
                & (dropoffOld["hour"].isin(row["timeframe"]))
            ]["ride_time"].mean()
            * np.random.uniform(0.9, 1.1)  # mean +/- up to 10% randomness
        )
        if len(
            dropoffOld[
                (dropoffOld["pickup_address"] == row["pickup_address"])
                & (dropoffOld["dropoff_address"] == row["dropoff_address"])
                & (dropoffOld["workday"] == row["workday"])
                & (dropoffOld["hour"].isin(row["timeframe"]))
            ]["ride_time"]
        )
        > 0
        else
        # if rides exist with same route & in a timeframe of +/-1 hour - workday/weekend does not matter
        round(
            dropoffOld[
                (dropoffOld["pickup_address"] == row["pickup_address"])
                & (dropoffOld["dropoff_address"] == row["dropoff_address"])
                & (dropoffOld["hour"].isin(row["timeframe"]))
            ]["ride_time"].mean()
            * np.random.uniform(0.9, 1.1)  # mean +/- up to 10% randomness
        )
        if len(
            dropoffOld[
                (dropoffOld["pickup_address"] == row["pickup_address"])
                & (dropoffOld["dropoff_address"] == row["dropoff_address"])
                & (dropoffOld["hour"].isin(row["timeframe"]))
            ]["ride_time"]
        )
        > 0
        else
        # if rides exist with same route - day & hour does not matter
        round(
            dropoffOld[
                (dropoffOld["pickup_address"] == row["pickup_address"])
                & (dropoffOld["dropoff_address"] == row["dropoff_address"])
            ][
                "ride_time"
            ].mean()  # mean +/- up to 10% randomness
        )
        if len(
            dropoffOld[
                (dropoffOld["pickup_address"] == row["pickup_address"])
                & (dropoffOld["dropoff_address"] == row["dropoff_address"])
            ]["ride_time"]
        )
        > 0
        else
        # else, use shortest ridetime: 30km/h over distance of the route
        row["shortest_ridetime"]
        * np.random.uniform(1.0, 1.1),  # mean up to +10% randomness,
        axis=1,
    )

    # genereate dropoff_at
    dropoffNew["dropoff_at"] = dropoffNew["pickup_at"] + pd.to_timedelta(
        dropoffNew["ride_time"], unit="seconds"
    )

    # generate dropoff_first_eta
    dropoffNew["dropoff_first_eta"] = dropoffNew["pickup_first_eta"] + pd.to_timedelta(
        dropoffNew["shortest_ridetime"], unit="seconds"
    )

    # generate dropoff_eta
    # distribution of the time between dropoff_at and dropoff_eta
    dropoffOld["deviation_of_dropoff_eta"] = dropoffOld.apply(
        lambda row: (
            (row["dropoff_eta"] - row["dropoff_at"]).round(freq="s")
        ).total_seconds(),
        axis=1,
    )
    dist = getdistribution(dropoffOld, "deviation_of_dropoff_eta")

    # determine timestamp 'dropoff_eta'
    dropoffNew["dropoff_eta"] = dropoffNew.apply(
        lambda row: (
            row["dropoff_at"]
            + pd.Timedelta(dist.rvs(1)[0], unit="seconds").round(freq="s")
        ),
        axis=1,
    )

    # check that dropoff_eta is after pickup_eta & pickup_at
    dropoffNew["dropoff_eta"] = np.where(
        (dropoffNew["pickup_eta"] > dropoffNew["dropoff_eta"])
        | (dropoffNew["pickup_at"] > dropoffNew["dropoff_eta"]),
        dropoffNew["dropoff_at"] + pd.Timedelta(minutes=3),  # TODO: mehr randomness
        dropoffNew["dropoff_eta"],
    )

    return dropoffNew[["dropoff_at", "dropoff_eta", "dropoff_first_eta"]]


# general function that returns n random values based on the probability distribution of a certain column
# used for the following ride attributes: number_of_passenger, free_ride, payment_type, arrival_indicator, rating
def generateValues(column_name, df, newRides):
    """This function generates a probability distribution for discrete values in a specified column in a DataFrame. Then, n random choices are made and returned in a list.

    Args:
        column_name (String): Name of the column for which a distribution is to be created and values are to be generated
        df (DataFrame): Pandas DataFrame that contains the column for which a distribution is to be created
        newRides (DataFrame): DataFrame containing an intermediate result within the process of ride simulation; the df length represents the amount of values to be generated.

    Returns:
        List: n random choices from a generated probability distribution of discrete values.
    """
    dist = (
        df[column_name]
        .value_counts()
        .rename_axis(column_name)
        .reset_index(name="counts")
    )
    dist["probabilities"] = dist.counts / dist.counts.sum()
    return np.random.choice(
        dist[column_name], p=dist["probabilities"], size=newRides.shape[0]
    )


# Attributes: ['arrival_deviation', 'waiting_time', 'boarding_time', 'ride_time', 'trip_time', 'delay', 'longer_route_factor']
def generateTimeperiods(newRides):
    """This function calculates the following time periods / KPIs for all new rides: 'arrival_deviation', 'waiting_time', 'boarding_time', 'ride_time', 'trip_time', 'delay', 'longer_route_factor'

    Args:
        newRides (DataFrame): DataFrame containing an intermediate result within the process of ride simulation; the df length represents the amount of values to be generated.

    Returns:
        DataFrame: Returns the following DataFrame columns ['arrival_deviation', 'waiting_time', 'boarding_time', 'ride_time', 'trip_time', 'delay', 'longer_route_factor'] for the new rides.
    """

    # Attribute: 'arrival_deviation'
    newRides["arrival_deviation"] = newRides.apply(
        lambda row: (
            (row["vehicle_arrived_at"] - row["arriving_push"]).round(freq="s")
        ).total_seconds()
        - 180
        if (row["vehicle_arrived_at"] == row["vehicle_arrived_at"])
        and (row["arriving_push"] == row["arriving_push"])
        else np.NaN,
        axis=1,
    )

    # Attribute: 'waiting_time'
    newRides["waiting_time"] = newRides.apply(
        lambda row: (
            (row["vehicle_arrived_at"] - row["earliest_pickup_expectation"]).round(
                freq="s"
            )
        ).total_seconds()
        if (row["vehicle_arrived_at"] == row["vehicle_arrived_at"])
        and (row["earliest_pickup_expectation"] == row["earliest_pickup_expectation"])
        else np.NaN,
        axis=1,
    )

    # Attribute: 'boarding_time'
    newRides["boarding_time"] = newRides.apply(
        lambda row: (
            (row["pickup_at"] - row["vehicle_arrived_at"]).round(freq="s")
        ).total_seconds()
        if (row["vehicle_arrived_at"] == row["vehicle_arrived_at"])
        and (row["pickup_at"] == row["pickup_at"])
        else np.NaN,
        axis=1,
    )

    # Attribute: 'ride_time'
    newRides["ride_time"] = newRides.apply(
        lambda row: (
            (row["dropoff_at"] - row["pickup_at"]).round(freq="s")
        ).total_seconds()
        if (row["dropoff_at"] == row["dropoff_at"])
        and (row["pickup_at"] == row["pickup_at"])
        else np.NaN,
        axis=1,
    )

    # Attribute: 'trip_time'
    newRides["trip_time"] = newRides.apply(
        lambda row: (row["ride_time"] + row["waiting_time"]),
        axis=1,
    )

    # Attribute: 'delay'
    newRides["delay"] = newRides.apply(
        lambda row: (row["trip_time"] - row["shortest_ridetime"]),
        axis=1,
    )

    # Attribute: 'longer_route_factor'
    newRides["longer_route_factor"] = newRides.apply(
        lambda row: round(row["ride_time"] / row["shortest_ridetime"], 2)
        if (row["shortest_ridetime"] != 0)
        else np.NaN,
        axis=1,
    )

    return newRides[
        [
            "arrival_deviation",
            "waiting_time",
            "boarding_time",
            "ride_time",
            "trip_time",
            "delay",
            "longer_route_factor",
        ]
    ]


def generateRideSpecs(oldRides, ridestops, routes, n, month, year):
    """This function creates an empty MoD ride data DataFrame and incrementally fills the attributes for n simulated rides based on probability distributions in the original data.

    Args:
        oldRides (DataFrame): DataFrame containing all origianl past rides - basis for building probability
        ridestops (DataFrame): DataFrame containing all MoDStops
        routes (DataFrame): DataFrame containing all routes between all MoDStops
        n (Integer): Defines the amount of rides that are to simulated
        month (Integer): Defines the month for which rides should be simulated. January=1 .. December=12
        year (Integer): Defines the year for which rides should be simulated

    Returns:
        DataFrame: DataFrame consisting of details of n simulated rides.
    """

    timestamp = str(round(time.time()))
    warnings.filterwarnings("ignore")
    newRides = pd.DataFrame(columns=oldRides.columns)
    oldRides = oldRides[oldRides["state"] == "completed"]
    newRides["id"] = [timestamp + "-" + str(x) for x in list(range(0, n))]
    newRides["user_id"] = [
        str(x) + "-" + timestamp for x in list(range(0, n))
    ]  # Ein Kunde mehrere Rides
    newRides["number_of_passenger"] = generateValues(
        "number_of_passenger", oldRides, newRides
    )
    newRides["free_ride"] = generateValues("free_ride", oldRides, newRides)
    newRides["payment_type"] = generateValues("payment_type", oldRides, newRides)
    newRides["state"] = "completed"
    newRides["arrival_indicator"] = generateValues(
        "arrival_indicator", oldRides, newRides
    )
    newRides["rating"] = generateValues(
        "rating", oldRides, newRides
    )  # zufällig ratings rein, die nicht bisher gerated wurden? Oder Rating ganz raus?
    newRides["created_at"] = generateCreatedAt(oldRides, newRides, month, year)
    newRides["scheduled_to"] = generateScheduledTo(oldRides, newRides)
    newRides[
        ["pickup_address", "dropoff_address", "distance", "shortest_ridetime"]
    ] = generateRoute(
        oldRides, newRides, ridestops, routes
    )  # prices are not considered
    # newRides[['pickup_address', 'dropoff_address','distance', 'shortest_ridetime']] = generateRoute_simple(oldRides, newRides, ridestops, routes) # prices are not considered
    # newRides[['pickup_address', 'dropoff_address','distance', 'shortest_ridetime']] = generateRoute_simple2(oldRides, newRides, ridestops, routes) # prices are not considered
    newRides["dispatched_at"] = generateDispatchedAt(oldRides, newRides)
    newRides[["vehicle_arrived_at", "pickup_arrival_time"]] = generateArrival(
        oldRides, newRides
    )
    newRides[
        [
            "arriving_push",
            "earliest_pickup_expectation",
            "pickup_at",
            "pickup_eta",
            "pickup_first_eta",
        ]
    ] = generatePickup(oldRides, newRides)
    newRides[["dropoff_at", "dropoff_eta", "dropoff_first_eta"]] = generateDropoff(
        oldRides, newRides, routes
    )
    newRides[
        [
            "arrival_deviation",
            "waiting_time",
            "boarding_time",
            "ride_time",
            "trip_time",
            "delay",
            "longer_route_factor",
        ]
    ] = generateTimeperiods(newRides)
    warnings.filterwarnings("default")

    return newRides


# creates new Time Attributes out of Timestamp for Distplots
def transformForDist(input_df, dataset_name):
    """Divides pickup_at Timestamp to hour, month, day_of_month, year, week, day of week and adds it to input_df to create distribution plot.
        Add attributes dataset name and route (pickup_adress+dropoff_adress)

    Args:
        input_df (DataFrame): Rides Data in format of MoD
        dataset_name (str): Either Orignal or Simulated - later used for Grouping and Plot label

    Returns:
        DataFrame: DataFrame which can be used for DistPlots
    """
    dist_df = input_df.copy()
    dist_df["dataset"] = dataset_name
    dist_df["route"] = dist_df["pickup_address"].astype(str) + dist_df[
        "dropoff_address"
    ].astype(str)
    dist_df["pickup_at"] = pd.to_datetime(dist_df["pickup_at"], errors="coerce")
    hour = dist_df["pickup_at"].dt.hour
    month = dist_df["pickup_at"].dt.month
    day_of_month = dist_df["pickup_at"].dt.day
    year = dist_df["pickup_at"].dt.year
    week = dist_df["pickup_at"].dt.isocalendar().week
    day_of_week = dist_df["pickup_at"].dt.dayofweek
    dist_df.insert(loc=1, column="week", value=week)
    dist_df.insert(loc=2, column="month", value=month)
    dist_df.insert(loc=3, column="day_of_month", value=day_of_month)
    dist_df.insert(loc=4, column="hour", value=hour)
    dist_df.insert(loc=5, column="year", value=year)
    dist_df.insert(loc=6, column="day_of_week", value=day_of_week)
    dist_df["month_year"] = dist_df["pickup_at"].dt.strftime("%Y%m")
    return dist_df


# transform Df for Routes visualisations
def transformForRoute(dist_df_input, dataset_name):
    """Performs absolute and relative value counts for routes to create Dataframe which will be used for Route plot

    Args:
        dist_df_input (DataFrame): Dataframe Output of the Function TransformForDist - needs to have attribute "route" (pickup_adress+dropoff_adress)
        dataset_name (_type_): Either Orignal or Simulated - later used for Grouping and Plot label

    Returns:
        DataFrame: DataFrame which will be used for Route plot with attributes ["route", "rel_counts", "abs_counts","dataset"]
    """
    input_df = dist_df_input.copy()
    df_value_counts_rel = pd.DataFrame(input_df["route"].value_counts(normalize=True))
    df_value_counts_abs = pd.DataFrame(input_df["route"].value_counts())
    df_value_counts = pd.concat(objs=[df_value_counts_rel, df_value_counts_abs], axis=1)
    df_value_counts = df_value_counts.reset_index()
    df_value_counts.columns = ["route", "rel_counts", "abs_counts"]
    df_value_counts["dataset"] = dataset_name

    return df_value_counts


def transformForBar(n, df_value_counts_rides, df_value_counts_sim_l):
    """Adds information of top n abs counts for routes of original rides. Then compares if the top n routes of simulated rides are in the top n of the original rides. Output is used for Bar Chart.

    Args:
        n (int): top n most frequent routes
        df_value_counts_rides (DataFrame): Dataframe Output of the Function TransformForDist - Original Rides
        df_value_counts_sim_l (DataFrame): Dataframe Output of the Function TransformForDist - Simulated Rides

    Returns:
        DataFrame: Filtered DataFrame which will be used for Route plot bar chart with attributes ["route", "rel_counts", "abs_counts","dataset","own_top_n"]
    """

    # df with n_largest abs_rides - foundation for top_df, used for simulated rides to find matches
    top_df_value_counts_rides = df_value_counts_rides.nlargest(
        n=n, columns="abs_counts"
    )
    top_df_value_counts_rides["own_top_n"] = True

    # df with n_largest of simulated rides - used to see if n_largest of sim match n_largest of orig rides
    nlargest_sim_l = df_value_counts_sim_l.nlargest(n=n, columns="abs_counts")

    # top_df for sim rides - contain attribute "own top 10" which shows if route is in the own 10 of the respective sim rides
    top_df_sim_l = df_value_counts_sim_l.loc[
        df_value_counts_sim_l["route"].isin(top_df_value_counts_rides["route"])
    ]
    top_df_sim_l["own_top_n"] = top_df_sim_l["route"].apply(
        lambda x: nlargest_sim_l["route"].eq(x).any()
    )

    top_df = pd.concat([top_df_value_counts_rides, top_df_sim_l])
    return top_df


def calculate_number_drivers(df, hour, comined_rides_factor=0.3):
    """Calculates the number of drivers needed for the given rides in the dataframe in the given hour.

    Args:
        df (pandas DataFrame): Dataframe containing the rides for that we calculate how many drivers we need
        hour (int): The hour where we calculate for how many drivers we need
        comined_rides_factor (float, optional): Gives the percentage of how many drives can be combined with another drive. Defaults to 0.3.

    Returns:
        tuple(float, float): Tuple containing the number of drivers needed and how many parallel drives are there in the dataframe for the given hour.
    """
    # Calculate how many drives a driver can to be hour
    df["driver_time"] = df["dropoff_at"] - df["dispatched_at"]
    df["driver_time"] = df["driver_time"].dt.seconds / 60
    avg_driver_time = df["driver_time"].mean()
    drives_per_hour = 60 / avg_driver_time

    # Calculate how many parallel drives per hour exists
    df["driver_time"] = df["dropoff_at"] - df["dispatched_at"]
    parallel_drives = df.resample("H", on="scheduled_to").id.count().reset_index()
    parallel_drives.rename(columns={"id": "parallel_drives"}, inplace=True)
    parallel_drives = parallel_drives[parallel_drives["scheduled_to"].dt.hour == hour]

    # Calculate average drives for given hour

    average_drives = parallel_drives["parallel_drives"].mean()

    average_drives = average_drives - average_drives * comined_rides_factor

    # Calculate max drives for given hour
    max_drives = parallel_drives["parallel_drives"].max()
    max_drives = max_drives - max_drives * comined_rides_factor

    return (average_drives / 3, average_drives)


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
    print("testtest")
    # gather data
    y = graph_metrics_df["#_simulated_rides"].to_numpy()
    x = graph_metrics_df[metric].to_numpy()

    # curve fitting
    popt, pcov = curve_fit(regression_function, x, y, maxfev=5000)

    return regression_function(max_days, *popt)


def test():
    return 2
