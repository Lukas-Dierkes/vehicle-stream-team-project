import collections
import json
import time
from datetime import datetime as dt

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats as stats
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

    drives["avg_ride_time"].fillna(0.001738, inplace=True)

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


# TODO: Check in the end if too many rides, which are too short or are not likely enough


def generateRoute(oldRides, newRides, ridestops, routes):
    # based on analysis of rides we distinguish between workdays (Monday till Friday noon) and weekend (Friday noon till Sunday)
    newRideStops = pd.DataFrame(
        newRides[["scheduled_to", "pickup_address", "dropoff_address"]],
        columns=["scheduled_to", "pickup_address", "dropoff_address"],
    )
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
    workdayOldRides = oldRidestops[
        (oldRidestops["workday"] == True) & (oldRidestops["state"] == "completed")
    ]
    weekendOldRides = oldRidestops[
        (oldRidestops["workday"] == False) & (oldRidestops["state"] == "completed")
    ]

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
        # get pickup ridestop distribution of rides on workdays, which are in a +/- 1h timeframe around the planned departure; And add not considered ridestops with minimal frequency count of used stops
        distPickupWorkday = (
            workdayOldRides[(workdayOldRides["hour"].isin(timeframe))]["pickup_address"]
            .value_counts()
            .rename_axis("pickup_address")
            .reset_index(name="counts")
        )
        distPickupWorkday = distPickupWorkday.merge(
            ridestops["MoDStop Id"],
            left_on="pickup_address",
            how="outer",
            right_on="MoDStop Id",
        )
        distPickupWorkday["pickup_address"] = distPickupWorkday["MoDStop Id"]
        distPickupWorkday = distPickupWorkday.fillna(distPickupWorkday["counts"].min())
        distPickupWorkday["probabilities"] = (
            distPickupWorkday.counts / distPickupWorkday.counts.sum()
        )
        # get dropoff ridestop distribution of rides on workdays, which are in a +/- 1h timeframe around the planned departure; And add not considered ridestops with minimal frequency count of used stops
        distDropoffWorkday = (
            workdayOldRides[(workdayOldRides["hour"].isin(timeframe))][
                "dropoff_address"
            ]
            .value_counts()
            .rename_axis("dropoff_address")
            .reset_index(name="counts")
        )
        distDropoffWorkday = distDropoffWorkday.merge(
            ridestops["MoDStop Id"],
            left_on="dropoff_address",
            how="outer",
            right_on="MoDStop Id",
        )
        distDropoffWorkday["dropoff_address"] = distDropoffWorkday["MoDStop Id"]
        distDropoffWorkday = distDropoffWorkday.fillna(
            distDropoffWorkday["counts"].min()
        )
        distDropoffWorkday["probabilities"] = (
            distDropoffWorkday.counts / distDropoffWorkday.counts.sum()
        )

        ##### weekend ridestop distribution #####
        # get pickup ridestop distribution of rides on workdays, which are in a +/- 1h timeframe around the planned departure; And add not considered ridestops with minimal frequency count of used stops
        distPickupWeekend = (
            weekendOldRides[(weekendOldRides["hour"].isin(timeframe))]["pickup_address"]
            .value_counts()
            .rename_axis("pickup_address")
            .reset_index(name="counts")
        )
        distPickupWeekend = distPickupWeekend.merge(
            ridestops["MoDStop Id"],
            left_on="pickup_address",
            how="outer",
            right_on="MoDStop Id",
        )
        distPickupWeekend["pickup_address"] = distPickupWeekend["MoDStop Id"]
        distPickupWeekend = distPickupWeekend.fillna(distPickupWeekend["counts"].min())
        distPickupWeekend["probabilities"] = (
            distPickupWeekend.counts / distPickupWeekend.counts.sum()
        )
        # get dropoff ridestop distribution of rides on workdays, which are in a +/- 1h timeframe around the planned departure; And add not considered ridestops with minimal frequency count of used stops
        distDropoffWeekend = (
            weekendOldRides[(weekendOldRides["hour"].isin(timeframe))][
                "dropoff_address"
            ]
            .value_counts()
            .rename_axis("dropoff_address")
            .reset_index(name="counts")
        )
        distDropoffWeekend = distDropoffWeekend.merge(
            ridestops["MoDStop Id"],
            left_on="dropoff_address",
            how="outer",
            right_on="MoDStop Id",
        )
        distDropoffWeekend["dropoff_address"] = distDropoffWeekend["MoDStop Id"]
        distDropoffWeekend = distDropoffWeekend.fillna(
            distDropoffWeekend["counts"].min()
        )
        distDropoffWeekend["probabilities"] = (
            distDropoffWeekend.counts / distDropoffWeekend.counts.sum()
        )

        # for all new rides planned at time h choose ridestops based on the distributions
        # pickup_address:
        newRideStops["pickup_address"] = np.where(
            (newRideStops["workday"] == True) & (newRideStops["hour"] == h),
            np.random.choice(
                distPickupWorkday["pickup_address"],
                p=distPickupWorkday["probabilities"],
            ),
            np.where(
                (newRideStops["workday"] == False) & (newRideStops["hour"] == h),
                np.random.choice(
                    distPickupWeekend["pickup_address"],
                    p=distPickupWeekend["probabilities"],
                ),
                newRideStops["pickup_address"],
            ),
        )
        # dropoff_address:
        newRideStops["dropoff_address"] = np.where(
            (newRideStops["workday"] == True) & (newRideStops["hour"] == h),
            np.random.choice(
                distDropoffWorkday["dropoff_address"],
                p=distDropoffWorkday["probabilities"],
            ),
            np.where(
                (newRideStops["workday"] == False) & (newRideStops["hour"] == h),
                np.random.choice(
                    distDropoffWeekend["dropoff_address"],
                    p=distDropoffWeekend["probabilities"],
                ),
                newRideStops["dropoff_address"],
            ),
        )

    # Extract 'distance' and 'shortest_ridetime' based on generated routes
    newRideStops["distance"] = newRideStops.merge(
        routes,
        left_on=["pickup_address", "dropoff_address"],
        right_on=["start_id", "end_id"],
        how="left",
    )["Route [m]"]
    # calculate shortest_ridetime in seconds with average speed of 30 km/h
    newRideStops["shortest_ridetime"] = (
        1 / (30 / (newRideStops["distance"] / 1000)) * 60 * 60
    )
    return newRideStops[
        ["pickup_address", "dropoff_address", "distance", "shortest_ridetime"]
    ]


def generateCreatedAt(oldRides, newRides, m, y):
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
    # use the weekday distribution to represent real occurrences of rides
    daydist["weekday"] = daydist["date"].apply(lambda x: dt.weekday(x))

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
    )  # np.where(daydist['weekday'])
    # normalization neccessary to get probability distribution (sum of odds is 1)
    daydist["probabilities"] = daydist["probabilities"] / (
        daydist["probabilities"].sum()
    )

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
            ),
            minutes=np.random.choice(
                dist_minute["minute"], p=dist_minute["probabilities"]
            ),
            seconds=np.random.choice(list(range(0, 60))),
        )
    )
    return values["created_at"]


def generateScheduledTo(df, newRides):
    hours = pd.DataFrame(columns=["hour"])
    hours["hour"] = newRides["created_at"].apply(lambda x: x.hour)

    # get prebooking time
    scheduled = pd.DataFrame(
        df[["created_at", "scheduled_to"]], columns=["created_at", "scheduled_to"]
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
    mean = scheduled[scheduled["isScheduled"] == True]["prebook_time"].mean()
    std = scheduled[scheduled["isScheduled"] == True]["prebook_time"].std()
    a = 1
    b = scheduled[scheduled["isScheduled"] == True]["prebook_time"].max()
    dist_avg_prebook_time = stats.truncnorm(
        (a - mean) / std, (b - mean) / std, loc=mean, scale=std
    )

    values = [
        (i + pd.Timedelta(dist_avg_prebook_time.rvs(1)[0], unit="seconds")).round(
            freq="5T"
        )
        if j
        in [1, 2, 3, 4, 5, 6]
        | np.random.choice(dist["isScheduled"], p=dist["probabilities"])
        else i
        for i, j in zip(newRides.created_at, hours.hour)
    ]
    # we have no rides before 7
    values = [
        dt(i.year, i.month, i.day, 7, 0) if j in [1, 2, 3, 4, 5, 6] else i
        for i, j in zip(newRides.created_at, hours.hour)
    ]

    return values


def generateValues(column_name, df, newRides):
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


def generateRideSpecs(oldRides, newRides, ridestops, routes, n, month, year):
    timestamp = str(round(time.time()))
    newRides["id"] = [timestamp + "-" + str(x) for x in list(range(0, n))]
    # Ein Kunde mehrere Rides
    newRides["user_id"] = [str(x) + "-" + timestamp for x in list(range(0, n))]
    newRides["number_of_passenger"] = generateValues(
        "number_of_passenger", oldRides, newRides
    )
    newRides["free_ride"] = generateValues("free_ride", oldRides, newRides)
    newRides["payment_type"] = generateValues("payment_type", oldRides, newRides)
    newRides["state"] = "completed"
    newRides["arrival_indicator"] = generateValues(
        "arrival_indicator", oldRides, newRides
    )
    # zufällig ratings rein, die nicht bisher gerated wurden? Oder Rating ganz raus?
    newRides["rating"] = generateValues("rating", oldRides, newRides)
    newRides["created_at"] = generateCreatedAt(oldRides, newRides, month, year)
    newRides["scheduled_to"] = generateScheduledTo(oldRides, newRides)
    # newRides[['pickup_address', 'dropoff_address','distance', 'shortest_ridetime']] = generateRoute(oldRides, newRides, ridestops, routes) # prices are not considered
    # newRides[['pickup_address', 'dropoff_address','distance', 'shortest_ridetime']] = generateRoute_simple(oldRides, newRides, ridestops, routes) # prices are not considered
    newRides[
        ["pickup_address", "dropoff_address", "distance", "shortest_ridetime"]
    ] = generateRoute_simple2(
        oldRides, newRides, ridestops, routes
    )  # prices are not considered

    return newRides


def generateRoute_simple2(oldRides, newRides, ridestops, routes):
    oldRideStops = oldRides[["pickup_address", "dropoff_address"]]
    oldRideStops["route"] = (
        oldRideStops["pickup_address"].astype(str)
        + "-"
        + oldRideStops["dropoff_address"].astype(str)
    )

    dist = (
        oldRideStops["route"]
        .value_counts()
        .rename_axis("route")
        .reset_index(name="counts")
    )
    dist["probabilities"] = dist.counts / dist.counts.sum()

    newRideStops = pd.DataFrame(
        newRides[["pickup_address", "dropoff_address"]],
        columns=["pickup_address", "dropoff_address"],
    )
    newRideStops["route"] = np.random.choice(
        dist["route"], p=dist["probabilities"], size=newRides.shape[0]
    )
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
    # calculate shortest_ridetime in seconds with average speed of 30 km/h
    newRideStops["shortest_ridetime"] = (
        1 / (30 / (newRideStops["distance"] / 1000)) * 60 * 60
    )
    return newRideStops[
        ["pickup_address", "dropoff_address", "distance", "shortest_ridetime"]
    ]
