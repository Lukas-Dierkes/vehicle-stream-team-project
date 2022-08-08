import collections
import json
import time
from datetime import datetime as dt
from scipy.stats import truncnorm

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
    hotspots = [i[0] for i in counter.most_common(10)]
    
    return hotspots


def add_drone_flights(df_edges, drives, drone_spots=[1008], radius=500):
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

# help function that returns a probability distribution for continous variables based on mean & standard deviation
def getdistribution(data, column, min=None, max=None):
    # distribution over scheduled rides
    mean = data[column].median() # use median to better deal with outlier
    std = data[column].std()
    if min==None: 
        a = data[column].min() # min value
    else:
        a = min 
    if max==None:
        b = data[column].max() # max value
    else:
        b = max
    return stats.truncnorm((a - mean) / std, (b - mean) / std, loc=mean, scale=std)

# TODO: Check in the end if too many rides, which are too short or are not likely enough
def generateRoute(oldRides, newRides, ridestops, routes):
    # add route identifier to routes dataframe
    allRoutes = routes[routes['Route [m]'] > 500] # Assumption: real rides are at least 500 m long
    allRoutes['route'] = allRoutes['Start #'].astype(str) + "-" + allRoutes['Ende #'].astype(str)

    # based on analysis of rides we distinguish between workdays (Monday till Friday noon) and weekend (Friday noon till Sunday)
    newRideStops = pd.DataFrame(newRides[['created_at', 'scheduled_to', 'pickup_address', 'dropoff_address']], columns=['created_at', 'scheduled_to', 'pickup_address', 'dropoff_address'])
    newRideStops['route'] = ""
    newRideStops['day'] = newRideStops['scheduled_to'].apply(lambda x: dt.weekday(x))
    newRideStops['hour'] = newRideStops['scheduled_to'].apply(lambda x: x.hour)
    newRideStops['workday'] = np.where(
        (
            newRideStops['day'].isin([0,1,2,3,4]) # 0 = Monday, 6 = Sunday
            & ~(
                    (newRideStops['day'] == 4) 
                    & (newRideStops['hour'] > 13)
            )
        ),
        True,
        False
    )

    oldRidestops = pd.DataFrame(oldRides[['state', 'scheduled_to', 'pickup_address', 'dropoff_address']], columns=['state', 'scheduled_to', 'pickup_address', 'dropoff_address'])
    oldRidestops['route'] = oldRidestops['pickup_address'].astype(str) + "-" + oldRidestops['dropoff_address'].astype(str)
    oldRidestops['scheduled_to'] = pd.to_datetime(oldRidestops['scheduled_to'])
    oldRidestops['day'] = oldRidestops['scheduled_to'].apply(lambda x: dt.weekday(x))
    oldRidestops['hour'] = oldRidestops['scheduled_to'].apply(lambda x: x.hour)
    oldRidestops['workday'] = np.where(
        (
            oldRidestops['day'].isin([0,1,2,3,4]) # 0 = Monday, 6 = Sunday
            & ~(
                    (oldRidestops['day'] == 4) 
                    & (oldRidestops['hour'] > 13)
            )
        ),
        True,
        False
    )
    workdayOldRides = oldRidestops[(oldRidestops['workday']==True)]
    weekendOldRides = oldRidestops[(oldRidestops['workday']==False)]

    # generate ridestops
    for h in [0] + list(range(7,24)): # rides start between 7:00 and 0:59
        # timeframe used to get ridestop distribution
        if h in [23,0]:
            timeframe = [22,23,0]
        elif h == 7:
            timeframe = [7,8,9]
        else:
            timeframe = list(range(h-1,h+2))

        ##### workday ridestop distribution #####
        distWorkday = workdayOldRides[(workdayOldRides['hour'].isin(timeframe))]['route'].value_counts().rename_axis('route').reset_index(name='counts')
        numberOfNoise = distWorkday['counts'].sum() / 80 * 20 # 20% noise
        allRoutes['counts'] = distWorkday['counts'].min() # noise is weighted similar to least frequent real driven route
        distWorkday = pd.concat([distWorkday, allRoutes[~allRoutes['route'].isin(distWorkday['route'])].sample(frac=1)[:int(numberOfNoise)][['route', 'counts']]])
        distWorkday['probabilities'] = (distWorkday.counts / distWorkday.counts.sum())

        ##### weekend ridestop distribution #####
        distWeekend = weekendOldRides[(weekendOldRides['hour'].isin(timeframe))]['route'].value_counts().rename_axis('route').reset_index(name='counts')
        numberOfNoise = distWeekend['counts'].sum() / 80 * 20 # 20% noise
        allRoutes['counts'] = distWeekend['counts'].min() # noise is weighted similar to least frequent real driven route
        distWeekend = pd.concat([distWeekend, allRoutes[~allRoutes['route'].isin(distWeekend['route'])].sample(frac=1)[:int(numberOfNoise)][['route', 'counts']]])
        distWeekend['probabilities'] = (distWeekend.counts / distWeekend.counts.sum())

        # split newRideStops dataframe in 1. ride-hour=h & weekend, 2. ride-hour=h & workday, 3. rest
        newRideStops_h_wend = newRideStops[(newRideStops['hour']==h) & (newRideStops['workday']==False)]
        newRideStops_h_work = newRideStops[(newRideStops['hour']==h) & (newRideStops['workday']==True)]
        newRideStops_not_h = newRideStops[~((newRideStops['hour']==h) & (newRideStops['workday']==False)) & ~((newRideStops['hour']==h) & (newRideStops['workday']==True))]

        # generate routes based on distributions
        newRideStops_h_wend['route'] = np.random.choice(distWeekend['route'], p=distWeekend['probabilities'], size=newRideStops_h_wend.shape[0])
        newRideStops_h_work['route'] = np.random.choice(distWorkday['route'], p=distWorkday['probabilities'], size=newRideStops_h_work.shape[0])

        # concat 3 pieces back together
        newRideStops = pd.concat([newRideStops_not_h, newRideStops_h_wend, newRideStops_h_work])

    # Extract pickup & dropoff address from route column
    newRideStops[['pickup_address', 'dropoff_address']] = newRideStops['route'].str.split('-', expand=True)
    newRideStops['pickup_address'] = pd.to_numeric(newRideStops['pickup_address'])
    newRideStops['dropoff_address'] = pd.to_numeric(newRideStops['dropoff_address'])

    # Extract 'distance' and 'shortest_ridetime' based on generated routes
    newRideStops['distance'] = newRideStops.merge(routes, left_on=['pickup_address', 'dropoff_address'], right_on=['Start #', 'Ende #'], how='left')['Route [m]']
    newRideStops['shortest_ridetime'] = 1/(30 / (newRideStops['distance'] / 1000) )*60*60 # calculate shortest_ridetime in seconds with average speed of 30 km/h
    newRideStops.sort_values(by=['created_at'])
    return newRideStops[['pickup_address', 'dropoff_address','distance', 'shortest_ridetime']]

# function that returns n random 'created_at' timestamps over a period of one specified month based on the probability distribution in original data
# first step: choose a date from the month based on the probability distribution of rides over the weekdays (Monday-Sunday)
# second step: choose a timestamp based on the probability distribution of rides that are on the same weekday
def generateCreatedAt(oldRides, newRides, m, y):
    # creat list with all days of the month to build up the probability distribution 
    if m == 12:
        m1 = 1
        y1 = y + 1
    else:
        m1 = m + 1
        y1 = y
    daydist = pd.DataFrame(pd.date_range(start=str(m)+'/01/'+str(y), end=str(m1)+'/01/'+str(y1),).to_pydatetime().tolist()[:-1], columns=['date'])
    daydist['weekday'] = daydist['date'].apply(lambda x: dt.weekday(x)) # use the weekday distribution to represent real occurrences of rides

    # extract all dates and their weekday, hour and minute 
    created = pd.DataFrame(pd.to_datetime(oldRides['created_at']), columns=['created_at'])
    created['day'] = created['created_at'].apply(lambda x: dt.weekday(x))
    created['hour'] = created['created_at'].apply(lambda x: x.hour)
    created['minute'] = created['created_at'].apply(lambda x: x.minute)
    
    # get the weekday distribution of old rides
    dist_day = created['day'].value_counts().rename_axis('day').reset_index(name='counts')
    dist_day['probabilities'] = (dist_day.counts / dist_day.counts.sum())
    dist_day = dist_day.sort_values('day')

    # get the hour distribution of old rides per weekday 
    dist_hour = []
    for i in range(0,7):
        dist_hour.append(created[created['day']==i]['hour'].value_counts().rename_axis('hour').reset_index(name='counts'))
        dist_hour[i]['probabilities'] = (dist_hour[i].counts / dist_hour[i].counts.sum())
        dist_hour[i] = dist_hour[i].sort_values('hour')

    
    # get the minute distribution of old rides
    dist_minute = created['minute'].value_counts().rename_axis('minute').reset_index(name='counts')
    dist_minute['probabilities'] = (dist_minute.counts / dist_minute.counts.sum())  
    dist_minute = dist_minute.sort_values('minute')


    # match probability that a ride is on that weekday to all dates in the simulated month
    daydist['probabilities'] =  daydist['weekday'].apply(lambda x: dist_day[dist_day['day']==x]['probabilities'].values[0]) 
    daydist['probabilities'] = daydist['probabilities']/(daydist['probabilities'].sum()) # normalization neccessary to get probability distribution (sum of odds is 1)

    # generate list of values
    values = pd.DataFrame(np.random.choice(daydist['date'], p=daydist['probabilities'], size=newRides.shape[0]), columns=['created_at'])
    values = values.sort_values('created_at')
    values = values.reset_index()
    values['day'] = values['created_at'].apply(lambda x: dt.weekday(x))
    values['created_at'] = values['created_at'] + values['day'].apply(
        lambda x: pd.Timedelta(
            hours=np.random.choice(dist_hour[x]['hour'], p=dist_hour[x]['probabilities']), # choose hour based on distribution of that weekday
            minutes=np.random.choice(dist_minute['minute'], p=dist_minute['probabilities']), # choose minute based on distribution of that hour
            seconds=np.random.choice(list(range(0,60))) # random choice of seconds
        )
    )
    values.sort_values(by=['created_at'])
    return values['created_at']

# function that returns n random 'scheduled_to' timestamps based on the probability distribution in original data
# first, consider distribution of scheduled & immediate 
# second, for a scheduled ride add a random prebooking time (based on probability distribution of the prebooking time in original data) to created_at
def generateScheduledTo(oldRides, newRides):
    scheduledNew = pd.DataFrame(columns=['hour'])
    scheduledNew['created_at'] = newRides['created_at']
    scheduledNew['hour'] = scheduledNew['created_at'].apply(lambda x: x.hour)

    # get prebooking time
    scheduled = pd.DataFrame(oldRides[['created_at', 'scheduled_to']], columns=['created_at', 'scheduled_to'])
    scheduled['isScheduled'] = (scheduled.created_at != scheduled.scheduled_to)
    scheduled['created_at'] = pd.to_datetime(scheduled['created_at'])
    scheduled['scheduled_to'] = pd.to_datetime(scheduled['scheduled_to'])
    scheduled['prebook_time'] = scheduled.scheduled_to - scheduled.created_at
    scheduled['prebook_time'] = scheduled['prebook_time'].apply(lambda x: x.total_seconds())
    
    # distribution of prebooked and non-prebooked rides
    dist = scheduled['isScheduled'].value_counts().rename_axis('isScheduled').reset_index(name='counts')
    dist['probabilities'] = (dist.counts / dist.counts.sum())

    # distribution of average prebook time 
    left_border = 8*60 # min value of 8 min -> assumption: scheduled ride must be at least 8 min in the future
    dist_avg_prebook_time = getdistribution(scheduled[scheduled['isScheduled'] == True], 'prebook_time', min=left_border)

    scheduledNew['scheduled_to'] = [(i + pd.Timedelta(dist_avg_prebook_time.rvs(1)[0], unit='seconds')).round(freq='10min') if np.random.choice(dist['isScheduled'], p=dist['probabilities']) else i for i, j in zip(scheduledNew.created_at, scheduledNew.hour)]
    # we have no rides before 7
    scheduledNew['hour'] = scheduledNew['scheduled_to'].apply(lambda x: x.hour)
    scheduledNew['scheduled_to'] = [dt(i.year, i.month, i. day, 7, 0) if j in [1, 2, 3, 4, 5, 6] else i for i, j in zip(scheduledNew.scheduled_to, scheduledNew.hour)]
    return scheduledNew['scheduled_to']

# function that returns n random 'dispatched_at' timestamps
# case 1: scheduled ride -> dispatched_at = scheduled_at - 8 min
# case 2: immediate ride -> dispatched_at = scheduled_at
def generateDispatchedAt(oldRides, newRides):
    scheduled = pd.DataFrame(newRides[['created_at', 'scheduled_to']], columns=['created_at', 'scheduled_to'])
    scheduled['isScheduled'] = (scheduled.created_at != scheduled.scheduled_to)
    scheduled['created_at'] = pd.to_datetime(scheduled['created_at'])
    scheduled['scheduled_to'] = pd.to_datetime(scheduled['scheduled_to'])
    scheduled['dispatched_at'] = np.where(
        (scheduled['isScheduled']==True),
        np.where(
            (scheduled['scheduled_to'] - pd.Timedelta(minutes=8) > scheduled['created_at']),
            scheduled['scheduled_to'] - pd.Timedelta(minutes=8),
            scheduled['created_at']
        ),
        scheduled['scheduled_to']
    )
    return scheduled['dispatched_at']

# function that returns n random timestamps for 'arriving_push', vehicle_arrived_at', 'pickup_arrival_time'
# first, generate 'vehicle_arrived_at'
# for scheduled rides: scheduled_to + random scheduling deviation (based on probability distribution of scheduling_deviation=vehicle_arrived_at-scheduled_to in original data)
# for immediatie rides: scheduled_to + random pickup_arrival_time (based on probability distribution of pickup_arrival_time in original data)
# second, calculate pickup_arrival_time=vehicle_arrived_at-dispatched_at etc.
def generateArrival(oldRides, newRides):
    # get needed information regarding the vehicle arrival in old data
    arrivalOld = pd.DataFrame(oldRides[['created_at', 'scheduled_to', 'dispatched_at', 'vehicle_arrived_at', 'pickup_arrival_time' ]], columns=['created_at', 'scheduled_to', 'dispatched_at', 'vehicle_arrived_at', 'pickup_arrival_time'])
    arrivalOld['isScheduled'] = (arrivalOld.created_at != arrivalOld.scheduled_to)
    arrivalOld['created_at'] = pd.to_datetime(arrivalOld['created_at'])
    arrivalOld['scheduled_to'] = pd.to_datetime(arrivalOld['scheduled_to'])
    arrivalOld['vehicle_arrived_at'] = pd.to_datetime(arrivalOld['vehicle_arrived_at'])
    # arrivalOld['arriving_push'] = pd.to_datetime(arrivalOld['arriving_push'])

    # create dataframe with needed attributes to determine 'vehicle_arrived_at'
    arrivalNew = pd.DataFrame(newRides[['created_at', 'scheduled_to', 'dispatched_at']], columns=['created_at', 'scheduled_to', 'dispatched_at'])
    arrivalNew['isScheduled'] = (arrivalNew.created_at != arrivalNew.scheduled_to)
    arrivalNew['created_at'] = pd.to_datetime(arrivalNew['created_at'])
    arrivalNew['scheduled_to'] = pd.to_datetime(arrivalNew['scheduled_to'])
    arrivalNew['dispatched_at'] = pd.to_datetime(arrivalNew['dispatched_at'])
    
    ##### generate timestamp 'vehicle_arrived_at' 
    arrivalOld['schedule_deviation'] = arrivalOld.apply(
        lambda row: (
            (row["vehicle_arrived_at"] - row["scheduled_to"] ).round(freq="s")
        ).total_seconds(),
        axis=1,
    )

    # distribution over scheduled rides
    left_border = -8*60 # min value of -8 minutes -> earliest arrived_at
    dist_scheduledRides = getdistribution(arrivalOld[arrivalOld['isScheduled'] == True], 'schedule_deviation', min=left_border )

    # distribution over instant rides - based on pickup_arrival_times distribution
    left_border = 1 # min value of 1 second -> earliest arrived_at
    dist_instantRides = getdistribution(arrivalOld[arrivalOld['isScheduled'] == False], 'pickup_arrival_time', min=left_border )


    # determine timestamp 'vehicle_arrived_at' 
    arrivalNew['vehicle_arrived_at'] = arrivalNew.apply(
        lambda row: (
            (row["scheduled_to"] + pd.Timedelta(dist_scheduledRides.rvs(1)[0], unit='seconds').round(freq="s"))
        )
        if (row["isScheduled"] == True)
        else 
            (row["scheduled_to"] + pd.Timedelta(dist_instantRides.rvs(1)[0], unit='seconds').round(freq="s")),
        axis=1,
    )
    # check that vehicle_arrived_at is after dispatched_at
    arrivalNew['vehicle_arrived_at'] = np.where(
        arrivalNew['dispatched_at'] > arrivalNew['vehicle_arrived_at'],
        arrivalNew['dispatched_at'] + (arrivalNew['scheduled_to'] -  arrivalNew['dispatched_at']) * np.random.uniform(0.1,0.9),
        arrivalNew['vehicle_arrived_at']
    )

    ##### calculate 'pickup_arrival_time'
    arrivalNew["pickup_arrival_time"] = (
        arrivalNew["vehicle_arrived_at"] - arrivalNew["dispatched_at"]
    ).dt.seconds

    return arrivalNew[[ 'vehicle_arrived_at', 'pickup_arrival_time']]

def generatePickup(oldRides, newRides):
    # get needed information regarding the pickup in old data
    pickupOld = pd.DataFrame(oldRides[['arriving_push','created_at', 'scheduled_to', 'dispatched_at', 'vehicle_arrived_at', 'pickup_at', 'pickup_first_eta', 'pickup_eta']], columns=['arriving_push','created_at', 'scheduled_to', 'dispatched_at', 'vehicle_arrived_at', 'pickup_at', 'pickup_first_eta', 'pickup_eta'])
    pickupOld[['arriving_push','created_at', 'scheduled_to', 'dispatched_at', 'vehicle_arrived_at', 'pickup_at', 'pickup_first_eta', 'pickup_eta']] = pickupOld[['arriving_push','created_at', 'scheduled_to', 'dispatched_at', 'vehicle_arrived_at', 'pickup_at', 'pickup_first_eta', 'pickup_eta']].apply(pd.to_datetime)
    pickupOld['isScheduled'] = (pickupOld.created_at != pickupOld.scheduled_to)
    pickupOld['arriving_push'] = pd.to_datetime(pickupOld['arriving_push'])

    # create dataframe with needed attributes to determine pickup attributes
    pickupNew = pd.DataFrame(newRides[['created_at', 'scheduled_to', 'dispatched_at', 'vehicle_arrived_at']], columns=['created_at', 'scheduled_to', 'dispatched_at', 'vehicle_arrived_at'])
    pickupNew['isScheduled'] = (pickupNew.created_at != pickupNew.scheduled_to)
    pickupNew[['created_at', 'scheduled_to', 'dispatched_at', 'vehicle_arrived_at']] = pickupNew[['created_at', 'scheduled_to', 'dispatched_at', 'vehicle_arrived_at']].apply(pd.to_datetime)
    
    ##### generate earliest_pickup_expectation
    pickupNew['earliest_pickup_expectation'] = pickupNew['dispatched_at'] + pd.Timedelta(minutes=3)

    ##### genrate pickup_at 
    pickupOld['time_until_pickup'] = pickupOld.apply(
        lambda row: (
            (row["pickup_at"] - row["vehicle_arrived_at"] ).round(freq="s")
        ).total_seconds(),
        axis=1,
    )
    # distribution of the time a driver waits until pickup over scheduled rides
    left_border = 1 # min value of 1 second -> earliest arrived_at
    dist_scheduledRides = getdistribution(pickupOld[pickupOld['isScheduled'] == True], 'time_until_pickup', min=left_border )

    # distribution of the time a driver waits until pickup over instant rides
    left_border = 1 # min value of 1 second -> earliest arrived_at
    dist_instantRides = getdistribution(pickupOld[pickupOld['isScheduled'] == False], 'time_until_pickup', min=left_border )

    # determine timestamp 'pickup_at' 
    pickupNew['pickup_at'] = pickupNew.apply(
        lambda row: (
            (row["vehicle_arrived_at"] + pd.Timedelta(dist_scheduledRides.rvs(1)[0], unit='seconds').round(freq="s"))
        )
        if (row["isScheduled"] == True)
        else 
            (row["vehicle_arrived_at"] + pd.Timedelta(dist_instantRides.rvs(1)[0], unit='seconds').round(freq="s")),
        axis=1,
    )

    ##### generate pickup_eta
    # distribution of the time between pickup_at and pickup_eta
    pickupOld['deviation_of_pickup_eta'] = pickupOld.apply(
        lambda row: (
            (row["pickup_eta"] - row["pickup_at"] ).round(freq="s")
        ).total_seconds(),
        axis=1,
    )
    dist = getdistribution(pickupOld, 'deviation_of_pickup_eta')

    # determine timestamp 'pickup_eta' 
    pickupNew['pickup_eta'] = pickupNew.apply(
        lambda row: (
            (row["pickup_at"] + pd.Timedelta(dist.rvs(1)[0], unit='seconds').round(freq="s"))
        ),
        axis=1,
    )

    # check that pickup_eta is after dispatched_at  
    pickupNew['pickup_eta'] = np.where(
        pickupNew['dispatched_at'] > pickupNew['pickup_eta'],
        pickupNew['dispatched_at'] + pd.Timedelta(minutes=3), #TODO: mehr randomness
        pickupNew['pickup_eta']
    )

    ##### generate pickup_first_eta
    # distribution of the time between pickup_at and pickup_first_eta
    pickupOld['deviation_of_pickup_first_eta'] = pickupOld.apply(
        lambda row: (
            (row["pickup_first_eta"] - row["pickup_at"] ).round(freq="s")
        ).total_seconds(),
        axis=1,
    )
    dist = getdistribution(pickupOld, 'deviation_of_pickup_first_eta')

    # determine timestamp 'pickup_first_eta' 
    pickupNew['pickup_first_eta'] = pickupNew.apply(
        lambda row: (
            (row["pickup_at"] + pd.Timedelta(dist.rvs(1)[0], unit='seconds').round(freq="s"))
        ),
        axis=1,
    )

    # check that pickup_first_eta is at least 3 min. after created_at 
    pickupNew['pickup_first_eta'] = np.where(
        (pickupNew['created_at'] + pd.Timedelta(minutes=3)) > pickupNew['pickup_first_eta'],
        pickupNew['created_at'] + pd.Timedelta(minutes=3), # created_at + 3 min. is minimum
        pickupNew['pickup_first_eta']
    )
    # check that pickup_first_eta is after dispatched_at 
    pickupNew['pickup_first_eta'] = np.where(
        pickupNew['dispatched_at'] > pickupNew['pickup_first_eta'],
        pickupNew['dispatched_at'] + pd.Timedelta(minutes=3), # TODO: mehr randomness? 
        pickupNew['pickup_first_eta']
    )

    # check that pickup_first_eta before pickup_eta
    pickupNew['pickup_first_eta'] = np.where(
        pickupNew['pickup_first_eta'] > pickupNew['pickup_eta'],
        pickupNew['pickup_eta'],
        pickupNew['pickup_first_eta']
    )


    ##### generate arriving_push
    # distribution of the time between arriving_push and vehicle_arrived_at
    pickupOld['deviation_of_arriving_push'] = pickupOld.apply(
        lambda row: (
            (row["arriving_push"] - row["pickup_eta"] ).round(freq="s")
        ).total_seconds(),
        axis=1,
    )
    dist = getdistribution(pickupOld, 'deviation_of_arriving_push')

    # determine timestamp 'arriving_push' 
    pickupNew['arriving_push'] = pickupNew.apply(
        lambda row: (
            (row["pickup_eta"] + pd.Timedelta(dist.rvs(1)[0], unit='seconds').round(freq="s"))
        ),
        axis=1,
    )

    # check that arriving_push is after dispatched_at  
    pickupNew['arriving_push'] = np.where(
        pickupNew['dispatched_at'] > pickupNew['arriving_push'],
        pickupNew['dispatched_at'] + (pickupNew['pickup_eta'] -  pickupNew['dispatched_at']) * np.random.uniform(0.1,0.9),
        pickupNew['arriving_push']
    )

    return pickupNew[['arriving_push','earliest_pickup_expectation', 'pickup_at', 'pickup_eta', 'pickup_first_eta']]

def generateDropoff(oldRides, newRides, routes):
    # get needed information regarding the dropoff in old data
    dropoffOld = pd.DataFrame(oldRides[['pickup_address', 'dropoff_address', 'scheduled_to', 'dropoff_at', 'dropoff_first_eta', 'dropoff_eta', 'ride_time']], columns=['pickup_address', 'dropoff_address', 'scheduled_to', 'dropoff_at', 'dropoff_first_eta', 'dropoff_eta', 'ride_time'])
    dropoffOld[['scheduled_to', 'dropoff_at', 'dropoff_first_eta', 'dropoff_eta']] = dropoffOld[['scheduled_to', 'dropoff_at', 'dropoff_first_eta', 'dropoff_eta']].apply(pd.to_datetime)
    dropoffOld['day'] = dropoffOld['scheduled_to'].apply(lambda x: dt.weekday(x))
    dropoffOld['hour'] = dropoffOld['scheduled_to'].apply(lambda x: x.hour)
    dropoffOld['workday'] = np.where(
        (
            dropoffOld['day'].isin([0,1,2,3,4]) # 0 = Monday, 6 = Sunday
            & ~(
                    (dropoffOld['day'] == 4) 
                    & (dropoffOld['hour'] > 13)
            )
        ),
        True,
        False
    )

    # create dataframe with needed attributes to determine dropoff attributes
    dropoffNew = pd.DataFrame(newRides[['pickup_address', 'dropoff_address', 'scheduled_to', 'pickup_at', 'pickup_first_eta', 'pickup_eta', 'shortest_ridetime']], columns=['pickup_address', 'dropoff_address', 'scheduled_to', 'pickup_at', 'pickup_first_eta', 'pickup_eta', 'shortest_ridetime'])
    dropoffNew[['scheduled_to', 'pickup_at', 'pickup_first_eta', 'pickup_eta']] = dropoffNew[['scheduled_to', 'pickup_at', 'pickup_first_eta', 'pickup_eta']].apply(pd.to_datetime)
    dropoffNew['day'] = dropoffNew['scheduled_to'].apply(lambda x: dt.weekday(x))
    dropoffNew['hour'] = dropoffNew['scheduled_to'].apply(lambda x: x.hour)
    dropoffNew['timeframe'] = dropoffNew['hour'].apply(
        lambda h: (
            [22,23,0]
            if h in [23,0]
            else 
            ([7,8,9]
            if h == 7
            else
            list(range(h-1,h+2))) 
            )
    )
    dropoffNew['workday'] = np.where(
        (
            dropoffNew['day'].isin([0,1,2,3,4]) # 0 = Monday, 6 = Sunday
            & ~(
                    (dropoffNew['day'] == 4) 
                    & (dropoffNew['hour'] > 13)
            )
        ),
        True,
        False
    )

    ##### generate ride_time based on ride_time of most similar rides
    dropoffNew['ride_time'] = dropoffNew.apply(
        lambda row:
            # if rides exist with same route & workday/weekend flag & in a timeframe of +/-1 hour
            round(dropoffOld[(dropoffOld['pickup_address']==row['pickup_address']) 
                & (dropoffOld['dropoff_address']==row['dropoff_address'])
                & (dropoffOld['workday']==row['workday'])          
                & (dropoffOld['hour'].isin(row['timeframe']))]['ride_time'].mean())
            if len(dropoffOld[(dropoffOld['pickup_address']==row['pickup_address']) 
                & (dropoffOld['dropoff_address']==row['dropoff_address'])
                & (dropoffOld['workday']==row['workday'])                   
                & (dropoffOld['hour'].isin(row['timeframe']))]['ride_time']) > 0
            else
            # if rides exist with same route & in a timeframe of +/-1 hour - workday/weekend does not matter
            round(dropoffOld[(dropoffOld['pickup_address']==row['pickup_address']) 
                & (dropoffOld['dropoff_address']==row['dropoff_address'])
                & (dropoffOld['hour'].isin(row['timeframe']))]['ride_time'].mean())
            if len(dropoffOld[(dropoffOld['pickup_address']==row['pickup_address']) 
                & (dropoffOld['dropoff_address']==row['dropoff_address'])
                & (dropoffOld['hour'].isin(row['timeframe']))]['ride_time']) > 0
            else
            # if rides exist with same route - day & hour does not matter
            round(dropoffOld[(dropoffOld['pickup_address']==row['pickup_address']) 
                & (dropoffOld['dropoff_address']==row['dropoff_address'])]['ride_time'].mean())
            if len(dropoffOld[(dropoffOld['pickup_address']==row['pickup_address']) 
                & (dropoffOld['dropoff_address']==row['dropoff_address'])]['ride_time']) > 0
            else
            # else, use shortest ridetime: 30km/h over distance of the route
            round((routes[(routes['Start #']==row['pickup_address']) 
                & (routes['Ende #']==row['dropoff_address'])]['Route [m]'].values[0] * 3600 / 30000) * np.random.uniform(1.0,1.2)),
            axis=1,
    )

    ##### genereate dropoff_at 
    dropoffNew['dropoff_at'] = dropoffNew['pickup_at'] + pd.to_timedelta(dropoffNew['ride_time'], unit='seconds')

    ##### generate dropoff_first_eta
    dropoffNew['dropoff_first_eta'] = dropoffNew['pickup_first_eta'] + pd.to_timedelta(dropoffNew['shortest_ridetime'], unit='seconds')

    ##### generate dropoff_eta
    # distribution of the time between dropoff_at and dropoff_eta
    dropoffOld['deviation_of_dropoff_eta'] = dropoffOld.apply(
        lambda row: (
            (row["dropoff_eta"] - row["dropoff_at"] ).round(freq="s")
        ).total_seconds(),
        axis=1,
    )
    dist = getdistribution(dropoffOld, 'deviation_of_dropoff_eta')

    # determine timestamp 'dropoff_eta' 
    dropoffNew['dropoff_eta'] = dropoffNew.apply(
        lambda row: (
            (row["dropoff_at"] + pd.Timedelta(dist.rvs(1)[0], unit='seconds').round(freq="s"))
        ),
        axis=1,
    )

    # check that dropoff_eta is after pickup_eta & pickup_at
    dropoffNew['dropoff_eta'] = np.where(
        (dropoffNew['pickup_eta'] > dropoffNew['dropoff_eta'])
        | (dropoffNew['pickup_at'] > dropoffNew['dropoff_eta']),
        dropoffNew['dropoff_at'] + pd.Timedelta(minutes=3), # TODO: mehr randomness
        dropoffNew['dropoff_eta']
    )


    return dropoffNew[['dropoff_at', 'dropoff_eta', 'dropoff_first_eta']]

# general function that returns n random values based on the probability distribution of a certain column
# used for the following ride attributes: number_of_passenger, free_ride, payment_type, arrival_indicator, rating
def generateValues(column_name, df, newRides):
    dist = df[column_name].value_counts().rename_axis(column_name).reset_index(name='counts')
    dist['probabilities'] = (dist.counts / dist.counts.sum())
    return np.random.choice(dist[column_name], p=dist['probabilities'], size=newRides.shape[0])

# Attributes: ['pickup_arrival_time', 'arrival_deviation', 'waiting_time', 'boarding_time', 'ride_time', 'trip_time', 'shortest_ridetime', 'delay', 'longer_route_factor']
def generateTimeperiods(newRides):
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

    return newRides[['arrival_deviation', 'waiting_time', 'boarding_time', 'ride_time', 'trip_time', 'delay', 'longer_route_factor']] 

def generateRideSpecs(oldRides, ridestops, routes, n, month, year):
    timestamp = str(round(time.time()))
    newRides = pd.DataFrame(columns=oldRides.columns)
    oldRides = oldRides[oldRides['state']=='completed']
    newRides['id'] = [timestamp + '-' + str(x) for x in list(range(0,n))]
    newRides['user_id'] = [str(x) + '-' + timestamp for x in list(range(0,n))] # Ein Kunde mehrere Rides
    newRides['number_of_passenger'] = generateValues('number_of_passenger', oldRides, newRides)
    newRides['free_ride'] = generateValues('free_ride', oldRides, newRides)
    newRides['payment_type'] = generateValues('payment_type', oldRides, newRides)
    newRides['state'] = 'completed'
    newRides['arrival_indicator'] = generateValues('arrival_indicator', oldRides, newRides)
    newRides['rating'] = generateValues('rating', oldRides, newRides) #zufällig ratings rein, die nicht bisher gerated wurden? Oder Rating ganz raus?
    newRides['created_at'] = generateCreatedAt(oldRides, newRides, month, year)
    newRides['scheduled_to'] = generateScheduledTo(oldRides, newRides)
    newRides[['pickup_address', 'dropoff_address','distance', 'shortest_ridetime']] = generateRoute(oldRides, newRides, ridestops, routes) # prices are not considered
    # newRides[['pickup_address', 'dropoff_address','distance', 'shortest_ridetime']] = generateRoute_simple(oldRides, newRides, ridestops, routes) # prices are not considered
    # newRides[['pickup_address', 'dropoff_address','distance', 'shortest_ridetime']] = generateRoute_simple2(oldRides, newRides, ridestops, routes) # prices are not considered
    newRides['dispatched_at'] = generateDispatchedAt(oldRides, newRides)
    newRides[['vehicle_arrived_at', 'pickup_arrival_time']] = generateArrival(oldRides, newRides)
    newRides[['arriving_push','earliest_pickup_expectation', 'pickup_at', 'pickup_eta', 'pickup_first_eta']] = generatePickup(oldRides, newRides)
    newRides[['dropoff_at', 'dropoff_eta', 'dropoff_first_eta']] = generateDropoff(oldRides, newRides, routes)
    newRides[['arrival_deviation', 'waiting_time', 'boarding_time', 'ride_time', 'trip_time', 'delay', 'longer_route_factor']] = generateTimeperiods(newRides)
    return newRides

