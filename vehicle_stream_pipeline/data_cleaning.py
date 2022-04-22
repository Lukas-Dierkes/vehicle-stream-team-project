import datetime
import sched
from itertools import count

import numpy as np
import pandas as pd


def clean_created_at(df):
    created_at = pd.to_datetime(df["created_at"])
    return created_at


def clean_scheduled_to(df):
    scheduled_to = pd.to_datetime(df["scheduled_to"])
    scheduled_to = scheduled_to.fillna(df["created_at"])
    return scheduled_to


def clean_dispatched_at(df):
    # Cast to correct dtype
    dispatched_at = pd.to_datetime(df["dispatched_at"])

    # Fill missing values of dispatched_at
    dispatched_at = np.where(
        dispatched_at.isna(),
        np.where(
            (dispatched_at.isna()) & (np.logical_not(df["scheduled_to"].isna())),
            np.where(
                df["scheduled_to"] - pd.Timedelta(minutes=8) < df["created_at"],
                df["created_at"],
                df["scheduled_to"] - pd.Timedelta(minutes=8),
            ),
            df["created_at"],
        ),
        df["dispatched_at"],
    )

    return dispatched_at


def clean_vehicle_arrived_at(df):

    vehicle_arrived_at = pd.to_datetime(df["vehicle_arrived_at"])

    pickup_arrival_time = df["pickup_arrival_time"].fillna("99:99:99")
    ftr = [3600, 60, 1]
    pickup_arrival_time = pickup_arrival_time.apply(
        lambda row: sum([a * b for a, b in zip(ftr, map(int, row.split(":")))]),
    )

    avg_pickup_arrival_time = sum(x for x in pickup_arrival_time if x != 362439) / len(
        list(x for x in pickup_arrival_time if x != 362439)
    )

    vehicle_arrived_at = np.where(
        vehicle_arrived_at.isna(),
        np.where(
            pickup_arrival_time == 362439,
            np.where(
                (
                    df["dispatched_at"] + pd.Timedelta(seconds=avg_pickup_arrival_time)
                    < df["pickup_at"]
                )
                | (df["pickup_at"].isna() == True),
                df["dispatched_at"] + pd.Timedelta(seconds=avg_pickup_arrival_time),
                df["pickup_at"],
            ),
            np.where(
                (
                    df["dispatched_at"] + pd.to_timedelta(pickup_arrival_time)
                    < df["pickup_at"]
                )
                | (df["pickup_at"].isna() == True),
                df["dispatched_at"] + pd.to_timedelta(pickup_arrival_time),
                df["pickup_at"],
            ),
        ),
        vehicle_arrived_at,
    )

    vehicle_arrived_at = pd.to_datetime(vehicle_arrived_at)

    return vehicle_arrived_at


def clean_arriving_push(df):
    arriving_push = pd.to_datetime(df["arriving_push"])
    arriving_push = arriving_push.fillna(
        df["vehicle_arrived_at"] - pd.Timedelta(minutes=3)
    )
    return arriving_push


def clean_pickup_arrival_time(df):
    pick_up_arrival_time = df["pickup_arrival_time"].fillna(
        (df["vehicle_arrived_at"] - df["dispatched_at"]).dt.seconds
    )

    return pick_up_arrival_time.astype(int)


def clean_earlierst_pickup_expectation(df):
    earlierst_pickup_expectation = pd.to_datetime(df["earliest_pickup_expectation"])
    earlierst_pickup_expectation = earlierst_pickup_expectation.fillna(
        df["dispatched_at"] - pd.Timedelta(minutes=3)
    )

    return earlierst_pickup_expectation


def clean_pick_up_at(df):
    pickup_at = pd.to_datetime(df["pickup_at"])
    boarding_time = df["boarding_time"].fillna("99:99:99")
    ftr = [3600, 60, 1]
    boarding_time = df["boarding_time"].apply(
        lambda row: sum([a * b for a, b in zip(ftr, map(int, row.split(":")))]),
    )

    avg_boarding_time = sum(x for x in boarding_time if x != 362439) / len(
        list(x for x in boarding_time if x != 362439)
    )

    pickup_at = np.where(
        pickup_at.isna() == True,
        df["vehicle_arrived_at"] + pd.Timedelta(avg_boarding_time),
        pickup_at,
    )

    return pickup_at


def clean_drop_off_at(df):
    pass


# def clean_drop_off_at(df):
#    dropoff_at = df["dropoff_at"].fillna(df["dropoff_eta"])
#    return dropoff_at


def clean_dropoff_eta(df):
    dropoff_eta = df["dropoff_eta"].fillna(df["dropoff_at"])
    return dropoff_eta


def clean_pick_up_first_eta(df):
    pass


def clean_pick_up_eta(df):
    pick_up_eta = df["pickup_eta"].fillna(df["pickup_first_eta"])
    return pick_up_eta


def clean_dropoff_first_eta(df):
    dropoff_first_eta = df["dropoff_first_eta"].fillna(
        df["pickup_first_eta"] + df["shortest_ride_time"]
    )
    return dropoff_first_eta
