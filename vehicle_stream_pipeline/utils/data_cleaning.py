import os
import time
import warnings
from datetime import datetime as dt
from re import M

import git
import numpy as np
import pandas as pd
from numpy import NaN

warnings.filterwarnings("ignore")


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


def clean_duplicates(df):
    """This function checks for duplicates in the DataFrame and only retains the ids with most filled attributes (scheduled_to).

    Args:
        df: rides_combined.csv.

    Returns:
        df: DataFrame w/o duplicates.
    """
    duplicate_ids = df[df.duplicated(subset=["id"]) & (df["id"].isna() == False)]["id"]
    duplicates = df[df["id"].isin(duplicate_ids)]
    duplicates = duplicates.sort_values(["id", "scheduled_to"])
    duplicates.reset_index(inplace=True)
    df.drop(df[df["id"].isin(duplicate_ids)].index, inplace=True)
    for index, row in duplicates.iterrows():
        if pd.notnull(row["scheduled_to"]):
            timestamp_columns = [
                "scheduled_to",
                "dispatched_at",
                "arriving_push",
                "vehicle_arrived_at",
                "earliest_pickup_expectation",
                "pickup_first_eta",
                "pickup_eta",
                "pickup_at",
                "dropoff_first_eta",
                "dropoff_eta",
                "dropoff_at",
            ]
            for col in timestamp_columns:
                if not pd.notnull(row[col]):
                    duplicates[col][index] = duplicates[col][index + 1]
        else:
            duplicates.drop(index, inplace=True)

    df = df.append(duplicates, ignore_index=True)
    return df


# Format-Check
def check_format(df, col_type_dict):
    """This function checks the format in the dataframe.

    Args:
        df, col_type_dict: Dataframe and param col_type_dic.

    Returns:
        df, df_inconsistencies: Dataframe with checked format and DataFrame with inconsitent format.
    """
    df_inconsistencies = pd.DataFrame(columns=list(df.columns))
    # check time format in order to avoid errors in cleaning
    for col, col_type in col_type_dict.items():
        if col_type == "timestamp":
            df_inconsistencies_temp = df[
                ~(
                    (
                        df[col].str.match(
                            r"[0-9]{1,4}.[0-9]{1,2}.[0-9]{1,2} [0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2}"
                        )
                        == True
                    )
                    | (df[col].isna())
                    | (
                        df[col].str.match(
                            r"[0-9]{1,4}-[0-9]{1,2}-[0-9]{1,2} [0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2}"
                        )
                        == True
                    )
                )
            ]

            df_inconsistencies = pd.concat(
                [df_inconsistencies, df_inconsistencies_temp], axis=0, ignore_index=True
            )
            df = df[
                (
                    df[col].str.match(
                        r"[0-9]{1,4}.[0-9]{1,2}.[0-9]{1,2} [0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2}"
                    )
                    == True
                )
                | (df[col].isna())
            ]
        elif col_type == "time":
            df_inconsistencies_temp = df[
                ~(
                    (df[col].str.match(r"[0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2}") == True)
                    | (df[col].str.contains("1899"))
                    | (df[col].str.contains("1900"))
                    | (df[col].isna())
                )
            ]
            df = df[
                (df[col].str.match(r"[0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2}") == True)
                | (
                    df[col].str.match(r"[0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2}.[0-9]*")
                    == True
                )
                | (df[col].str.contains("1899"))
                | (df[col].str.contains("1900"))
                | (df[col].isna())
            ]
            df_inconsistencies = pd.concat(
                [df_inconsistencies, df_inconsistencies_temp], axis=0, ignore_index=True
            )

        elif col_type == "numerical":
            df_inconsistencies_temp = df[
                ~(
                    df[col].astype(str).str.replace(".", "").str.isdigit()
                    | (df[col].isna())
                )
            ]
            df = df[
                df[col].astype(str).str.replace(".", "").str.isdigit() | df[col].isna()
            ]
            df[col] = df[col].astype(float)
            df_inconsistencies = pd.concat(
                [df_inconsistencies, df_inconsistencies_temp], axis=0, ignore_index=True
            )

    return (df, df_inconsistencies)


def clean_free_ride(df):
    """This function sets all values from the free_ride column to FALSE or TRUE.

    Args:
        df: Dataframe.

    Returns:
        free_ride: Dataframe with cleaned free-ride column.
    """
    free_ride = np.where(df["free_ride"] == 1, True, False)
    return free_ride


# Attribute: 'id'
def clean_ride_id(df):
    """This function fills the empty id's.

    Args:
        df: Dataframe.

    Returns:
        id: Dateframe with cleaned id's.
    """
    id = pd.DataFrame(data=df.loc[:, "id"], columns=["id"])
    id.id.fillna(df.created_from_offer.astype("Int64"), inplace=True)
    return id

def clean_distance(df):
    """This function cleans distance where pickup_address == dropoff_address.

    Args:
        df: Dataframe.

    Returns:
        id: Dateframe without same pickup & dropoff address.
    """
    df = df[df["pickup_address"] != df["dropoff_address"]]
    return df


# Attributes: ['pickup_address', 'dropoff_address']
def get_stop_id(address, df_stops):
    if address[0].isdigit():
        lat = address.split("|")[0]
        long = address.split("|")[1]
        for index, row in df_stops.iterrows():
            if str(row["MoDStop Lat"]) == lat and str(row["MoDStop Long"]) == long:
                return row["MoDStop Id"]
        return 0
    else:
        # fix different namings between MoDStop table and rides table
        if address == "Rewe Mußbach":
            address = address + " (Shoppenwiese)"
        elif address == "Lachener Straße":
            address = "Laachener Straße"
        for index, row in df_stops.iterrows():
            if row["MoDStop Name"] == address:
                return row["MoDStop Id"]
            elif address == "Würzmühle":
                return 11009
        return 0


def clean_addresses(df, df_stops):
    """This function checks if the addresses match those of the MoD Stop table and exports a list with the addresses that do not match.  

    Args:
        df, df_stops: Cleaning dataframe and MoD Stops dataframe.

    Returns:
        addresse: Dateframe with cleaned addresses.
    """
    addresses = pd.DataFrame(
        data=df.loc[:, ["pickup_address", "dropoff_address"]],
        columns=["pickup_address", "dropoff_address"],
    )
    addresses[["pickup_id", "dropoff_id"]] = ""
    for index, row in addresses.iterrows():
        addresses.at[index, "pickup_id"] = get_stop_id(row["pickup_address"], df_stops)
        addresses.at[index, "dropoff_id"] = get_stop_id(
            row["dropoff_address"], df_stops
        )
    # export list of unmatched addresses
    repo = git.Repo(".", search_parent_directories=True).git.rev_parse(
        "--show-toplevel"
    )
    file = f"{repo}/data/cleaning/unmatched_addresses_{int(time.time())}.xlsx"
    mask = (addresses["pickup_id"] == 0) | (addresses["dropoff_id"] == 0)
    df[mask].to_excel(file)
    addresses.drop(columns=["pickup_address", "dropoff_address"], axis=1, inplace=True)
    return addresses


# Attribute: 'created_at'
def clean_created_at(df):
    """This function sets the format of created_at to datetime.

    Args:
        df: Cleaning dataframe.

    Returns:
        created_at: Dateframe with cleaned created_at column.
    """
    created_at = pd.to_datetime(df["created_at"])
    return created_at


# Attribute: 'scheduled_to'
def clean_scheduled_to(df):
    # clean scheduled_to
    df["scheduled_to"] = pd.to_datetime(df["scheduled_to"])
    df["scheduled_to"] = df["scheduled_to"].fillna(df["created_at"])

    # filter that scheduled_to is not before created_at
    df["scheduled_to"] = np.where(
        df["scheduled_to"] < df["created_at"], df["created_at"], df["scheduled_to"]
    )
    scheduled_to = pd.to_datetime(df["scheduled_to"])
    return scheduled_to


# Attribute: 'dispatched_at'
def clean_dispatched_at(df):
    # Cast to correct dtype
    df["dispatched_at"] = pd.to_datetime(df["dispatched_at"])
    df["scheduled_to"] = pd.to_datetime(df["scheduled_to"])
    # Fill values of dispatched_at which are completed and scheduled rides with scheduled-8 Min else with created_at
    df["dispatched_at"] = np.where(
        (df["state"] == "completed") & (df["scheduled_to"] != df["created_at"]),
        # Clear cases where scheduled_to - 8Min is smaller than created_at else dispatched_at would be smaller than created_at
        np.where(
            (df["scheduled_to"] - pd.Timedelta(minutes=8) < df["created_at"]),
            df["created_at"],
            df["scheduled_to"] - pd.Timedelta(minutes=8),
        ),
        np.where(
            (df["state"] == "completed") & (df["scheduled_to"] == df["created_at"]),
            df["created_at"],
            df["dispatched_at"],
        ),
    )
    dispatched_at = pd.to_datetime(df["dispatched_at"])

    return dispatched_at


def getAvgPickupArrivalTime(df):
    # get the average pickup arrival time
    times = [3600, 60, 1]
    df["pickup_arrival_time"] = pd.to_datetime(df.pickup_arrival_time)
    # get all values in one format
    df["pickup_arrival_time"] = df["pickup_arrival_time"].dt.strftime("%H:%M:%S")
    # replace all values with -9 if pickup_arrival_time is NaN or if it is bigger than 3 hours (assumption)
    df["pickup_arrival_time"] = np.where(
        (pd.to_timedelta(df["pickup_arrival_time"]) > pd.Timedelta(hours=3))
        | (df["pickup_arrival_time"].isna()),
        "-9",
        df["pickup_arrival_time"],
    )
    df["pickup_arrival_time"] = (
        df["pickup_arrival_time"]
        .str[0:8]
        .apply(
            lambda row: sum(
                [
                    a * b
                    for a, b in zip(times, map(int, row.split(":")))
                    if len(row) == 8
                ]
            )
        )
    )
    avg_pickup_arrival_time = sum(
        x for x in df["pickup_arrival_time"] if x != -9
    ) / len(list(x for x in df["pickup_arrival_time"] if x != -9))
    avg_pickup_arrival_time = round(avg_pickup_arrival_time)
    return avg_pickup_arrival_time


# Attribute: 'vehicle_arrived_at'
def clean_vehicle_arrived_at(df):
    df["arriving_push"] = pd.to_datetime(df["arriving_push"])
    vehicle_arrived_at = pd.to_datetime(df["vehicle_arrived_at"])
    df["pickup_at"] = pd.to_datetime(df["pickup_at"])
    avg_pickup_arrival_time = getAvgPickupArrivalTime(df)
    # fill the NaN values with dispatched_at plus the average pickup arrival time since pickup_arrival_time = vehicle_arrivd_at - dispatched_at
    vehicle_arrived_at = np.where(
        (vehicle_arrived_at.isna()) & (df["state"] == "completed"),
        # only if dispatched_at + average pickup time is smaller than pickup_at we add the average time to dispatched_at else we take the pickup_at
        np.where(
            (
                df["dispatched_at"] + pd.Timedelta(seconds=avg_pickup_arrival_time)
                < df["pickup_at"]
            )
            | (df["pickup_at"].isna() == True),
            df["dispatched_at"] + pd.Timedelta(seconds=avg_pickup_arrival_time),
            df["pickup_at"],
        ),
        vehicle_arrived_at,
    )
    vehicle_arrived_at = pd.to_datetime(vehicle_arrived_at)

    # vehicle_arrived_at must take place on the same date as scheduled_to
    vehicle_arrived_at = np.where(
        vehicle_arrived_at - df["scheduled_to"] > pd.Timedelta(days=1),
        df["dispatched_at"] + pd.Timedelta(seconds=avg_pickup_arrival_time),
        # assumption that vehicle arrives in at least 1 hour from the actual schedule time
        np.where(
            (vehicle_arrived_at < df["arriving_push"])
            | (vehicle_arrived_at + pd.Timedelta(minutes=60) < df["scheduled_to"])
            | (vehicle_arrived_at - pd.Timedelta(minutes=60) > df["scheduled_to"])
            | (vehicle_arrived_at < df["dispatched_at"]),
            np.where(
                (df["arriving_push"].isna())
                | (df["arriving_push"] < df["dispatched_at"]),
                np.where(
                    (
                        df["dispatched_at"]
                        + pd.Timedelta(seconds=avg_pickup_arrival_time)
                        < df["pickup_at"]
                    )
                    | (df["pickup_at"].isna() == True)
                    | (df["pickup_at"] < df["dispatched_at"]),
                    df["dispatched_at"] + pd.Timedelta(seconds=avg_pickup_arrival_time),
                    df["pickup_at"],
                ),
                # arriving push is the assumption from the system that the pickup will be arrived in less than 3 minutes
                np.where(
                    ((df["arriving_push"] + pd.Timedelta(minutes=3)) < df["pickup_at"]),
                    df["arriving_push"] + pd.Timedelta(minutes=3),
                    df["arriving_push"],
                ),
            ),
            vehicle_arrived_at,
        ),
    )
    vehicle_arrived_at = pd.to_datetime(vehicle_arrived_at)
    vehicle_arrived_at = vehicle_arrived_at.floor("s")

    return vehicle_arrived_at


# Attribute: 'arriving_push'
def clean_arriving_push(df):
    arriving_push = pd.to_datetime(df["arriving_push"])

    arriving_push = df["arriving_push"].fillna(
        df["vehicle_arrived_at"] - pd.Timedelta(minutes=3)
    )

    # Check ordering
    arriving_push = np.where(
        # check if it is not too far away from scheduled_to or check if arriving_push is not more than 15 minutes before scheduled_to
        (arriving_push - df["scheduled_to"] > pd.Timedelta(minutes=120))
        | ((df["scheduled_to"] - arriving_push) > pd.Timedelta(minutes=30)),
        df["vehicle_arrived_at"] - pd.Timedelta(minutes=3),
        # arrriving_push is before created_at than use scheduled_to
        np.where((arriving_push < df["created_at"]), df["scheduled_to"], arriving_push),
    )
    arriving_push = pd.to_datetime(arriving_push)

    return arriving_push


# Attribute: 'earliest_pickup_expectation'
def clean_earlierst_pickup_expectation(df):
    earlierst_pickup_expectation = pd.to_datetime(df["earliest_pickup_expectation"])
    # earliest pickup expectation is defined as dispatched + 3 Minuten
    earlierst_pickup_expectation = np.where(
        # case that it is not a scheduled ride or that scheduled - 8Min < created_at
        (df["scheduled_to"] == df["created_at"])
        | (df["scheduled_to"] - pd.Timedelta(minutes=8) < df["created_at"]),
        df["dispatched_at"] + pd.Timedelta(minutes=3),
        # case that it is a scheduled ride
        df["scheduled_to"] - pd.Timedelta(minutes=5),
    )
    # Check ordering
    earlierst_pickup_expectation = np.where(
        earlierst_pickup_expectation - df["scheduled_to"] > pd.Timedelta(days=1),
        df["vehicle_arrived_at"] - pd.Timedelta(minutes=3),
        earlierst_pickup_expectation,
    )
    earlierst_pickup_expectation = pd.to_datetime(earlierst_pickup_expectation)

    return earlierst_pickup_expectation


# Attribute: 'pickup_at'
def clean_pickup_at(df):
    pickup_at = pd.to_datetime(df["pickup_at"])
    pickup_eta = pd.to_datetime(df["pickup_eta"])
    # calculate the average boarding time because boarding_time = pickup_at - vehicle_arrived_at
    boarding_time = pd.Series(
        np.where(
            df["vehicle_arrived_at"] < pickup_at,
            (pickup_at - df["vehicle_arrived_at"]).dt.seconds,
            -9,
        )
    )
    boarding_time = boarding_time.fillna(-9)

    avg_boarding_time = sum(x for x in boarding_time if x != -9) / len(
        list(x for x in boarding_time if x != -9)
    )
    avg_boarding_time = round(avg_boarding_time)

    # fill NaN values
    pickup_at = np.where(
        (pickup_at.isna()) & (df["state"] == "completed"),
        # if pickup_eta is Nan or pickup_eta is too far away from scheduled_to than fill the values with vehicle_arrived_at + avg boarding time else put pickup_eta as value
        np.where(
            (df["pickup_eta"].isna())
            | (pickup_eta - df["scheduled_to"] >= pd.Timedelta(days=1)),
            df["vehicle_arrived_at"] + pd.Timedelta(seconds=avg_boarding_time),
            df["pickup_eta"],
        ),
        pickup_at,
    )
    pickup_at = pd.to_datetime(pickup_at)

    # Check ordering
    pickup_at = np.where(
        # pickup_at must be after or at the same time than vehicle_arrived_at
        # pickup_at can not be far away from scheduled_to
        (pickup_at < df["vehicle_arrived_at"])
        | (pickup_at - df["scheduled_to"] > pd.Timedelta(days=1)),
        np.where(
            (df["pickup_eta"].isna())
            | (df["pickup_eta"] < df["vehicle_arrived_at"])
            | (pickup_at - df["scheduled_to"] >= pd.Timedelta(days=1)),
            np.where(
                (
                    df["vehicle_arrived_at"] + pd.Timedelta(seconds=avg_boarding_time)
                    < df["dropoff_at"]
                )
                | (df["dropoff_at"].isna()),
                df["vehicle_arrived_at"] + pd.Timedelta(seconds=avg_boarding_time),
                df["vehicle_arrived_at"],
            ),
            df["pickup_eta"],
        ),
        pickup_at,
    )

    pickup_at = pd.to_datetime(pickup_at)

    return pickup_at


# Attribute: 'pickup_eta'
def clean_pickup_eta(df):
    # Attribute: 'pickup_eta'
    pickup_eta = pd.to_datetime(df["pickup_eta"])
    pickup_eta = pickup_eta.fillna(df["pickup_at"])

    # Check ordering
    pickup_eta = np.where(
        (pickup_eta < df["dispatched_at"])
        | (pickup_eta - df["scheduled_to"] > pd.Timedelta(days=1)),
        df["pickup_at"],
        pickup_eta,
    )
    pickup_eta = pd.to_datetime(pickup_eta)

    return pickup_eta


# Attribute: 'pickup_first_eta'
def clean_pickup_first_eta(df):
    pickup_first_eta = pd.to_datetime(df["pickup_first_eta"])
    pickup_first_eta = pickup_first_eta.fillna(df["pickup_eta"])

    # Check ordering
    pickup_first_eta = np.where(
        # if pickup_first_eta not at same day than scheduled_to and case that ride takes place at midnight
        (pickup_first_eta < df["dispatched_at"])
        | (
            (pickup_first_eta.dt.day != df["scheduled_to"].dt.day)
            & (pickup_first_eta - df["scheduled_to"] > pd.Timedelta(minutes=80))
        ),
        df["pickup_eta"],
        pickup_first_eta,
    )
    pickup_first_eta = pd.to_datetime(pickup_first_eta)

    return pickup_first_eta


# Attribute: 'dropoff_at'
def clean_dropoff_at(df):
    df["dropoff_at"] = pd.to_datetime(df["dropoff_at"])
    dropoff_eta = pd.to_datetime(df["dropoff_eta"])
    ftr = [3600, 60, 1]
    shortest_ridetime = (
        df["shortest_ridetime"]
        .str[0:8]
        .apply(lambda row: sum([a * b for a, b in zip(ftr, map(int, row.split(":")))]))
    )

    df["dropoff_at"] = np.where(
        (df["dropoff_at"].isna()) & (df["state"] == "completed"),
        np.where(
            (df["dropoff_eta"].isna())
            | (dropoff_eta - df["scheduled_to"] >= pd.Timedelta(days=1)),
            df["dropoff_at"] + pd.to_timedelta(shortest_ridetime, unit="s"),
            df["dropoff_eta"],
        ),
        df["dropoff_at"],
    )
    df["dropoff_at"] = pd.to_datetime(df["dropoff_at"])

    # Check ordering
    df["dropoff_at"] = np.where(
        (df["dropoff_at"] <= df["pickup_at"])
        | (df["dropoff_at"] - df["scheduled_to"] > pd.Timedelta(days=1)),
        df["pickup_at"] + pd.to_timedelta(shortest_ridetime, unit="s"),
        df["dropoff_at"],
    )

    dropoff_at = pd.to_datetime(df["dropoff_at"])

    return dropoff_at


# Attribute: 'dropoff_eta'
def clean_dropoff_eta(df):
    dropoff_eta = pd.to_datetime(df["dropoff_eta"])
    dropoff_eta = dropoff_eta.fillna(df["dropoff_at"])

    # Check ordering
    dropoff_eta = np.where(
        (dropoff_eta < df["dispatched_at"])
        | (dropoff_eta - df["scheduled_to"] > pd.Timedelta(days=1)),
        df["dropoff_at"],
        dropoff_eta,
    )
    dropoff_eta = pd.to_datetime(dropoff_eta)

    return dropoff_eta


# Attribute: 'dropoff_first_eta'
def clean_dropoff_first_eta(df):
    dropoff_first_eta = pd.to_datetime(df["dropoff_first_eta"])
    ftr = [3600, 60, 1]
    shortest_ridetime = (
        df["shortest_ridetime"]
        .str[0:8]
        .apply(lambda row: sum([a * b for a, b in zip(ftr, map(int, row.split(":")))]))
    )
    dropoff_first_eta = dropoff_first_eta.fillna(
        df["pickup_first_eta"] + pd.to_timedelta(shortest_ridetime, unit="s")
    )

    # Check ordering
    dropoff_first_eta = np.where(
        # if dropoff_first_eta not at same day than scheduled_to and case that ride takes place at midnight
        (dropoff_first_eta < df["dispatched_at"])
        | (
            (dropoff_first_eta.dt.day != df["scheduled_to"].dt.day)
            & (dropoff_first_eta - df["scheduled_to"] > pd.Timedelta(minutes=80))
        ),
        df["pickup_first_eta"] + pd.to_timedelta(shortest_ridetime, unit="s"),
        dropoff_first_eta,
    )
    dropoff_first_eta = pd.to_datetime(dropoff_first_eta)

    return dropoff_first_eta


# Attributes: ['pickup_arrival_time', 'arrival_deviation', 'waiting_time', 'boarding_time', 'ride_time', 'trip_time', 'shortest_ridetime', 'delay', 'longer_route_factor']
def clean_time_periods(df):
    # Attribute: 'pickup_arrival_time'
    df["pickup_arrival_time"] = (
        df["vehicle_arrived_at"] - df["dispatched_at"]
    ).dt.seconds

    # Attribute: 'arrival_deviation'
    df["arrival_deviation"] = df.apply(
        lambda row: (
            (row["vehicle_arrived_at"] - row["arriving_push"]).round(freq="s")
        ).total_seconds()
        - 180
        if (row["vehicle_arrived_at"] == row["vehicle_arrived_at"])
        and (row["arriving_push"] == row["arriving_push"])
        else np.NaN,
        axis=1,
    )
    df["arrival_deviation"] = np.where(
        df["arrival_deviation"] < 0, 0, df["arrival_deviation"]
    )

    # Attribute: 'waiting_time'
    df["waiting_time"] = df.apply(
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
    df["boarding_time"] = df.apply(
        lambda row: (
            (row["pickup_at"] - row["vehicle_arrived_at"]).round(freq="s")
        ).total_seconds()
        if (row["vehicle_arrived_at"] == row["vehicle_arrived_at"])
        and (row["pickup_at"] == row["pickup_at"])
        else np.NaN,
        axis=1,
    )

    # Attribute: 'ride_time'
    df["ride_time"] = df.apply(
        lambda row: (
            (row["dropoff_at"] - row["pickup_at"]).round(freq="s")
        ).total_seconds()
        if (row["dropoff_at"] == row["dropoff_at"])
        and (row["pickup_at"] == row["pickup_at"])
        else np.NaN,
        axis=1,
    )

    # Attribute: 'trip_time'
    df["trip_time"] = df.apply(
        lambda row: (row["ride_time"] + row["waiting_time"]),
        axis=1,
    )

    # Attribute: 'shortest_ridetime'
    df["shortest_ridetime"] = df.apply(
        lambda row: (
            pd.to_timedelta(row["shortest_ridetime"]).round(freq="s").total_seconds()
        )
        if (row["shortest_ridetime"] == row["shortest_ridetime"])
        else np.NaN,
        axis=1,
    )

    # Attribute: 'delay'
    df["delay"] = df.apply(
        lambda row: (row["trip_time"] - row["shortest_ridetime"]),
        axis=1,
    )

    # Attribute: 'longer_route_factor'
    df["longer_route_factor"] = df.apply(
        lambda row: round(row["ride_time"] / row["shortest_ridetime"], 2)
        if (row["shortest_ridetime"] != 0)
        else np.NaN,
        axis=1,
    )

    return df


# Attribute: 'rating'
def clean_rating(df):
    rating = df["rating"]
    rating = np.where(
        (
            df["rating"].str.match(
                r"[0-9]{1,4}.[0-9]{1,2}.[0-9]{1,2} [0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2}"
            )
            == True
        ),
        rating.str[9].astype(float),
        df["rating"],
    )
    return rating


def data_cleaning(df, df_stops):

    columns = {
        "distance": "numerical",
        "number_of_passenger": "numerical",
        "created_at": "timestamp",
        "scheduled_to": "timestamp",
        "dispatched_at": "timestamp",
        "pickup_arrival_time": "time",
        "arriving_push": "timestamp",
        "vehicle_arrived_at": "timestamp",
        "earliest_pickup_expectation": "timestamp",
        "pickup_first_eta": "timestamp",
        "pickup_eta": "timestamp",
        "pickup_at": "timestamp",
        "dropoff_first_eta": "timestamp",
        "dropoff_eta": "timestamp",
        "dropoff_at": "timestamp",
        "waiting_time": "time",
        "boarding_time": "time",
        "ride_time": "time",
        "trip_time": "time",
        "shortest_ridetime": "time",
        "delay": "time",
    }

    repo = git.Repo(".", search_parent_directories=True).git.rev_parse(
        "--show-toplevel"
    )

    df, df_inconsistencies = check_format(df, columns)
    if df_inconsistencies.empty == False:
        df_inconsistencies.to_excel(
            f"{repo}/data/cleaning/inconsistencies_{int(time.time())}.xlsx"
        )

    print("clean id")
    df["id"] = clean_ride_id(df)

    print("clean distance")
    df = clean_distance(df)  # Hier mit gesamten df, da Zeilen gelöscht werden

    print("clean addresses")
    df[["pickup_address", "dropoff_address"]] = clean_addresses(df, df_stops)

    print("clean free_rides")
    df["free_ride"] = clean_free_ride(df)

    print("clean created_at")
    df["created_at"] = clean_created_at(df)

    print("clean scheduled_to")
    df["scheduled_to"] = clean_scheduled_to(df)

    print("clean dispatched_at")
    df["dispatched_at"] = clean_dispatched_at(df)

    print("clean vehicle_arrived_at")
    df["vehicle_arrived_at"] = clean_vehicle_arrived_at(df)

    print("clean arriving_push")
    df["arriving_push"] = clean_arriving_push(df)

    print("clean earliest_pickup_expectation")
    df["earliest_pickup_expectation"] = clean_earlierst_pickup_expectation(df)

    print("clean pickup_at")
    df["pickup_at"] = clean_pickup_at(df)

    print("clean pickup_eta")
    df["pickup_eta"] = clean_pickup_eta(df)

    print("clean pickup_first_eta")
    df["pickup_first_eta"] = clean_pickup_first_eta(df)

    print("clean dropoff_at")
    df["dropoff_at"] = clean_dropoff_at(df)

    print("clean dropoff_eta")
    df["dropoff_eta"] = clean_dropoff_eta(df)

    print("clean dropoff_first_eta")
    df["dropoff_first_eta"] = clean_dropoff_first_eta(df)

    print("clean time periods")
    df = clean_time_periods(df)

    print("clean rating")
    df["rating"] = clean_rating(df)
    return df


#def add_shared_rides(df, vehicle_usage_df, external_df):
    print("add shared rides")

    # vehicle_usage_df preprocessing - filteirng on Stop Point type and status + drop remaining duplicates
    merge_vehicle_df = vehicle_usage_df[
        (vehicle_usage_df["Stop Point Type"] == "dropoff")
        & (vehicle_usage_df["Stop Point status"] == "completed")
    ]
    merge_vehicle_df.dropna(subset=["Ride Id"], inplace=True)
    merge_vehicle_df.sort_values(by="Vehicle Id", inplace=True)
    merge_vehicle_df.drop_duplicates(subset=["Ride Id"], inplace=True)
    merge_vehicle_df.rename(columns={"Ride Id": "Ride_id_vehicle_usage"}, inplace=True)
    merge_vehicle_df = merge_vehicle_df[["Ride_id_vehicle_usage", "Vehicle Id"]]

    # external df preprocssing - only id which are in vehicle usage df
    merge_external_df = external_df[
        external_df["Id"].isin(merge_vehicle_df["Ride_id_vehicle_usage"])
    ]
    merge_external_df.rename(columns={"Id": "Ride_id_external"}, inplace=True)
    merge_external_df = merge_external_df[["Ride_id_external", "External Id"]]

    # Left Join filtered vehicle df and external df + preprocessing for join with rides_df
    vehicle_external_merge = merge_vehicle_df.merge(
        merge_external_df,
        how="left",
        left_on="Ride_id_vehicle_usage",
        right_on="Ride_id_external",
    )
    vehicle_external_merge.dropna(subset=["External Id"], inplace=True)
    vehicle_external_merge.drop_duplicates(subset=["External Id"], inplace=True)
    vehicle_external_merge = vehicle_external_merge[
        vehicle_external_merge["External Id"].isin(df["id"])
    ]

    # Left Join removed duplicates rides_df & filterd vehicle_external_merge
    rides_vehicle_merge_df = df.merge(
        vehicle_external_merge, how="left", left_on="id", right_on="External Id"
    )
    rides_vehicle_merge_df.drop(
        columns=["External Id", "Ride_id_external", "Ride_id_vehicle_usage"],
        inplace=True,
    )

    # find shared rides and add columns "shared_rides"
    df = rides_vehicle_merge_df.copy()
    # Create empty combined ride columns - need to be adjusted if more than 3 rides combined
    df["shared_rides_1"] = NaN
    df["shared_rides_2"] = NaN
    df["shared_rides_3"] = NaN

    for index, row in df.iterrows():

        # skip offers and rides w/o vehicle Id
        if row["Vehicle Id"] == NaN:
            continue

        ride_id = row["id"]
        vehicle_id = row["Vehicle Id"]
        pickup = row["pickup_at"]
        dropoff = row["dropoff_at"]

        # Expressions Match vehicle Id and different time scenarios
        exp_vehicle = df["Vehicle Id"] == vehicle_id
        # smaller time means earlier
        exp_1 = (df["pickup_at"] > pickup) & (df["dropoff_at"] < dropoff)
        exp_2 = (
            (df["pickup_at"] < pickup)
            & (df["dropoff_at"] < dropoff)
            & (df["dropoff_at"] > pickup)
        )
        exp_3 = (df["pickup_at"] < pickup) & (df["dropoff_at"] > dropoff)
        exp_4 = (
            (df["pickup_at"] > pickup)
            & (df["dropoff_at"] > dropoff)
            & (df["pickup_at"] < dropoff)
        )

        filt_df = exp_vehicle & (exp_1 | exp_2 | exp_3 | exp_4)
        true_count_filt_df = filt_df[filt_df == True].count()

        if true_count_filt_df == 1:
            vehicle_id_list = df["id"][filt_df].to_list()
            df.loc[df.id == ride_id, ["shared_rides_1"]] = vehicle_id_list
        elif true_count_filt_df == 2:
            vehicle_id_list = df["id"][filt_df].to_list()
            df.loc[
                df.id == ride_id, ["shared_rides_1", "shared_rides_2"]
            ] = vehicle_id_list
        elif true_count_filt_df == 3:
            vehicle_id_list = df["id"][filt_df].to_list()
            df.loc[
                df.id == ride_id, ["shared_rides_1", "shared_rides_2", "shared_rides_3"]
            ] = vehicle_id_list

    return df


def data_check(df):
    """This function checks the resulting DataFrames for any mistakes regarding the ordering, calculations and outliers.
    Args:
        df: Path to the folder which stores the excel sheets.

    Returns:
        df, df_incorrect: The checked DataFrame excluding the bugs & a DataFrame with the incorrect lines.
    """
    # check the most important orderings and calculations - move incorrect entities into df_incorrect
    df = df[df["state"] == "completed"]

    df[
        [
            "created_at",
            "scheduled_to",
            "dispatched_at",
            "arriving_push",
            "vehicle_arrived_at",
            "earliest_pickup_expectation",
            "pickup_first_eta",
            "pickup_eta",
            "pickup_at",
            "dropoff_first_eta",
            "dropoff_eta",
            "dropoff_at",
            "updated_at",
        ]
    ] = df[
        [
            "created_at",
            "scheduled_to",
            "dispatched_at",
            "arriving_push",
            "vehicle_arrived_at",
            "earliest_pickup_expectation",
            "pickup_first_eta",
            "pickup_eta",
            "pickup_at",
            "dropoff_first_eta",
            "dropoff_eta",
            "dropoff_at",
            "updated_at",
        ]
    ].apply(
        pd.to_datetime
    )

    # filter wrong ordering: created_at
    df_incorrect = df.loc[
        (df.created_at > df.scheduled_to)
        | (df.created_at > df.dispatched_at)
        | (df.created_at > df.arriving_push)
        | (df.created_at > df.vehicle_arrived_at)
        | (df.created_at > df.earliest_pickup_expectation)
        | (df.created_at > df.pickup_first_eta)
        | (df.created_at > df.pickup_eta)
        | (df.created_at > df.pickup_at)
        | (df.created_at > df.dropoff_first_eta)
        | (df.created_at > df.dropoff_eta)
        | (df.created_at > df.dropoff_at)
        | (df.created_at > df.updated_at)
    ]

    # filter wrong ordering: scheduled_to
    df_incorrect = df.loc[(df.scheduled_to < df.dispatched_at)]

    # filter all timestamps that are not on the same date than scheduled_to
    # automatically validated if all other timestamps are on the same day
    df_incorrect = df.loc[
        (df.dispatched_at.dt.day != df.scheduled_to.dt.day)
        | (df.arriving_push.dt.day != df.scheduled_to.dt.day)
        | (df.vehicle_arrived_at.dt.day != df.scheduled_to.dt.day)
        | (df.earliest_pickup_expectation.dt.day != df.scheduled_to.dt.day)
        | (df.pickup_first_eta.dt.day != df.scheduled_to.dt.day)
        | (df.pickup_eta.dt.day != df.scheduled_to.dt.day)
        | (df.pickup_at.dt.day != df.scheduled_to.dt.day)
        | (df.dropoff_first_eta.dt.day != df.scheduled_to.dt.day)
        | (df.dropoff_eta.dt.day != df.scheduled_to.dt.day)
        | (df.dropoff_at.dt.day != df.scheduled_to.dt.day)
    ]

    # filter cases where the timestamps are not on the same day because they were at midnight
    # use dropoff_at for comparison (should include all other timestamps)
    df_incorrect = df_incorrect.loc[
        (df_incorrect.dropoff_at - df_incorrect.scheduled_to > pd.Timedelta(minutes=60))
    ]

    # filter wrong ordering: dispatched_at
    df_incorrect = df.loc[
        (df.dispatched_at > df.vehicle_arrived_at)
        | (df.dispatched_at > df.earliest_pickup_expectation)
        | (df.dispatched_at > df.pickup_first_eta)
        | (df.dispatched_at > df.pickup_eta)
        | (df.dispatched_at > df.pickup_at)
        | (df.dispatched_at > df.dropoff_first_eta)
        | (df.dispatched_at > df.dropoff_eta)
        | (df.dispatched_at > df.dropoff_at)
    ]

    # filter wrong ordering: arriving_push
    df_incorrect = df.loc[
        (df.arriving_push > df.vehicle_arrived_at)
        | (df.arriving_push > df.pickup_at)
        | (df.arriving_push > df.dropoff_at)
    ]

    # filter wrong ordering: vehicle_arrived_at
    df_incorrect = df.loc[
        (df.vehicle_arrived_at > df.pickup_at) | (df.vehicle_arrived_at > df.dropoff_at)
    ]

    # filter wrong ordering: pickup_at
    df_incorrect = df.loc[(df.pickup_at > df.dropoff_at)]

    # test the calculations
    # pickup_arrival_time
    df_incorrect = df.loc[
        (
            (df.vehicle_arrived_at - df.dispatched_at).dt.seconds
            != df.pickup_arrival_time
        )
        # arrival_deviation
        | (
            ((df.vehicle_arrived_at - df.arriving_push).dt.seconds - 180)
            != df.arrival_deviation
        )
        # waiting_time
        | (
            (
                (df.vehicle_arrived_at - df.earliest_pickup_expectation).dt.seconds
                != df.waiting_time
            )
            & (df.vehicle_arrived_at > df.earliest_pickup_expectation)
        )
        # filter cases where the values where negative
        | (
            (
                (df.vehicle_arrived_at - df.earliest_pickup_expectation).dt.seconds
                - 86400
                != df.waiting_time
            )
            & (df.vehicle_arrived_at < df.earliest_pickup_expectation)
        )
        # boarding_time
        | ((df.pickup_at - df.vehicle_arrived_at).dt.seconds != df.boarding_time)
        # ride_time
        | ((df.dropoff_at - df.pickup_at).dt.seconds != df.ride_time)
        # trip_time
        | ((df.ride_time + df.waiting_time) != df.trip_time)
        # delay
        | ((df.trip_time - df.shortest_ridetime) != df.delay)
    ]

    # filter the biggest outliers
    # pickup_arrival_time
    df_incorrect = df.loc[
        (df.pickup_arrival_time >= 10000)
        # arrival_deviation
        | (df.arrival_deviation >= 2000)
        # waiting_time
        | (df.waiting_time >= 3000)
        # boarding_time
        | (df.boarding_time >= 2000)
        # ride_time
        | (df.ride_time >= 5000)
        # trip_time
        | (df.trip_time >= 5000)
        # delay
        | (df.delay >= 5000)
    ]

    # remove incorrect entities and outliers from the cleaned_df
    df = (
        pd.merge(df, df_incorrect, indicator=True, how="outer")
        .query('_merge=="left_only"')
        .drop("_merge", axis=1)
    )

    return (df, df_incorrect)
