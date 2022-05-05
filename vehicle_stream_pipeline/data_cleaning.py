import time
from datetime import datetime as dt
from re import M

import git
import numpy as np
import pandas as pd


# remove duplicates
def clean_duplicates(df):
    duplicate_ids = df[df.duplicated(subset=["id"]) & (df["id"].isna() == False)]["id"]
    duplicates = df[df["id"].isin(duplicate_ids)]
    duplicates = duplicates.sort_values(["id", "scheduled_to"])
    duplicates.reset_index(inplace=True)
    df.drop(df[df["id"].isin(duplicate_ids)].index, inplace=True)
    for index, row in duplicates.iterrows():
        if pd.notnull(row["scheduled_to"]):
            print(row["id"])
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
                    print(col)
                    print(duplicates[col][index + 1])
                    duplicates[col][index] = duplicates[col][index + 1]
        else:
            duplicates.drop(index, inplace=True)

    df = df.append(duplicates, ignore_index=True)
    return df


# Format-Check
def check_format(df, col_type_dict):
    # check time format in order to avoid errors in cleaning
    for col, col_type in col_type_dict.items():
        if col_type == "timestamp":
            df_inconsistencies = df[
                ~(
                    (
                        df[col].str.match(
                            r"[0-9]{1,4}.[0-9]{1,2}.[0-9]{1,2} [0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2}"
                        )
                        == True
                    )
                    | (df[col].isna())
                )
            ]
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
            df_inconsistencies = df[
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
    return (df, df_inconsistencies)


# Attribute: 'id'
def clean_ride_id(df):
    id = pd.DataFrame(data=df.loc[:, "id"], columns=["id"])
    id.id.fillna(df.created_from_offer.astype("Int64"), inplace=True)
    return id


# Attribute: 'distance'
def clean_distance(df):
    # remove observations where pickup_address == dropoff_address
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
        return "No match of lat and long"
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
        return "No match of address name"


def clean_addresses(df, df_stops):
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
    mask = (
        (addresses["pickup_id"] == "No match of address name")
        | (addresses["pickup_id"] == "No match of lat and long")
        | (addresses["dropoff_id"] == "No match of lat and long")
        | (addresses["dropoff_id"] == "No match of address name")
    )
    df[mask].to_excel(file)
    addresses.drop(columns=["pickup_address", "dropoff_address"], axis=1, inplace=True)
    return addresses


# Attribute: 'created_at'
def clean_created_at(df):
    created_at = pd.to_datetime(df["created_at"])
    return created_at


# Attribute: 'scheduled_to'
def clean_scheduled_to(df):
    scheduled_to = pd.to_datetime(df["scheduled_to"])
    scheduled_to = scheduled_to.fillna(df["created_at"])

    ##### Hier gibt es 3 rides mit einem scheduled_to Datum, das vor created_at liegt, dadurch wurde automatisch vom System dispatched_at
    # mit 8Min. vor dem scheduled_at gefüllt und muss korrigiert werden und es wurde earliest_pickup_expectation
    # 5 minuten vor scheduled_to gefüllt, das muss auch korrigiert werden (letzters wird vermutlich dann bei clean_earliest.. gefixed)
    # Ansonsten scheint die order korrekt zu sein --> Wird alles später gelöst
    scheduled_to = np.where(
        scheduled_to < df["created_at"], df["created_at"], scheduled_to
    )
    return scheduled_to


# Attribute: 'dispatched_at'
def clean_dispatched_at(df):
    # Cast to correct dtype
    dispatched_at = pd.to_datetime(df["dispatched_at"])

    # Fill missing values of dispatched_at
    dispatched_at = np.where(
        (dispatched_at.isna()) & (df["state"] == "completed"),
        df["created_at"],
        dispatched_at,
    )

    # Check correct ordering
    dispatched_at = np.where(
        (
            (dispatched_at < df["created_at"])
            | (dispatched_at >= (df["scheduled_to"] + pd.Timedelta(minutes=9)))
        )
        & (df["scheduled_to"] != df["created_at"]),
        df["scheduled_to"] - pd.Timedelta(minutes=8),
        np.where(
            (
                (dispatched_at < df["created_at"])
                | (dispatched_at >= (df["scheduled_to"] + pd.Timedelta(minutes=9)))
            )
            & (df["scheduled_to"] == df["created_at"]),
            df["scheduled_to"],
            dispatched_at,
        ),
    )
    dispatched_at = pd.to_datetime(dispatched_at)

    return dispatched_at


# Attribute: 'vehicle_arrived_at'
def clean_vehicle_arrived_at(df):
    arriving_push = pd.to_datetime(df["arriving_push"])
    vehicle_arrived_at = pd.to_datetime(df["vehicle_arrived_at"])
    pickup_at = pd.to_datetime(df["pickup_at"])

    times = [3600, 60, 1]
    pickup_arrival_time = df["pickup_arrival_time"].fillna("-9")
    pickup_arrival_time = pd.Series(
        np.where(pickup_arrival_time.str.contains("1899"), "-9", pickup_arrival_time)
    )

    pickup_arrival_time = pickup_arrival_time.str[0:8].apply(
        lambda row: sum(
            [a * b for a, b in zip(times, map(int, row.split(":"))) if len(row) == 8]
        )
    )

    avg_pickup_arrival_time = sum(x for x in pickup_arrival_time if x != -9) / len(
        list(x for x in pickup_arrival_time if x != -9)
    )

    vehicle_arrived_at = np.where(
        (vehicle_arrived_at.isna()) & (df["state"] == "completed"),
        np.where(
            (
                df["dispatched_at"] + pd.Timedelta(seconds=avg_pickup_arrival_time)
                < pickup_at
            )
            | (pickup_at.isna() == True),
            df["dispatched_at"] + pd.Timedelta(seconds=avg_pickup_arrival_time),
            pickup_at,
        ),
        vehicle_arrived_at,
    )

    vehicle_arrived_at = pd.to_datetime(vehicle_arrived_at)

    # Check ordering
    vehicle_arrived_at = np.where(
        (vehicle_arrived_at < arriving_push)
        | (vehicle_arrived_at + pd.Timedelta(minutes=60) < df["scheduled_to"])
        | (vehicle_arrived_at - pd.Timedelta(minutes=60) > df["scheduled_to"]),
        np.where(
            arriving_push.isna(),
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
                ((arriving_push + pd.Timedelta(minutes=3)) < df["pickup_at"]),
                arriving_push + pd.Timedelta(minutes=3),
                arriving_push,
            ),
        ),
        vehicle_arrived_at,
    )

    vehicle_arrived_at = pd.to_datetime(vehicle_arrived_at)
    vehicle_arrived_at = vehicle_arrived_at.floor("s")

    return vehicle_arrived_at


# Attribute: 'arriving_push'
def clean_arriving_push(df):
    arriving_push = pd.to_datetime(df["arriving_push"])
    arriving_push = arriving_push.fillna(
        df["vehicle_arrived_at"] - pd.Timedelta(minutes=3)
    )
    return arriving_push


# Attribute: 'earliest_pickup_expectation'
def clean_earlierst_pickup_expectation(df):
    earlierst_pickup_expectation = pd.to_datetime(df["earliest_pickup_expectation"])
    earlierst_pickup_expectation = np.where(
        df["scheduled_to"] == df["created_at"],
        df["dispatched_at"] + pd.Timedelta(minutes=3),
        df["scheduled_to"] - pd.Timedelta(minutes=5),
    )
    return earlierst_pickup_expectation


# Attribute: 'pickup_at'
def clean_pickup_at(df):
    pickup_at = pd.to_datetime(df["pickup_at"])

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

    pickup_at = np.where(
        (pickup_at.isna() == True) & (df["state"] == "completed"),
        np.where(
            df["pickup_eta"].isna(),
            df["vehicle_arrived_at"] + pd.Timedelta(avg_boarding_time),
            df["pickup_eta"],
        ),
        pickup_at,
    )

    pickup_at = pd.to_datetime(pickup_at)

    ## Check ordering
    pickup_at = np.where(
        (pickup_at < df["vehicle_arrived_at"]),
        np.where(
            df["pickup_eta"].isna() | (df["pickup_eta"] < df["vehicle_arrived_at"]),
            np.where(
                (
                    df["vehicle_arrived_at"] + pd.Timedelta(seconds=avg_boarding_time)
                    < df["dropoff_at"]
                )
                | (df["dropoff_at"].isna() == True),
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
    pickup_eta = pd.to_datetime(df["pickup_eta"])

    pickup_eta = pickup_eta.fillna(df["pickup_at"])

    return pickup_eta


# Attribute: 'pickup_first_eta'
def clean_pickup_first_eta(df):
    pickup_first_eta = pd.to_datetime(df["pickup_first_eta"])

    pickup_first_eta = pickup_first_eta.fillna(df["pickup_eta"])

    return pickup_first_eta


# Attribute: 'dropoff_at'
def clean_dropoff_at(df):
    dropoff_at = pd.to_datetime(df["dropoff_at"])
    ftr = [3600, 60, 1]
    shortest_ridetime = (
        df["shortest_ridetime"]
        .str[0:8]
        .apply(lambda row: sum([a * b for a, b in zip(ftr, map(int, row.split(":")))]))
    )

    dropoff_at = np.where(
        (dropoff_at.isna()) & (df["state"] == "completed"),
        np.where(
            df["dropoff_eta"].isna(),
            dropoff_at + pd.to_timedelta(shortest_ridetime),
            df["dropoff_eta"],
        ),
        dropoff_at,
    )
    dropoff_at = pd.to_datetime(dropoff_at)

    # Check ordering
    dropoff_at = np.where(
        (dropoff_at < df["pickup_at"]),
        np.where(
            df["dropoff_eta"].isna(),
            dropoff_at + pd.to_timedelta(shortest_ridetime),
            df["dropoff_eta"],
        ),
        dropoff_at,
    )

    dropoff_at = pd.to_datetime(dropoff_at)

    return dropoff_at


# Attribute: 'dropoff_eta'
def clean_dropoff_eta(df):
    dropoff_eta = pd.to_datetime(df["dropoff_eta"])

    dropoff_eta = dropoff_eta.fillna(df["dropoff_at"])

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
        df["pickup_first_eta"] + pd.to_timedelta(shortest_ridetime)
    )
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


def data_cleaning(df, df_stops):

    time_columns = {
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

    df, df_inconsistencies = check_format(df, time_columns)
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
    return df


if __name__ == "__main__":
    repo = git.Repo(".", search_parent_directories=True).git.rev_parse(
        "--show-toplevel"
    )
    df = pd.read_csv(f"{repo}/data/rides_combined.csv", index_col=0)
    df_stops = pd.read_excel(
        f"{repo}/data/other/MoDstops+Preismodell.xlsx", sheet_name="MoDstops"
    )

    df = clean_duplicates(df)

    df = data_cleaning(df, df_stops)
   

    df.to_csv(f"{repo}/data/cleaning/test_{int(time.time())}.xlsx", index = False)

    print("Done!")
