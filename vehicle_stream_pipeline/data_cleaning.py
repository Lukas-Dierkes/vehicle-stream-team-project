import time
from datetime import datetime as dt
from re import M

import git
import numpy as np
import pandas as pd


# Format-Check
def check_format(df, col_type_dict):
    # check time format in order to avoid errors in cleaning
    for col, col_type in col_type_dict.items():
        if col_type == "timestamp":
            df = df[
                (
                    df[col].str.match(
                        r"[0-9]{1,4}.[0-9]{1,2}.[0-9]{1,2} [0-9]{1,2}:[0-9]{1,2}"
                    )
                    == True
                )
                | (df[col].isna())
            ]
            df_inconsistencies = df[
                ~(
                    (
                        df[col].str.match(
                            r"[0-9]{1,4}.[0-9]{1,2}.[0-9]{1,2} [0-9]{1,2}:[0-9]{1,2}"
                        )
                        == True
                    )
                    | (df[col].isna())
                )
            ]
        elif col_type == "time":
            df = df[
                (df[col].str.match(r"[0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2}") == True)
                | (
                    df[col].str.match(r"[0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2}.[0-9]*")
                    == True
                )
                | (df[col].str.contains("1899"))
                | (df[col].str.contains("1900") | (df[col].isna()))
            ]
            df_inconsistencies = df[
                ~(
                    (df[col].str.match(r"[0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2}") == True)
                    | (df[col].str.contains("1899"))
                    | (df[col].str.contains("1900"))
                    | (df[col].isna())
                )
            ]
    return (df, df_inconsistencies)


# Attribute: 'id'
def clean_ride_id(df):
    id = pd.DataFrame(data=df.loc[:, "id"], columns=["id"])
    id.id.fillna(
        df.created_from_offer.astype(float).astype("Int64"), inplace=True
    )  # Why first to float and then to Int64
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
        # different naming
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
    # file = "../cleaning/unmatched_addresses_{}.xlsx".format(int(time.time()))
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
    # Ansonsten scheint die order korrekt zu sein
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

    ##### Ich verstehe die 1. Bedingungen hier nicht:
    # 1. Bedingung: Wenn dispatched vor oder gleich created ist, dann soll scheduled_to-8Min genommen werden
    #       Aber vorher wurden zB alle NaN von dispatched mit created gefüllt, d.h. bei diesen rides
    #       wird immer scheduled_to-8Min genommen auch für den Fall, dass es keine Vorbuchung ist und somit
    #       scheduled_to im vorherigen Cleaning auf created_at gesetzt wurde. Und eigentlich sind doch alle rides, die
    #       nicht vorbestellt wurden direkt gleichzeit zu created_at auch dispatched_at und fallen somit darein

    # Lösung: Strikt kleiner als kleiner gleich

    # Check correct ordering
    dispatched_at = np.where(
        (dispatched_at < df["created_at"])
        | (dispatched_at <= df["scheduled_to"] - pd.Timedelta(minutes=9)),
        df["scheduled_to"] - pd.Timedelta(minutes=8),
        dispatched_at,
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
    print(avg_pickup_arrival_time)

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

    ##### Könnnen wir bei der ersten Bedigung arrived_at < arriving_push pauschal sagen, dass es falsch ist?
    # Es kann ja auch sein, dass der Fahrer früher als erwartet ankommt und die Push Benachrichtigung zu spät kommt.
    # Vielleicht sollte man hier auch einen Puffer einbauen, um so was zu umgehen, also zB arrived_at + 60Min.
    # - Und, wenn arrived_at < arriving_push und arriving_push is not nan, dann müsste arrived_at doch arriving_push + 3Min. sein?
    #   Ich habe nämlich jetzt recht viele arrival_deviations mit -180 sek.

    # Check ordering
    vehicle_arrived_at = np.where(
        (vehicle_arrived_at < arriving_push)
        | (vehicle_arrived_at + pd.Timedelta(minutes=60) < df["scheduled_to"]),
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
            arriving_push,
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


# Attribute: 'pickup_arrival_time'
def clean_pickup_arrival_time(df):
    pickup_arrival_time = (df["vehicle_arrived_at"] - df["dispatched_at"]).dt.seconds
    return pickup_arrival_time


# Attribute: 'earliest_pickup_expectation'
def clean_earlierst_pickup_expectation(df):
    earlierst_pickup_expectation = pd.to_datetime(df["earliest_pickup_expectation"])
    earlierst_pickup_expectation = df["dispatched_at"] + pd.Timedelta(minutes=3)

    return earlierst_pickup_expectation


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
    print(avg_boarding_time)

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
            df["pickup_eta"].isna(),
            np.where(
                (
                    df["vehicle_arrived_at"] + pd.Timedelta(seconds=avg_boarding_time)
                    < df["dropoff_at"]
                )
                | (df["dropoff_at"].isna() == True),
                df["vehicle_arrived_at"] + pd.Timedelta(seconds=avg_boarding_time),
                df["dropoff_at"],
            ),
            df["pickup_eta"],
        ),
        pickup_at,
    )

    pickup_at = pd.to_datetime(pickup_at)

    return pickup_at


def clean_pickup_eta(df):
    pickup_eta = pd.to_datetime(df["pickup_eta"])

    pickup_eta = pickup_eta.fillna(df["pickup_at"])

    return pickup_eta


def clean_pickup_first_eta(df):
    pickup_first_eta = pd.to_datetime(df["pickup_first_eta"])

    pickup_first_eta = pickup_first_eta.fillna(df["pickup_eta"])

    return pickup_first_eta


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


def clean_dropoff_eta(df):
    dropoff_eta = pd.to_datetime(df["dropoff_eta"])

    dropoff_eta = dropoff_eta.fillna(df["dropoff_at"])

    return dropoff_eta


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


def clean_time_periods(df):
    df["arrival_deviation2"] = df.apply(
        lambda row: (
            (row["vehicle_arrived_at"] - row["arriving_push"]).round(freq="s")
        ).total_seconds()
        - 180
        if (row["vehicle_arrived_at"] == row["vehicle_arrived_at"])
        and (row["arriving_push"] == row["arriving_push"])
        else np.NaN,
        axis=1,
    )

    df["waiting_time_s"] = df.apply(
        lambda row: (pd.to_timedelta(row["waiting_time"]).total_seconds())
        if (row["waiting_time"] == row["waiting_time"])
        and (len(row["waiting_time"]) == 8)
        else np.NaN,
        axis=1,
    )
    df["waiting_time2"] = df.apply(
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

    df["boarding_time_s"] = df.apply(
        lambda row: (pd.to_timedelta(row["boarding_time"]).total_seconds())
        if (row["boarding_time"] == row["boarding_time"])
        and (len(row["boarding_time"]) == 8)
        else (
            dt.strptime(row["boarding_time"], "%Y-%m-%d %H:%M:%S") - dt(1900, 1, 1)
        ).total_seconds()
        if (row["boarding_time"] == row["boarding_time"])
        else np.NaN,
        axis=1,
    )
    df["boarding_time2"] = df.apply(
        lambda row: (
            (row["pickup_at"] - row["vehicle_arrived_at"]).round(freq="s")
        ).total_seconds()
        if (row["vehicle_arrived_at"] == row["vehicle_arrived_at"])
        and (row["pickup_at"] == row["pickup_at"])
        else np.NaN,
        axis=1,
    )

    df["ride_time_s"] = df.apply(
        lambda row: (pd.to_timedelta(row["ride_time"]).total_seconds())
        if (row["ride_time"] == row["ride_time"])
        else np.NaN,
        axis=1,
    )
    df["ride_time2"] = df.apply(
        lambda row: (
            (row["dropoff_at"] - row["pickup_at"]).round(freq="s")
        ).total_seconds()
        if (row["dropoff_at"] == row["dropoff_at"])
        and (row["pickup_at"] == row["pickup_at"])
        else np.NaN,
        axis=1,
    )

    df["trip_time_s"] = df.apply(
        lambda row: (pd.to_timedelta(row["trip_time"]).total_seconds())
        if (row["trip_time"] == row["trip_time"]) and (len(row["trip_time"]) == 8)
        else np.NaN,
        axis=1,
    )
    df["trip_time2"] = df.apply(
        lambda row: (
            row["ride_time2"] + row["waiting_time2"]  # '2' weg wenn wir überschreiben
        ),
        axis=1,
    )

    df["shortest_ridetime_s"] = df.apply(
        lambda row: (
            pd.to_timedelta(row["shortest_ridetime"]).round(freq="s").total_seconds()
        )
        if (row["shortest_ridetime"] == row["shortest_ridetime"])
        else np.NaN,
        axis=1,
    )

    df["delay2"] = df.apply(
        lambda row: (
            row["trip_time2"]
            - row["shortest_ridetime_s"]
            # '2' weg wenn wir überschreiben
        ),
        axis=1,
    )
    df["delay_s"] = df.apply(
        lambda row: (pd.to_timedelta(row["delay"]).total_seconds())
        if (row["delay"] == row["delay"]) and (len(row["delay"]) == 8)
        else np.NaN,
        axis=1,
    )

    df["longer_route_factor2"] = df.apply(
        lambda row: round(row["ride_time2"] / row["shortest_ridetime_s"], 2)
        if (row["shortest_ridetime_s"] != 0)
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

    df_inconsistencies.to_excel(
        f"{repo}/data/cleaning/inconsistencies_{int(time.time())}.xlsx"
    )

    print("clean id")
    df["id"] = clean_ride_id(df)

    print("clean distance")
    df = clean_distance(df)  # Hier mit gesamten df, da Zeilen gelöscht werden

    print("clean addresses")
    df[["pickup_address", "dropoff_address"]] = clean_addresses(df, df_stops)

    # Zeistempel-Korrektur, daher nur completed rides:
    # df_copy = df.copy()
    # df = df[(df["state"] == "completed")]
    # df_copy = df_copy[df_copy["state"] != "completed"]

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

    # ggf. zu Zeitperioden hinzufügen
    print("clean pickup_arrival_time")
    df["pickup_arrival_time"] = clean_pickup_arrival_time(df)

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

    # time periods nach Korrektur der Zeitstempel
    # df = (pd.concat([df, df_copy], ignore_index=False)).sort_index()

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

    # df = pd.read_csv("../data/rides_combined.csv", index_col=0)
    # df_stops = pd.read_excel("../data/MoDstops+Preismodell.xlsx", sheet_name="MoDstops")

    df = df[
        df["id"].isnull() | ~df[df["id"].notnull()].duplicated(subset="id", keep="last")
    ]

    df = data_cleaning(df, df_stops)
    # df.to_excel(f"{repo}/data/cleaning/test_{int(time.time())}.xlsx")

    df.to_excel("../cleaning/test_{}.xlsx".format(int(time.time())))
    print("Done!")
