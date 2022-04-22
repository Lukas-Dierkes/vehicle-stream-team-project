import pandas as pd
import numpy as np
from pyrsistent import v
from functions_detect_inconsistencies import Detect
from numpy import NaN
from datetime import datetime as dt

# read rides_combined.csv - change to your loaction
df = pd.read_csv(
    "/Users/ericchittka/rides_combined.csv"
)

# check the types of the dataframe columns 
df.dtypes

# convert all column with dates into datetime objects 
df['created_at'] = pd.to_datetime(df['created_at'])
df['scheduled_to'] = pd.to_datetime(df['scheduled_to'])
df['dispatched_at'] = pd.to_datetime(df['dispatched_at'])
df['canceled_at'] = pd.to_datetime(df['canceled_at'])
df['arriving_push'] = pd.to_datetime(df['arriving_push'])
df['vehicle_arrived_at'] = pd.to_datetime(df['vehicle_arrived_at'])
df['earliest_pickup_expectation'] = pd.to_datetime(df['earliest_pickup_expectation'])
df['pickup_first_eta'] = pd.to_datetime(df['pickup_first_eta'])
df['pickup_eta'] = pd.to_datetime(df['pickup_eta'])
df['pickup_at'] = pd.to_datetime(df['pickup_at'])
df['dropoff_first_eta'] = pd.to_datetime(df['dropoff_first_eta'])
df['dropoff_eta'] = pd.to_datetime(df['dropoff_eta'])
df['dropoff_at'] = pd.to_datetime(df['dropoff_at'])
df['updated_at'] = pd.to_datetime(df['updated_at'])
df['pickup_arrival_time'] = pd.to_datetime(df['pickup_arrival_time'])
df['waiting_time'] = pd.to_datetime(df['waiting_time'], format='%H:%M:%S')
df['boarding_time'] = pd.to_datetime(df['boarding_time'])
df['ride_time'] = pd.to_datetime(df['ride_time'], format='%H:%M:%S')
df['trip_time'] = pd.to_datetime(df['trip_time'], format='%H:%M:%S')
df['delay'] = pd.to_datetime(df['delay'], format='%H:%M:%S')

# add empty columns {"Errorcode": "0", "Errormessage": ""}
df["Errorcode"] = "0"
df["Errormessage"] = ""

df.dtypes
# check every row for inconsistencies and right errorlist into df
for index, row in df.iterrows():

    # infinite loop protection - can be commented out later
    expression = index > 5
    if expression:
        print(expression)
        break

    # create empty errorlist per row
    errorlist = pd.Series({"Errorcode": "0", "Errormessage": ""})

    # use check function depending on status
    current_status = row["state"]
    if current_status == "completed":                            
        errorlist = Detect.check_state_completed(row, errorlist)
        errorlist = Detect.check_timestamp_order(row, errorlist)
        errorlist = Detect.check_timestamp_calculations(row, errorlist)
    elif current_status == "offer" or current_status == "offer-rejected":
        errorlist = Detect.check_state_offer(row, errorlist)
    elif current_status == "canceled":
        errorlist = Detect.check_state_canceled(row, errorlist)

    # update df columns Errorcode and Erromessage at index
    df.loc[index, ["Errorcode", "Errormessage"]] = errorlist


# filter output_df to error rows only (errorcode !=  0)
# filter also all unused columns here
filtered_df = df[df["Errorcode"] != "0"]

#print(df)
print(filtered_df)
