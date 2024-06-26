"""
    The script reads all required files and then automatically extracts the date range in the cleaned ride data, to execute the ride simulation for every given month.

    Input files:
        - data_cleaned.csv: clean ride data that can be used for further analyses
        - MoDstops+Preismodell.xlsx: MoDStops table and all possible routes with distances

    Output files:
        - incorrect{int(time.time())}.xlsx: Excel file with incorrect simulated ride data (quality assurance)
        - ride_simulation.csv: simulated ride data
"""

import math
import warnings
from datetime import datetime as dt

import git
import numpy as np
import pandas as pd

from vehicle_stream_pipeline.utils import ride_simulation as rs

warnings.filterwarnings("ignore")

# fetch orginal data to simulate from
repo = git.Repo(".", search_parent_directories=True).git.rev_parse("--show-toplevel")

df_stops = pd.read_excel(
    f"{repo}/data/other/MoDstops+Preismodell.xlsx", sheet_name="MoDstops"
)

df_edges = pd.read_excel(
    f"{repo}/data/other/MoDstops+Preismodell.xlsx", sheet_name="Liste 2022"
)
df_edges.rename(columns={"Start #": "start_id", "Ende #": "end_id"}, inplace=True)

rides_df = pd.read_csv(f"{repo}/data/cleaning/data_cleaned.csv")
rides_df = rides_df[(rides_df["state"] == "completed")]
rides_df["scheduled_to"] = pd.to_datetime(rides_df["scheduled_to"])

# get date range from orignal data, to execute rides simulation for every given month
start_date = min(rides_df["scheduled_to"])
end_date = max(rides_df["scheduled_to"])
date_range = rs.get_date_range(start_date, end_date)
data_range_len = len(date_range)

# simulate rides
month_sim_rides = 5000  # CHANGE to Adjust monthly number of simualted rides
new_rides_all = pd.DataFrame(columns=rides_df.columns)
for (year, month) in date_range:
    print("\nSimulation started for month", month, "in year", year)
    new_rides = rs.generateRideSpecs(
        rides_df,
        df_stops,
        df_edges,
        month_sim_rides,
        month,
        year,
    )
    new_rides_all = pd.concat([new_rides, new_rides_all])

# check simulated rides for inconsistencies
new_rides_all, new_rides_all_incorrect = dc.data_check(new_rides_all)
if new_rides_all_incorrect.empty == False:
    print("Inconsistencies found and saved in /data/simulated/")
    new_rides_all_incorrect.to_excel(
        f"{repo}/data/simulated/incorrect{int(time.time())}.xlsx"
    )

# save simulated rides as csv
new_rides_all.to_csv(f"{repo}/data/simulated/ride_simulation.csv")
