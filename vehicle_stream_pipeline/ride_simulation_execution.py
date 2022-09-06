import math
import warnings
from datetime import datetime as dt

import git
import numpy as np
import pandas as pd

from vehicle_stream_pipeline import utils

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

rides_df = pd.read_excel(f"{repo}/data/cleaning/data_cleaned_1808.xlsx")
rides_df = rides_df[(rides_df["state"] == "completed")]
rides_df["scheduled_to"] = pd.to_datetime(rides_df["scheduled_to"])

# get date range from orignal data, to execute rides simulation for every given month
start_date = min(rides_df["scheduled_to"])
end_date = max(rides_df["scheduled_to"])
date_range = utils.get_date_range(start_date, end_date)
data_range_len = len(date_range)

# simulate rides
total_sim_rides = 100  # CHANGE to Adjust total number of simualted rides
month_sim_rides = math.ceil(
    total_sim_rides / data_range_len
)  # No. of simulated rides per month
new_rides_all = pd.DataFrame(columns=rides_df.columns)
for (year, month) in date_range:
    new_rides = utils.generateRideSpecs(
        rides_df,
        df_stops,
        df_edges,
        month_sim_rides,
        month,
        year,
    )
    new_rides_all = pd.concat([new_rides, new_rides_all])

# save simulated rides as csv
new_rides_all.to_csv(f"{repo}/data/simulated/sim_rides_test.csv")
