"""
    This script reads all required files and then automatically eliminates the duplicates, cleans the data, adds shared rides and checks if the data is correctly calculated and ordered. For the datapipeline we have made use of the descriptions and equations of MoD for the different attributes.
    Input files:
        - rides_combined.csv: all ride data combined in one csv file
        - MoDstops+Preismodell.xlsx: MoDStops table and all possible routes with distances
        - MoD_Vehicle Usage_2021+2022-05-15.xlsx: necessary to mach old rides (until May) with vehicle ID
        - Autofleet_Rides with External ID_2021+2022-05-15.xlsx: necessary to mach old rides (until May) with vehicle ID

    Output files:
        - unmatched_addresses_{int(time.time())}.xlsx: this file lists all rides that contain a pickup_address or dropoff_address that can not be matched with the MoDStops list
        - incorrect{int(time.time())}.xlsx: Excel file with incorrect ride data after cleaning (quality assurance)
        - data_cleaned.csv: clean ride data that can be used for further analyses

"""
import time

import git
import pandas as pd

from vehicle_stream_pipeline.utils import data_cleaning as dc

repo = git.Repo(".", search_parent_directories=True).git.rev_parse("--show-toplevel")
df = pd.read_csv(f"{repo}/data/other/rides_combined.csv", index_col=0)
df_stops = pd.read_excel(
    f"{repo}/data/other/MoDstops+Preismodell.xlsx", sheet_name="MoDstops"
)
vehicle_usage_df = pd.read_excel(
    f"{repo}/data/vehicle_data/MoD_Vehicle Usage_2021+2022-05-15.xlsx"
)
external_df = pd.read_excel(
    f"{repo}/data/vehicle_data/Autofleet_Rides with External ID_2021+2022-05-15.xlsx"
)

df = dc.clean_duplicates(df)

df = dc.data_cleaning(df, df_stops)

#df = dc.add_shared_rides(df, vehicle_usage_df, external_df)

print("check cleaned data")
df, df_incorrect = dc.data_check(df)
if df_incorrect.empty == False:
    df_incorrect.to_excel(f"{repo}/data/cleaning/incorrect{int(time.time())}.xlsx")

df.to_csv(f"{repo}/data/cleaning/data_cleaned.csv", index=False)

print("Done!")
