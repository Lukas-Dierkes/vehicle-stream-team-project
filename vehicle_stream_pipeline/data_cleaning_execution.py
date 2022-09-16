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
