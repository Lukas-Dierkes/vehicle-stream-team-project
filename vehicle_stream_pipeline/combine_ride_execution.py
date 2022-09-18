"""
    In this file, the function create_overall_dataframe() is called, which creates three large data frames from the given excel files of MoD, so that we combine the data from all months.

    Input files:
        - Rides_XXX.xlsx: ride data of all months to be combined
        - MoDstops+Preismodell.xlsx: MoDStops table and all possible routes with distances

    Output files:
        - kpi_combined.csv: monthly kpi statistics combined
        - mtd_combined.csv: all the rides for each day of the month combined according to the excel spreadsheet
        - rides_combined.csv: we iterate over each daily sheet and collect the data for each day itself
"""
import git

from vehicle_stream_pipeline.utils import data_cleaning as dc

repo = git.Repo(".", search_parent_directories=True).git.rev_parse("--show-toplevel")

# Combine all rides from monthly excel sheets (Takes about one minute)
all_rides = dc.create_overall_dataframes(f"{repo}/data/normal_rides")

all_rides["df_kpi"].to_csv(f"{repo}/data/other/kpi_combined.csv")
all_rides["df_mtd"].to_csv(f"{repo}/data/other/mtd_combined.csv")
all_rides["df_rides"].to_csv(f"{repo}/data/other/rides_combined.csv")
