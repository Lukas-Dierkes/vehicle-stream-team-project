import git

from vehicle_stream_pipeline.utils import data_cleaning as dc

repo = git.Repo(".", search_parent_directories=True).git.rev_parse("--show-toplevel")

# Combine all rides from monthly excel sheets (Takes about one minute)
all_rides = dc.create_overall_dataframes(f"{repo}/data/normal_rides")

all_rides["df_kpi"].to_csv(f"{repo}/data/other/kpi_combined.csv")
all_rides["df_mtd"].to_csv(f"{repo}/data/other/mtd_combined.csv")
all_rides["df_rides"].to_csv(f"{repo}/data/other/rides_combined.csv")
