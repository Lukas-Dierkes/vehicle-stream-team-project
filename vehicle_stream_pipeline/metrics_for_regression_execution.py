import git
import pandas as pd

from vehicle_stream_pipeline.utils import feasibility_analysis as fa

# fetch data to built graphs from
repo = git.Repo(".", search_parent_directories=True).git.rev_parse("--show-toplevel")
df_edges = pd.read_excel(
    f"{repo}/data/other/MoDstops+Preismodell.xlsx", sheet_name="Liste 2022"
)
df_edges.rename(columns={"Start #": "start_id", "Ende #": "end_id"}, inplace=True)
df_simulated = pd.read_csv(f"{repo}/data/other/rides_main_routes.csv")

# execute regression metrics funciton to diameter and average_shortes_path for graphs with and without drones
regression_metrics = fa.getRegressionMetrics(df_simulated, df_edges, 10000)

# save regression metrics as csv
regression_metrics.to_csv(f"{repo}/data/regression/graph_metrics_500Ksteps.csv")
