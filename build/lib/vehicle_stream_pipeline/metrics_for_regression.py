import git
import pandas as pd

from vehicle_stream_pipeline import utils

repo = git.Repo(".", search_parent_directories=True).git.rev_parse("--show-toplevel")
df_edges = pd.read_excel(
    f"{repo}/data/other/MoDstops+Preismodell.xlsx", sheet_name="Liste 2022"
)
df_edges.rename(columns={"Start #": "start_id", "Ende #": "end_id"}, inplace=True)
df_simulated = pd.read_csv(f"{repo}/data/simulated/sim_rides_2m.csv")

regression_metrics = utils.getRegressionMetrics(df_simulated, df_edges, 500000)

regression_metrics.to_csv(f"{repo}/data/regression/graph_metrics_500Ksteps.csv")
