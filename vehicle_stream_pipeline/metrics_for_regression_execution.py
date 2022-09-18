"""
    This script calls the function getRegressionMetrics(), which samples rides for an increasing stepsize, transforms the data samples to a weighted directed graph and calculates the graph metrics diameter/longest shortest past and average shortest path for it.

    Input files:
        - ride_simulation.csv: simulated ride data
        - MoDstops+Preismodell.xlsx: MoDStops table and all possible routes with distances

    Output files:
    - graph_metrics_test.csv: matrix with graph metrics considering all MoDStops as vertices in the graphs
    - graph_metrics_main_routes_test: matrix with graph metrics considering only main MoDStops as vertices in the graphs
"""

import git
import pandas as pd

from vehicle_stream_pipeline.utils import feasibility_analysis as fa

# fetch data to built graphs from
repo = git.Repo(".", search_parent_directories=True).git.rev_parse("--show-toplevel")
df_edges = pd.read_excel(
    f"{repo}/data/other/MoDstops+Preismodell.xlsx", sheet_name="Liste 2022"
)
df_edges.rename(columns={"Start #": "start_id", "Ende #": "end_id"}, inplace=True)
df_simulated = pd.read_csv(f"{repo}/data/simulated/ride_simulation.csv")

# Filter simulations for only rides between hotspots
main_spots = [1008, 4025, 6004, 12007, 11017, 15013, 3021, 8001, 5001, 11003, 4016]
rides_main_routes = df_simulated[
    (df_simulated["pickup_address"].isin(main_spots))
    & (df_simulated["dropoff_address"].isin(main_spots))
]


# execute regression metrics funciton to diameter and average_shortes_path for graphs with and without drones
regression_metrics = fa.getRegressionMetrics(
    df_simulated, df_edges, 500000, 10000, only_main_routes=False
)
regression_metrics_main_routes = fa.getRegressionMetrics(
    rides_main_routes, df_edges, 100, 1000, only_main_routes=True
)

# save regression metrics as csv
regression_metrics.to_csv(f"{repo}/data/regression/graph_metrics.csv")
regression_metrics_main_routes.to_csv(
    f"{repo}/data/regression/graph_metrics_main_routes.csv"
)
