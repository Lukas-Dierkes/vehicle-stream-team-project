# Setup
import dash
import dash_bootstrap_components as dbc
import git
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from dash import MATCH, Input, Output, callback, dcc, html

from vehicle_stream_pipeline.utils import feasibility_analysis as fa

repo = git.Repo(".", search_parent_directories=True).git.rev_parse("--show-toplevel")

# Read in graph metrics for all routes
graph_metrics_df = pd.read_csv(f"{repo}/data/regression/graph_metrics_.csv")

# Read in graph metrics for main routes
graph_metrics_main_routes_df = pd.read_csv(
    f"{repo}/data/regression/graph_metrics_main_routes.csv"
)

# Read in cleaned data and convert some features
rides_df = pd.read_csv(f"{repo}/data/cleaning/data_cleaned.csv")
rides_df = rides_df[(rides_df["state"] == "completed")]
rides_df["scheduled_to"] = pd.to_datetime(rides_df["scheduled_to"])
rides_df["dropoff_at"] = pd.to_datetime(rides_df["dropoff_at"])
rides_df["dispatched_at"] = pd.to_datetime(rides_df["dispatched_at"])

# Read in simulated rides and convert some features
simulated_rides = pd.read_csv(f"{repo}/data/simulated/ride_simulation.csv")
simulated_rides["scheduled_to"] = pd.to_datetime(simulated_rides["scheduled_to"])
simulated_rides["dropoff_at"] = pd.to_datetime(simulated_rides["dropoff_at"])
simulated_rides["dispatched_at"] = pd.to_datetime(simulated_rides["dispatched_at"])

dash.register_page(__name__)

controls = dbc.Card(
    [
        html.Hr(),
        # Maximum Delivery days input
        dbc.Row(
            [
                html.Label("Maximum Delivery Days"),
                html.Div(
                    dcc.Input(
                        id="input_number_max_days",
                        type="number",
                        value=5,
                    ),
                ),
            ],
            # className="d-grid gap-2 col-6 mx-auto",
        ),
        html.Hr(),
        # Dropdown for metrics
        dbc.Row(
            [
                html.Label("Graph Metric"),
                dcc.Dropdown(
                    id="dropdown_graph_metric",
                    options=[
                        {
                            "label": "Maximum Delivery Days without Drones",
                            "value": "diameter_w/o_drones",
                        },
                        {
                            "label": "Maximum Delivery Days with Drones",
                            "value": "diameter_with_drones",
                        },
                        {
                            "label": "Average Delivery Days without Drones",
                            "value": "avg_w/o_drones",
                        },
                        {
                            "label": "Average Delivery Days with Drones",
                            "value": "avg_with_drones",
                        },
                    ],
                    value="avg_w/o_drones",
                    clearable=False,
                ),
            ],
        ),
        html.Hr(),
        # Slider for the percentage rides combined
        dbc.Row(
            [
                html.Label("Percentage rides combined"),
                html.Div(
                    dcc.Slider(0, 1, 0.1, value=0.2, id="combined_rides_factor"),
                ),
            ]
        ),
        html.Hr(),
        # Radio items to choose between main routes and all routes
        dbc.Row(
            [
                html.Div(
                    dcc.RadioItems(
                        id={
                            "type": "dynmaic-dpn-main-routes",
                        },
                        options=[
                            {"label": "Only main routes", "value": "1"},
                            {"label": "All routes", "value": "0"},
                        ],
                        value="0",
                        labelStyle={"display": "block"},
                    )
                )
            ]
        ),
    ]
)


layout = dbc.Container(
    [
        html.H1("Ride Simulation Analysis", style={"textAlign": "center"}),
        dbc.Row(
            [
                dbc.Col([controls], width=3),
                dbc.Col(
                    [
                        # Needed rides plot
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(id="needed_rides_plot"),
                                ),
                                dbc.Col(
                                    html.Div(
                                        id="needed_rides_text",
                                        style={"whiteSpace": "pre-line"},
                                    ),
                                    width=3,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                # Plot for drivers needed
                                dbc.Col(
                                    [
                                        dcc.Graph(id="fig_drivers"),
                                    ]
                                ),
                                # Plot for amout of parallel drives
                                dbc.Col([dcc.Graph(id="fig_drives")]),
                            ]
                        ),
                    ]
                ),
            ]
        ),
    ]
)


@callback(
    [
        Output("needed_rides_plot", "figure"),
        Output("fig_drivers", "figure"),
        Output("fig_drives", "figure"),
        Output("needed_rides_text", "children"),
    ],
    [
        Input("input_number_max_days", "value"),
        Input("dropdown_graph_metric", "value"),
        Input("combined_rides_factor", "value"),
        Input(
            component_id={"type": "dynmaic-dpn-main-routes"},
            component_property="value",
        ),
    ],
)
def update_charts(
    max_days=5,
    current_metric="avg_w/o_drones",
    combined_rides_factor=0.2,
    main_routes="0",
):
    # Use dataframes from global name space so you don't need to read them every time you cange an input
    global graph_metrics_df
    global graph_metrics_main_routes_df
    global rides_df
    global simulated_rides

    # Choose between graph metrics based on all routes or just the main routes depending on the input value
    if main_routes == "0":
        graph_metrics_df_1 = graph_metrics_df.copy()
    else:
        graph_metrics_df_1 = graph_metrics_main_routes_df.copy()

    # Calculate needed rides and master data for graph
    range_x = [0,max_days+15]
    range_y = [0, fa.get_rides_num(0.9, graph_metrics_df_1, current_metric)]
    x_data_regression = np.linspace(0,graph_metrics_df_1[current_metric].max()*1.1,len(graph_metrics_df_1[current_metric])*20) #needed for flawless regression curve
    needed_rides = fa.get_rides_num(max_days, graph_metrics_df_1, current_metric)

    # Update Figure diplaying Orginal Data Scatter Plot, Regression and max_days_line
    # Scatter Plot of Orginal Rides Data
    needed_rides_fig1 = px.scatter(
        graph_metrics_df_1,
        x=current_metric,
        y="#_simulated_rides",
        color_discrete_sequence=["DarkKhaki"],
        title="Break Even of Rides",
        range_x=range_x,
        range_y=range_y,
    )
    needed_rides_fig1["data"][0]["name"] = "Simulated Rides Data"
    needed_rides_fig1["data"][0]["showlegend"] = False
    # Line Plot of Regressed Data
    needed_rides_fig2 = px.line(
        x=x_data_regression,
        y=fa.regression_function(
            x_data_regression,
            *fa.get_opt_parameter(graph_metrics_df_1, current_metric),
        ),
        color_discrete_sequence=["DarkCyan"],
        range_x=range_x,
    )
    needed_rides_fig2["data"][0]["name"] = "Regression of Rides Data"
    needed_rides_fig2["data"][0]["showlegend"] = False
    # Line Plot working as cursor for current max days
    needed_rides_fig3 = px.line(
        x=[max_days, max_days], y=[0, needed_rides], color_discrete_sequence=["tomato"]
    )
    needed_rides_fig3["data"][0]["name"] = "Max Days for Delivery"
    needed_rides_fig3["data"][0]["showlegend"] = False
    needed_rides_fig4 = px.line(
        x=[0, max_days],
        y=[needed_rides, needed_rides],
        color_discrete_sequence=["tomato"],
    )

    # Combine all plots into on figure
    needed_rides_fig = go.Figure(
        data=needed_rides_fig1.data
        + needed_rides_fig2.data
        + needed_rides_fig3.data
        + needed_rides_fig4.data,
        layout=needed_rides_fig1.layout,
    )

    # Calculate needed drivers

    # Sample as many rides as needed as calculated before.
    if needed_rides - len(rides_df) > 0:
        simulated_rides_1 = simulated_rides.sample(int(needed_rides - len(rides_df)))
        total_rides = pd.concat([rides_df, simulated_rides_1])
    else:
        total_rides = rides_df.sample(int(needed_rides))

    # For each hour calculate how many parallel drives are existing in the sample and thus how many drivers we need
    hours = list(range(0, 24))
    numb_drivers_per_hour = []
    avg_drives_per_hour = []
    for i in hours:
        numb_drivers_per_hour.append(
            fa.calculate_number_drivers(total_rides, i, combined_rides_factor)[0]
        )
        avg_drives_per_hour.append(
            fa.calculate_number_drivers(total_rides, i, combined_rides_factor)[1]
        )

    # Store number of drivers and hour in dataframe
    df_drivers_per_hour = pd.DataFrame(
        list(zip(hours, numb_drivers_per_hour)), columns=["hour", "drivers"]
    )

    # Store number of parallel drives and hour in a dataframe
    df_drives_per_hour = pd.DataFrame(
        list(zip(hours, avg_drives_per_hour)), columns=["hour", "drives"]
    )

    # build figures for drivers per hour and parallel driver per hour
    fig_drivers = px.bar(df_drivers_per_hour, x="hour", y="drivers")
    fig_drives = px.bar(df_drives_per_hour, x="hour", y="drives")

    # Create a text for the outout showing the number of needed drives
    nl = "\n \n"
    needed_rides_text = (
        f"Total amount rides needed:{nl} {str(int(needed_rides))} per month"
    )

    return [
        needed_rides_fig,
        fig_drivers,
        fig_drives,
        needed_rides_text,
    ]  # output fig needs to be list in order work
