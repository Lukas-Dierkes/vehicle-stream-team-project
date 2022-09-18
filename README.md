# Table of Contents 
- [Table of Contents](#table-of-contents)
- [1. Open-Objective Data Mining of Real World Vehicle Data](#1-open-objective-data-mining-of-real-world-vehicle-data)
- [2. Technologies](#2-technologies)
- [3. Installation](#3-installation)
- [4. Folder Structure](#4-folder-structure)
- [5. How to Use the Project](#5-how-to-use-the-project)
  - [5.1 Combine external data](#51-combine-external-data)
  - [5.2 Data Pipeline](#52-data-pipeline)
    - [5.2.1 Functions](#521-functions)
  - [5.3 Ride Simulation](#53-ride-simulation)
    - [5.3.1 Execution](#531-execution)
    - [5.3.2 Main Function: generateRideSpecs()](#532-main-function-generateridespecs)
    - [5.3.3. Help Functions](#533-help-functions)
  - [5.4 Probablistic Graph Model](#54-probablistic-graph-model)
  - [5.5 Feasibilty Analysis](#55-feasibilty-analysis)

# 1. Open-Objective Data Mining of Real World Vehicle Data 
 
***
Within the team project of the University of Mannheim, 2 research projects of INES were presented:
- Mobile e-hub, which investigates how drones can use the rides of cars so that ordered packages can be delivered to customers by drones and cars
- HitchHikeBox, investigating how customer ordered packages can be delivered from boxes carried on other rides that are part of the environment

This project is about **analyzing real world data** of a startup (Mobility-on-Demand) to determine the feasibility of using existing mobility data to support Ines' projects. 

# 2. Technologies  
***
In this section a list of the **required technologies** used within the project is provided:
* [pip](https://pypi.org/project/pip/): Version 22.2.2
* [GitPython](https://git-scm.com): Version 3.1.27
* [pandas](https://pandas.pydata.org): Version 1.4.3
* [numpy](https://numpy.org): Version 1.20.2
* [matplotlib.pyplot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html): Version 3.5.2
* [plotly](https://plotly.com): Version 5.9.0
  * [plotly.express](https://plotly.com/python/plotly-express/)
  * [plotly.graph_objects](https://plotly.com/python/graph-objects/)
  * [plotly.figure_factory](https://plotly.com/python/figure-factory-subplots/)
  * [dash](https://dash.plotly.com/introduction): Version 2.4.0
  * [dash_bootstrap_components](https://pypi.org/project/dash-bootstrap-components/): Version 1.2.0
  * [dash_labs](https://pypi.org/project/dash-labs/): Version 1.0.8
  * [dash_html_components](https://dash.plotly.com/dash-html-components): Version 2.0.0
* [seaborn](https://seaborn.pydata.org): Version 0.12
* [datetime](https://docs.python.org/3/library/datetime.html)
* [math](https://docs.python.org/3/library/math.html)
* [os](https://docs.python.org/3/library/os.html)
* [re](https://docs.python.org/3/library/re.html)
  * [M](https://docs.python.org/3/library/re.html)
* [networkx](https://networkx.org): Version 3.0b1.dev0
* [scipy](https://scipy.org): Version 1.8.1
  * [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html)
  * [curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
  * [scipy_stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
* [collections](https://docs.python.org/3/library/collections.html)
* [json](https://docs.python.org/3/library/json.html): Version 0.9.8
* [geopandas](https://geopandas.org/en/stable/): Version 0.11.0
* [Shapley (shapely.geometry)](https://shapely.readthedocs.io/en/stable/manual.html): Version 1.7.1

# 3. Installation 
***
A little intro about the **installation** (macOS/Unix). First you need to clone the repository to one of the folders you have chosen. Next we worked in a **virtual environment**, to set it up you can proceed the following **steps**:   
```
$ git clone https://github.com/Lukas-Dierkes/vehicle-stream-team-project.git
$ cd ../path/to/the/file
$ python3 -m pip install --user --upgrade pip
$ python3 -m pip install --user virtualenv
$ cd ../path/to/the/project/directory
$ python3 -m venv env
$ source env/bin/activate
$ python3 -m pip install -r requirements.txt
```

At the end, install the **requirements.txt** file which should install all the required packages. However, there is one **exception**:   
```
$ pip uninstall networkx
$ git clone https://github.com/networkx/networkx.git
$ cd networkx
$ pip install -e .
```

# 4. Folder Structure 
***
This table is used to briefly illustrate the folder structure of the project.  
| Folder        | Description        |
| ------------- | ------------- |
| [/vehicle-stream-team-project/data](https://github.com/Lukas-Dierkes/vehicle-stream-team-project/tree/master/data) | Contains all **external & resulting data** in the subfolders |
| [/vehicle-stream-team-project/vehicle_stream_pipeline](https://github.com/Lukas-Dierkes/vehicle-stream-team-project/tree/master/vehicle_stream_pipeline)  | Contains all **execution scripts**, **the subfolders** for the **dashboard** and **utilities** scripts |
| [/vehicle-stream-team-project/vehicle_stream_pipeline/dashboard](https://github.com/Lukas-Dierkes/vehicle-stream-team-project/tree/master/vehicle_stream_pipeline/dashboard)  | Contains all the scripts to build and run the **dashboard** |
| [/vehicle-stream-team-project/vehicle_stream_pipeline/other](https://github.com/Lukas-Dierkes/vehicle-stream-team-project/tree/master/vehicle_stream_pipeline/other)  | Contains old scripts for connection to **SharePoint** and **SQL** |
| [/vehicle-stream-team-project/vehicle_stream_pipeline/utils](https://github.com/Lukas-Dierkes/vehicle-stream-team-project/tree/master/vehicle_stream_pipeline/dashboard)  | Contains all the scripts needed for the **data cleaning**, the **probablistic model**, the **ride simulation** and the **feasibility analysis** |

# 5. How to Use the Project  
***
In this section, the **individual steps** are shown and gone through using examples so that the **live dashboard** with the various analyses can be executed at the end. For a walkthrough, please refer to the **vehicle_stream_pipeline/walkthrough.ipynb** notebook.

## 5.1 Combine external data 
***
First of all, the individual data from MoD must be manually inserted from SharePoint into the folder structure. Since the data is confidential, you need an access permission to this data. 

Using the table, please assign the files to the appropriate folders:
| File          | Folder        | Description        |
| ------------- | ------------- | ------------- |
| ***Rides_.xlsx***  | [/vehicle-stream-team-project/data/normal_rides](https://github.com/Lukas-Dierkes/vehicle-stream-team-project/tree/master/data/normal_rides) | Include the normal rides downloaded from the MoD Sharepoint (usually one per month) |
| ***MoDstops+Preismodell.xlsx***  | [/vehicle-stream-team-project/data/other](https://github.com/Lukas-Dierkes/vehicle-stream-team-project/tree/master/data/other) | Include the Modstops files here which you can download from the MoD sharepoint |
| ***Autofleet_Rides with External ID_2021+2022-05-15.xlsx***/ ***MoD_Vehicle Usage_2021+2022-05-15.xlsx*** | [/vehicle-stream-team-project/data/vehicle_data](https://github.com/Lukas-Dierkes/vehicle-stream-team-project/tree/master/data/vehicle_data) | Include the rides with external IDs file & the vehicle usage report here which you can download from the MoD sharepoint |

After the files are inserted into the correct folders, you can start running the **vehicle_stream_pipeline/combine_ride_execution.py** script. In this file, the function ***create_overall_dataframe()*** is called, which creates three large data frames from the given excel files of MoD, so that we combine the data from all months.
1. **kpi_combined.csv**: This is the monthly kpi statistics combined (rarely used)
2. **mtd_combined.csv**: This file contains all the rides for each day of the month combined according to the excel spreadsheet
3. **rides_combined**: Here we iterated over each day and collected the data for each day itself. Surprisingly, this is different from the **mtd_combined.csv** and it seems that this data is more accurate. So we will use this data frame for further analysis

The above resulting .csv files are saved in the folder *data/other*.
## 5.2 Data Pipeline  
***
For starting the data pipeline you have to run the **vehicle_stream_pipeline/data_cleaning_execution.py** script, which reads all required files and then automatically eliminates the duplicates, cleans the data, adds shared rides and checks if the data is correctly calculated and ordered. For the datapipeline we have made use of the descriptions and equations of MoD for the different attributes. <br>
Based on the following table you can see an overview of the most important attributes of the **rides_combined.csv**:
| Attribute      | Description        |
| ------------- | ------------- |
| id | User id |
| user_id | Ride id |
| distance | Distance in meter |
| pickup_address | Start address of the ride |
| dropoff_address | End address of the ride |
| state | State of the ride (offer/ offer-rejected/ completed/ rejeceted/ canceled) |
| created_at | Creation date of the ride |
| scheduled_to | Prebooking time |
| dispatched_at | For offers that did not result in a booking, the time of origin of the offer is recorded here |
| arriving_push | Is sent automatically by Autofleet if the FMS assumes that the vehicle will reach the pickup in <= 3 minutes. This sometimes leads to misjudgements by the FMS, which then informs the customer too early, which in unfavorable cases leads to unnecessarily long waiting times at the MoDstop. Furthermore, it can lead to getting arrival_push several times |
| vehicle_arrived_at | Time when the driver arrives at the pickup_address and confirms this via button |
| pickup_arrival_time | vehicle_arrived_at - dipsatched_at |
| earliest_pickup_excepctation | dispatched_at + 3 min |
| pickup_first_eta | Earliest possible time when the client must be ready to drive from the pickup_address. Minimum here is created_at + 3 min (setting in Autofleet) |
| pickup_at | Actual pickup time |
| dropoff_first_eta | Earliest possible time the customer will arrive at the dropoff_address. This corresponds to pickup_first_eta + shortest ride time |
| dropoff_at | Actual dropoff time |
| arrival_deviation | vehicle_arrived_at - (arriving_push + 3 min) |
| waiting_time | vehicle_arrived_at - earliest_pickup_expectation |
| boarding_time | pickup_at - vehicle_arrived_at |
| ride_time | dropoff_at - pickup_at |
| trip_time | waiting_time + ride_time |
| shortest_ride_time | Fastest ride time between entry and exit |
| delay | trip_time - shortest_ride_time |
| shortest_ride_time | Fastest ride time between entry and exit |
| longer_route_factor | ride_time/shortest_ride_time |

**Required files:**
- rides_combined.csv
- MoDstops+Preismodell.xlsx
- Autofleet_Rides with External ID_2021+2022-05-15.xlsx

### 5.2.1 Functions
***
 1. Clean Duplicates 
   
    First the function ***clean_duplicates(df)*** is called. As the name already mentions, this function handles the **duplicates**. The lines which are a duplicate of another id are removed and the missing attributes of an id are completed with those of the duplicate. In the end we get a DataFrame without duplicate id's. 

 2. Data Cleaning

    The ***data_cleaning(df, df_stops)*** function performs several other functions to detect and correct inaccurate records from the data. Different assumptions were given by MoD or made in the team, so that the planned data correspond as realistically as possible to the actual system. <br>
    Below is a table explaining the various functions being used:

    | Function      | Description        |
    | ------------- | ------------- |
    | ***check_format(df, col_type_dic)*** | Checks the right format of the columns from the DataFrame (time, numerical, timestamp) |
    | ***clean_ride_id(df)*** | Fills empty id's |
    | ***clean_distance(df)*** | Removes rides where the pickup_address == dropoff_address |
    | ***clean_addresses(df, df_stops)*** | Checks if the addresses match those of the MoD Stop table and exports a list with the addresses that do not match |
    | ***clean_created_at(df)*** | Converts created_at column to datetime format |
    | ***clean_scheduled_to(df)*** | Fills the NaN values with the created_at time and notes that scheduled_to occurs after created_at |
    | ***clean_dispatched_at(df)*** | Fills dispatched_at = scheduled_to - 8Min, if it is a scheduled ride otherwise fill with created_at. Clean lines where scheduled_to - 8min is smaller than created_at because otherwise dispatched_at < created_at |
    | ***clean_vehicle_arrived_at(df)*** | Since pickup_arrival_time = vehicle_arrivd_at - dispatched_at, the NaN values are filled with dispatched_at + the average pickup arrival time calculated using the ***getAvgPickupArrivalTime(df)*** function. However, only if dispatched_at + average pickup time is less than pickup_at we add the average time to dispatched_at otherwise we take pickup_at. It also checks if vehicle_arrived_at occurs on the same day as scheduled_to. In special cases we use arriving_push + 3min as time |
    | ***clean_arriving_push(df)*** | Fills NaN values with vehicle_arrived_at - 3Min & checks the ordering e.g. that arriving_push is after created_at |
    | ***clean_earlierst_pickup_expectation(df)*** | Fills column with dispatched_at + 3min in normal cases when not a scheduled_ride. If it is a scheduled ride, then it is filled with scheduled_to + 5min |
    | ***clean_pickup_at(df)*** | For this, the average boarding time was calculated first, since boarding_time = pickup_at - vehicle_arrived_at. Fills the NaN values with pickup_eta or with vehicle_arrived_at + avg_boarding_time, if pickup_eta is too far away from scheduled_to. At the end we check the ordering |
    | ***clean_pickup_eta(df)*** | Fills the NaN values with pickup_at and checks the ordering. For example, pickup_eta must not occur before dispatched_at |
    | ***clean_pickup_first_eta(df)*** | Fills NaN values and other values of pickup_first_eta if it is not on the same day as scheduled_to with pickup_eta and checks the order |
    | ***clean_dropoff_at(df)*** | Reformats the shortest_ridetime column into seconds. NaN values are filled with dropoff_eta. If dropoff_eta is too far from scheduled_to, the dropoff_at values are calculated with dropoff_at + the shortest_ridetime. Checks ordering, i.e. that pickup_at time is before dropoff_at |
    | ***clean_dropoff_eta(df)*** | Fills NaN values with dropoff_at and checks ordering |
    | ***clean_dropoff_first_eta(df)*** | Fills NaN values with pickup_first_eta + shortest_ride_time and checks ordering. For example dropoff_first_eta doesn't take place before dispatched_at |
    | ***clean_time_periods(df)*** | After all timestamp attributes have been checked and filled, the time spans in between can be calculated using the MoD equations and entered into the respective columns |
    | ***clean_ratings(df)*** | Cleans the rating column |

    So, as you can see, in general the incorrect or non-existent values are filled with the values of other times, using given calculations (see table in 5.2) or making certain assumptions.  

 3. Add shared rides

    After the data is cleaned the **data_cleaning_execution.py** calls the function ***add_shared_rides(df, vehicle_usage_df, external_df)***. So the function uses 3 dataframes as input:
    1. *data/other/rides_combined.csv* 
    2. *data/vehicle_data/MoD_Vehicle Usage_2021+2022-05-15.xlsx*
    3. *data/vehicle_data/Autofleet_Rides with External ID_2021+2022-05-15.xlsx*

    At the beginning, filtering is done by stop point and status and then the remaining duplicates are removed. Afterwards the external dataframe is preprocessed and only the IDs contained in the vehicle usage dataframe are kept. Consequently the filtered vehicle usage table is joined to the external dataframe using a left join. After preparing the resulting dataframe for the join with the rides_combined table, we perform a left join between the dataframes removing the duplicates (see code snippet).
    ```
    rides_vehicle_merge_df = df.merge(
        vehicle_external_merge, how="left", left_on="id", right_on="External Id"
    )
    rides_vehicle_merge_df.drop(
        columns=["External Id", "Ride_id_external", "Ride_id_vehicle_usage"],
        inplace=True,
    )
    ```
    Based on the resulting data frame, we search for shared rides and add them in the "shared_rides" column. Next, empty columns are created for combined rides, which must be adjusted if more than 3 rides are combined. The quotes and rides without vehicle ID are omitted and the expressions correspond to the vehicle IDs and the different time scenarios. Finally, the function returns a dataframe containing shared rides.

    **Note**: Since from August the attributes in the Excel tables of MoD for the rides are different than before, the function has to be adapted. For this reason it is currently excluded in the code.

 4. Data check  
   
    Lastly, the ***data_check(df)*** function was called. This function is used as a test for our cleaned data and to remove records that we have classified as outliers. First the correct ordering of each timestamp is tested, such as for created_at. 
    ```
    # filter wrong ordering: created_at
    df_incorrect = df.loc[
        (df.created_at > df.scheduled_to)
        | (df.created_at > df.dispatched_at)
        | (df.created_at > df.arriving_push)
        | (df.created_at > df.vehicle_arrived_at)
        | (df.created_at > df.earliest_pickup_expectation)
        | (df.created_at > df.pickup_first_eta)
        | (df.created_at > df.pickup_eta)
        | (df.created_at > df.pickup_at)
        | (df.created_at > df.dropoff_first_eta)
        | (df.created_at > df.dropoff_eta)
        | (df.created_at > df.dropoff_at)
        | (df.created_at > df.updated_at)
    ]
    ```
    After that it is tested if the times between the timestamps were calculated correctly.  
    ```
     # boarding_time
    df_incorrect = df.loc[((df.pickup_at - df.vehicle_arrived_at).dt.seconds != df.boarding_time)]
    ```
    Finally, the largest outliers were filtered out so that the subsequent driving simulation is as close to everyday life as possible. It should be mentioned that these are not natural outliers, but those that were incorrectly calculated or entered by the system. Unfortunately, we could not handle every single case during the data cleaning process, because some of them are single data sets containing the error and if we would fix them, other factors have to be considered in other places, otherwise they would lead to errors again. 
     
## 5.3 Ride Simulation 
***
**Required files:**
- MoDstops+Preismodell.xlsx
- data_cleaned.csv
### 5.3.1 Execution
*** 
The ride simulation can be executed by running the python script **vehicle_stream_pipeline/ride_simulation_execution.py**. The script reads all required files and then automatically extracts the date range in the cleaned ride data, to execute the ride simulation for every given month. However, the date range can simply be adjusted by allocating DateTime objects to the variables start_date and end_date in lines 30-31:
```
# get date range from orignal data, to execute rides simulation for every given month
start_date = min(rides_df["scheduled_to"])
end_date = max(rides_df["scheduled_to"])
```
Next, the number of rides to be simulated per month is defined in line 39 within the variable month_sim_rides:
```
# simulate rides
month_sim_rides = 5000  # CHANGE to Adjust monthly number of simualted rides
new_rides_all = pd.DataFrame(columns=rides_df.columns)
```
Finally, the script iterates over the months of the defined date range and simulates the defined number of rides per month. Therefore, the script calls the function ***generateRideSpecs()*** that returns a DataFrame with the simulated rides. In this function the cleaned ride data from MoD acts as the basis from which numerous probability distributions are derived to randomly choose attribute values from. In the end, the result is stored in the data folder *data/simulated*: 
```
# save simulated rides as csv
new_rides_all.to_csv(f"{repo}/data/simulated/ride_simulation.csv")
```

### 5.3.2 Main Function: generateRideSpecs()
***
New artificially simulated rides can be generate with the function generateRideSpecs(oldRides, ridestops, routes, n, month, year), where:
- **oldRides** = DataFrame with original rides (basis for probability distributions)
- **ridestops** = DataFrame with all existing MoDStops
- **routes** = DataFrame with all pairwise combinations of MoDStops and their distances
- **n** = number of rides to be simulated
- **month** = month for which rides are to be simulated
- **year** = year for which rides are to be simulated

In an initial step, the function creates an empty DataFrame ‘newRides’ with all needed columns. Then, all needed attributes of that DataFrame are incrementally filled for all n new rides through help functions (see following sections). As a result, we receive a DataFrame with n completed rides in the specified month of the specified year. 

### 5.3.3. Help Functions 
***
1. **General Attributes**
  
    Firstly, artificial ride id’s and user id’s are generated by concatenating the same unix timestamp (time of calling the function ***generateRideSpecs()***) with a consecutive number.
   - id = timestamp + "-" + consecutive number; example: 1662366222-0
   - user_id = consecutive number + "-" + timestamp; example: 0-1662366222

    Secondly, values for all discrete attributes are generated based on probability distributions derived from the frequencies of the characteristics in the original ride data. The function ***generateValues(column_name, df, newRides)*** is used, which derives the probability distribution and returns n random values based on it. The following attributes are captured:
   - number_of_passenger
   - free_ride
   - payment_type
   - arrival_indicator
   - rating

    Additionally, the state column of all rides is populated with the string ‘completed’ because we simulate only completed rides.
   - state = "completed"

2. **Timestamps**

    The timestamps are successively populated along the process of timestamps, starting with **‘created_at’** and ending with ‘dropoff_at’. The function ***generateCreatedAt(oldRides, newRides)*** returns n random ‘created_at’ timestamps over a period of the specified month based on the probability distribution in the original data. In a first step, we choose a date from the month based on the probability distribution of rides over the weekdays (Monday – Sunday). In a second step, we choose a timestamp based on the hourly probability distribution of rides that are on the same weekday of the date chosen in step 1. 
    
    The next timestamp is **‘scheduled_to’**, which is the actual time when the customer wants to be picked up. The function ***generateScheduledTo(oldRides, newRides)*** returns n random ‘scheduled_to’ timestamps based on the probability distribution in the original data. Therefore, we need to consider the distribution of scheduled & immediate rides. Afterwards, we can fill the ‘scheduled_to’ attribute with the ‘created_at’ timestamp for all immediate rides. For all scheduled rides, we add a random prebooking time, which is based on a truncated normal distribution of the prebooking time in original data, to the ‘created_at’ timestamp. The distribution is truncated on the left side at 8 minutes because we assume a prebooked ride must be at least 8 min in the future because the dispatching of a driver is 8 minutes before the ‘scheduled_to’ timestamp. <br>
    Based on the first months of original ride data we assume that there are no rides between 1am and 6:59am. Consequently, we need to check if we randomly created such false ‘scheduled_to’ timestamps that are to be fixed. We can change a fraction of these timestamps to 7:00am in the morning, which is also the case in the original rides. However, we need to ensure that the hourly distribution of the ‘scheduled_to’ timestamp is not overweighted on rides at 7am. Therefore, in case too many rides were falsely simulated in the night we distribute the rest over the other working hours based on a probability distribution. 

    For the timestamp **‘dispatched_at’**, the function ***generateDispatchedAt(oldRides, newRides)*** returns n random values following the logic of:
    - Case 1: if **scheduled** ride then ‘dispatched_at’ = ‘scheduled_at’ - 8 min
    - Case 2: if **immediate** ride then ‘dispatched_at’ = ‘scheduled_at’

    This paragraph is about the simulation of all attributes related to the arrival of the vehicle at the pickup address: **’vehicle_arrived_at’** and **‘pickup_arrival_time’**. The function ***generateArrival(oldRides, newRides)*** returns n random timestamps for these attributes and begins with the ‘vehicle_arrived_at’ timestamp. Again, we need to distinguish between prebooked and immediate rides here because for prebooked rides the driver knows about the upcoming ride 8 minutes before the scheduled pickup time. <br>
    Therefore, we use the following logic:
    - Case 1: if **scheduled** ride then ‘vehicle_arrived_at’ = ‘scheduled_to’ + random scheduling deviation (based on probability distribution of ‘scheduling_deviation’ = ‘vehicle_arrived_at’ – ‘scheduled_to’ in original data
    - Case 2: if **immediate** ride then ‘vehicle_arrived_at’ = ‘scheduled_to’ + random ‘pickup_arrival_time’ (based on probability distribution of attribute ‘pickup_arrival_time’ in original data)
  
    Secondly, this function simply calculates the **‘pickup_arrival_time’** = ‘vehicle_arrived_at’ - ‘dispatched_at’. 

    In the following, we describe the procedure of generating values for attributes related to the pickup: **‘earliest_pickup_expectation’**, **‘pickup_at’**, **‘pickup_eta’**, **‘pickup_first_eta’** and **‘arriving_push’**. The function ***generatePickup(oldRides, newRides)*** returns n random timestamps for all these attributes and starts with a simple calculation of ‘earliest_pickup_expectation’ = ‘dispatched_at’ + 3 minutes.<br>
    Afterwards, the ‘pickup_at’ timestamp is simulated:
    - ‘pickup_at’ = ‘vehicle_arrived_at’ + random time until pickup (based on probability distribution of ‘time_until_pickup’ = ‘pickup_at’ - ‘vehicle_arrived_at’ in original data but distinguish between probability distributions of scheduled and immediate rides) 

    Third, we want to simulate the two estimated pickup timestamps by adding a random deviation of to the already simulated ‘pickup_at’ timestamp:
    - ‘pickup_first_eta’ = ‘pickup_at’ + random deviation of ‘pickup_first_eta’ (based on probability distribution of ‘deviation_of_pickup_first_eta’ = ‘pickup_first_eta’ – ‘pickup_at’ in original data)
    - ‘pickup_eta’ = ‘pickup_at’ + random deviation of ‘pickup_eta’ (based on probability distribution of ‘deviation_of_pickup_eta’ = ‘pickup_eta’ – ‘pickup_at’ in original data)
 
    The last attribute covered in this function is ‘arriving_push’. For all simulated rides we calculate ‘arriving_push’ = ‘vehicle_arrived_at’ + random deviation of ‘arriving_push’ (based on probability distribution of ‘deviation_of_arriving_push’ = ‘vehicle_arrived_at’ – ‘arriving_push’ in original data).

    The last timestamps that are to be simulated are the **drop off related attributes**. The function ***generateDropoff(oldRides, newRides, routes)***, where routes is the DataFrame with all pairwise combinations of MoDStops and their distances, returns n random timestamps for **‘dropoff_at’**, **‘dropoff_eta’** and **‘dropoff_first_eta’**. <br>
    First, we need to generate a ride_time for each simulated ride that represents the time between the pickup and the drop off a passenger. Here, we need to consider that the ride time strongly depends on the driven route (different distances) and the time (e.g. rush hours). Therefore, for every simulated ride we filtered the original ride data first to find the most similar original rides. Afterwards, we took the average ‘ride_time’ of these most similar rides and added up to +/- 10% randomness. And finally, we can calculate the ‘dropoff_at’ = ‘pickup_at’ + simulated ‘ride_time’. <br>
    To identify the most similar original rides of a simulated one, we applied **cascading conditions** to filter the original ride data:
    1. Rides exist with same route & same workday/weekend flag & at the same time (timeframe of +/-1 hour allowed)
    2. Rides exist with same route & at the same time (timeframe of +/-1 hour allowed) - workday/weekend does not matter
    3. Rides exist with same route - day & hour does not matter
    4. Else, i.e. the route was never driven before, we use ‘shortest_ridetime’ (assuming a speed of 30km/h over the distance of the route) 
   
    Subsequently, we can simply calculate the ‘dropoff_first_eta’ = ‘pickup_first_eta’ + ‘shortest_ridetime’. And in the end, the estimated drop off time can be calculated with a random deviation around the actual simulated ‘dropoff_at’ timestamp: ‘dropoff_eta’: ‘dropoff_at’ + random deviation of ‘dropoff_eta’ (based on probability distribution of ‘deviation_of_dropoff_eta’ = ‘dropoff_eta’ – ‘dropoff_at’ in original data)

3. **Route Characteristics**

    The characteristics of the route (combination of pickup address and drop off address) must be simulated after the ‘scheduled_to’ timestamp and before the ‘dropoff_at’ timestamp because the scheduled time is needed to choose a route for a simulated ride and to calculate the drop off timestamp, we need to know the driven route. The route related attributes are the **‘pickup_address’**, **‘dropoff_address’**, **‘distance’** and **‘shortest_ridetime’**, which can be simulated with the function ***generateRoute(oldRides, newRides, ridestops, routes)***, where ‘ridestops’ is a DataFrame with all existing MoDStops. The first step is to randomly choose a route for all simulated rides based on a probability distribution over the routes in the original data. However, the choice of the route strongly depends on the time of a ride, i.e. the hour of the day as well as whether it is a workday or a weekend day. Consequently, for all new rides we need to filter the original rides first by the time. We allow here original rides in a timeframe of +/- 1 hour around the time of a simulated ride. However, we can not only base the route choice on the past because then we would never simulate routes that were never driven before. As a consequence, we assume that 20% out of the number of simulated rides should take place on randomly added new (never driven before) routes. Therefore, an even number of new routes is added to the probability distribution of the routes that are weighted similar as the least frequently driven route in the filtered original rides because we assume that this weighting represents the fraction of completely random routes. Afterwards, we can derive a probability distribution for a simulated ride and randomly choose a route.
    After simulating the routes for all new rides, we can simply look up the ‘distance’ values in the DataFrame containing all existing MoDStops with their distances. Furthermore, we can now calculate the attribute ‘shortest_ridetime’ by assuming an average speed of 30km/h over the distance of the route (logic by MoD). 

4. **Timespans and other KPI's**
    
    The last group of attributes consist of several KPI’s, e.g. time spans between timestamps. Consequently, these attributes can all be simply calculated with simple arithmetic operations. The function ***generateTimeperiods(newRides)*** calculates and returns all of the following attributes for all simulated rides: 
   - **‘arrival_deviation’** = ‘vehicle_arrived_at’ – ‘arriving_push’ - 3 min.
   - **‘waiting_time’** = ‘vehicle_arrived_at’ – ‘earliest_pickup_expectation’
   - **‘boarding_time’** = ‘pickup_at’ – ‘vehicle_arrived_at’
   - **‘ride_time’** = ‘dropoff_at’ – ‘pickup_at’
   - **‘trip_time’** = ‘ride_time’ + ‘waiting_time’
   - **‘delay’** = ‘trip_time’ – ‘shortest_ridetime’
   - **‘longer_route_factor’** = ‘ride_time’ / ‘shortest_ridetime’

## 5.4 Probablistic Graph Model 
***

## 5.5 Feasibilty Analysis 
***
The Feasibility Analysis is based on the idea to generate the graph metrics **diameter/longest shortest past**  and **average shortest path** for an increasing number of rides and to use these metrics as an input for a regression analysis. In the end the regressed output function should return a number of simulated rides for a certain entered day threshold for a given metric (["diameter_w/o_drones", "avg_w/o_drones", "diameter_with_drones", "avg_with_drones"]).

For the Feasibility Analysis two steps are required:

1. Generate the Regression Metrics based on simulated rides data (Original rides data is currently not enough to solve the problem).
Execute the script **vehicle_stream_pipeline/metrics_for_regression_execution.py**.
This script calls the function ***getRegressionMetrics()***, which samples rides for an increasing stepsize, transforms the data samples to a weighted directed graph and calculates the graph metrics for it. The Output is the following:

| #_simulated_rides      | diameter_w/o_drones        | avg_w/o_drones      | diameter_with_drones        | avg_with_drones        |
| ------------- | ------------- | ------------- | ------------- | ------------- |
|10000|...|...|...|...|  
|25000|...|...|...|...| 
|...|...|...|...|...| 
|len(input_df)|...|...|...|...| 

The output is stored in the folder *data/regression*.

2. Perform a Regression on the generated regression metrics - to get the optimized parameters for the regression function.
The regression outcome can be produced and used in several ways:
    - ***get_rides_num(max_days, graph_metrics_df, metric="avg_w/o_drones")*** calculates the number of needed rides for a given max_days threshold and a given metric directly and performs the regression within the function via scipy curve_fit.
    - ***get_opt_parameter(graph_metrics_df, metric="avg_w/o_drones")*** outputs the optimization parameters for a given metric, which can be used as an input for the regression function.
    - ***regression_function(x, a, b, c)*** takes as input a x value which corresponds to a day threshold. Can be also a list of threshold values to plot a regression curve. The arguments a, b, c are parameters which where optimized via scipy curve_fit and define the the regression funciton distincly.
	
The regression function used is an exponential decay function in the following format 'a * np.exp(-b / x) + c'.

All these functions are used to plot the regression curve and state the required number of rides whithin the dashboard file *vehicle_stream_pipeline/dashboard/pages/ride_simulation.py* and the eda jupyter notebook *vehicle_stream_pipeline/eda_for_presentation.ipynb*.

**HAVE FUN!**
