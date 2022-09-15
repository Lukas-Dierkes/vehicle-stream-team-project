## Table of Contents 
- [Table of Contents](#table-of-contents)
- [1. Open-Objective Data Mining of Real World Vehicle Data](#1-open-objective-data-mining-of-real-world-vehicle-data)
- [2. Technologies](#2-technologies)
- [3. Installation](#3-installation)
- [4. Folder Structure](#4-folder-structure)
- [5. How to Use the Project](#5-how-to-use-the-project)
  - [5.1 Folder structure](#51-folder-structure)
  - [5.2 Data Pipeline](#52-data-pipeline)
  - [5.3 Ride Simulation](#53-ride-simulation)
  - [5.4 Probablistic Graph Model](#54-probablistic-graph-model)
  - [5.5 Feasibilty Analysis](#55-feasibilty-analysis)

## 1. Open-Objective Data Mining of Real World Vehicle Data 
 
***
Within the team project of the University of Mannheim, 2 research projects of INES were presented:
- Mobile e-hub, which investigates how drones can use the rides of cars so that ordered packages can be delivered to customers by drones and cars
- HitchHikeBox, investigating how customer ordered packages can be delivered from boxes carried on other rides that are part of the environment

This project is about **analyzing real world data** of a startup (Mobility-on-Demand) to determine the feasibility of using existing mobility data to support Ines' projects. 

## 2. Technologies  
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

## 3. Installation 
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
$ pip install -e .[default]
```

## 4. Folder Structure 
***
This table is used to briefly illustrate the folder structure of the project.  
| Folder        | Description        |
| ------------- | ------------- |
| [/vehicle-stream-team-project/data](https://github.com/Lukas-Dierkes/vehicle-stream-team-project/tree/master/data) | Contains all **external & resulting data** in the subfolders |
| [/vehicle-stream-team-project/vehicle_stream_pipeline](https://github.com/Lukas-Dierkes/vehicle-stream-team-project/tree/master/vehicle_stream_pipeline)  | Contains   |
| [/vehicle-stream-team-project/vehicle_stream_pipeline/dashboard](https://github.com/Lukas-Dierkes/vehicle-stream-team-project/tree/master/vehicle_stream_pipeline/dashboard)  | Contains   |
| [/vehicle-stream-team-project/vehicle_stream_pipeline/other](https://github.com/Lukas-Dierkes/vehicle-stream-team-project/tree/master/vehicle_stream_pipeline/other)  | Contains   |
| [/vehicle-stream-team-project/vehicle_stream_pipeline/utils](https://github.com/Lukas-Dierkes/vehicle-stream-team-project/tree/master/vehicle_stream_pipeline/dashboard)  | Contains   |


All external data will be placed in the folder ***/vehicle-stream-team-project/data***.
## 5. How to Use the Project  
***
In this section, the **individual steps** are shown and gone through using examples so that the **live dashboard** with the various analyses can be executed at the end.

### 5.1 Folder structure
***
First of all, the individual data from MoD must be manually inserted from SharePoint into the folder structure. Since the data is confidential, you need an access permission to this data. 

Using the table, please assign the files to the appropriate folders:
| File          | Folder        |
| ------------- | ------------- |
| Rides_***.xlsx  | /vehicle-stream-team-project/data/normal_rides |
| Content Cell  | Content Cell  |

### 5.2 Data Pipeline  
***

### 5.3 Ride Simulation 
***

### 5.4 Probablistic Graph Model 
***

### 5.5 Feasibilty Analysis 
***


