import pandas as pd
from functions_detect_inconsistencies import Detect
from numpy import NaN

# read rides_combined.csv - change to your loaction
df = pd.read_csv(
    "/Users/ericchittka/rides_combined.csv"
)

# check the types of the dataframe columns 
df.dtypes

# df = pd.read_csv('../data/rides_combined.csv')

# add empty columns {"Errorcode": "0", "Errormessage": ""}
df["Errorcode"] = "0"
df["Errormessage"] = ""

# check every row for inconsistencies and right errorlist into df
for index, row in df.iterrows():

    # infinite loop protection - can be commented out later
    expression = index > 5
    if expression:
        print(expression)
        break

    # create empty errorlist per row
    errorlist = pd.Series({"Errorcode": "0", "Errormessage": ""})

    # use check function depending on status
    current_status = row["state"]
    if current_status == "completed":
        errorlist = Detect.check_state_completed(row, errorlist)
        errorlist = Detect.check_timestamp_order(row, errorlist)
    elif current_status == "offer" or current_status == "offer-rejected":
        errorlist = Detect.check_state_offer(row, errorlist)
    elif current_status == "canceled":
        errorlist = Detect.check_state_canceled(row, errorlist)

    # update df columns Errorcode and Erromessage at index
    df.loc[index, ["Errorcode", "Errormessage"]] = errorlist


# filter output_df to error rows only (errorcode !=  0)
# filter also all unused columns here
filtered_df = df[df["Errorcode"] != "0"]

#print(df)
print(filtered_df)
