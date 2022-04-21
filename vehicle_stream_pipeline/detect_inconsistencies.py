import pandas as pd
from numpy import NaN


# check functions should be later transported to seperate .py file
def check_state_completed(row, errorlist):

    # check price offe and pickup at
    error_expression = (row["price_offer"] != NaN) and (row["pickup_at"] == NaN)
    if error_expression:
        if errorlist["Errorcode"] == "0":
            errorlist["Errorcode"] = "1"
        else:
            errorlist["Errorcode"] = errorlist["Errorcode"] + "; " + "1"
        errorlist["Errormessage"] = (
            errorlist["Errormessage"] + "; Price_offer, but no pickup_at"
        )

    # check pickup and dropoff address
    error_expression = row["pickup_address"] == row["dropoff_address"]
    if error_expression:
        if errorlist["Errorcode"] == "0":
            errorlist["Errorcode"] = "2"
        else:
            errorlist["Errorcode"] = errorlist["Errorcode"] + "; " + "2"
        errorlist["Errormessage"] = (
            errorlist["Errormessage"]
            + "; Pickup address equals dropoff address in completed ride"
        )

    # check pickup and dropoff address

    # return final errorlist
    return errorlist


def check_state_offer(row, errorlist):

    # check price offe and pickup at

    # return final errorlist
    return errorlist


def check_state_canceled(row, errorlist):

    # check presence of all timestamps

    # return final errorlist
    return errorlist


def check_timestamp_order(row, errorlist):

    # check presence of all timestamps
    error_expression = (
        row["created_at"] != NaN
        and row["dispatched_at"] != NaN
        and row["pickup_arrival_time"] != NaN
        and row["arriving_push"] != NaN
        and row["vehicle_arrived_at"] != NaN
        and row["earliest_pickup_expectation"] != NaN
        and row["pickup_first_eta"] != NaN
        and row["pickup_eta"] != NaN
        and row["pickup_at"] != NaN
        and row["dropoff_first_eta"] != NaN
        and row["dropoff_eta"] != NaN
        and row["dropoff_at"] != NaN
        and row["updated_at"] != NaN
    )
    if error_expression:
        if errorlist["Errorcode"] == "0":
            errorlist["Errorcode"] = "11"
        else:
            errorlist["Errorcode"] = errorlist["Errorcode"] + "; " + "11"
        errorlist["Errormessage"] = (
            errorlist["Errormessage"] + "; Not all timestamps set"
        )

    # return final errorlist
    return errorlist


def check_timestamp_calculations(row, errorlist):

    # return final errorlist
    return errorlist


df = pd.read_csv(
    "C:/Users/jostm7/Desktop/Uni Mannheim/2. Semester/Teamprojekt/vehicle-stream-team-project/data/rides_combined.csv"
)
# df = pd.read_csv('../data/rides_combined.csv')

df.info()
print(df)

# create empty error_df - later used to attach to row - errocode 0 can be filtered out as last step
error_df = pd.DataFrame()


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
        errorlist = check_state_completed(row, errorlist)
        errorlist = check_timestamp_order(row, errorlist)
    elif current_status == "offer" or current_status == "offer-rejected":
        errorlist = check_state_offer(row, errorlist)
    elif current_status == "canceled":
        errorlist = check_state_canceled(row, errorlist)

    # add current row +  errorlist to output_df
    error_row = pd.concat(objs=[row, errorlist], axis=0)
    print(error_row)
    # error_df = error_df.append(error_row)
    error_df = pd.concat(objs=[error_df, error_row], axis=1)
    print(error_df)

# filter output_df to error rows only (errorcode !=  0)
error_df = error_df.transpose()
filtered_error_df = error_df[error_df["Errorcode"] != "0"]

print(df)
print(error_df)
print(filtered_error_df)
