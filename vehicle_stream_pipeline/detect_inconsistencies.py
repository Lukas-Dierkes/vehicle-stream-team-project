import pandas as pd
from numpy import NaN


# check functions should be later transported to seperate .py file
def check_completed(row, errorlist):

    # check price offe and pickup at
    error_expression = (row["price_offer"] != NaN) and (row["pickup_at"] == NaN)
    if ~error_expression:
        if errorlist["Errorcode"] == "0":
            errorlist["Errorcode"] = "1"
        else:
            errorlist["Errorcode"] = errorlist["Errorcode"] + "; " + "1"
        errorlist["Errormessage"] = (
            errorlist["Errormessage"] + "; Price_offer, but no pickup_at"
        )

    # return final errorlist
    return errorlist


def check_timestamp_order(row, errorlist):

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
# create empty errorlist
errorlist = pd.Series({"Errorcode": "0", "Errormessage": ""})

for index, row in df.iterrows():
    # infinite loop protection - can be commented out later
    expression = index > 5
    if expression:
        print(expression)
        break

    # use check function depending on status
    current_status = row["state"]
    if current_status == "completed":
        errorlist = check_completed(row, errorlist)
        errorlist = check_timestamp_order(row, errorlist)
    elif current_status != "completed":
        print("not completed")

    # add current row +  errorlist to output_df
    error_row = pd.concat(objs=[row, errorlist], axis=0)
    print(error_row)
    error_df = error_df.append(error_row)
    # error_df = pd.concat(objs=[error_df, error_row], axis=1)
    print(error_df)

# filter output_df to error rows only (errorcode !=  0)


print(df)
print(error_df.transpose())
