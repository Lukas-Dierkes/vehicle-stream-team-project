import pandas as pd
import numpy as np
from datetime import datetime as dt
from numpy import NaN

class Detect:

    # check current expression and overwrite errorlist with given error_message - only used within class
    def check_expression(error_code, error_message, error_expression, errorlist):
        if error_expression:
            if errorlist["Errorcode"] == "0":
                errorlist["Errorcode"] = error_code
            else:
                errorlist["Errorcode"] = errorlist["Errorcode"] + "; " + error_code
            errorlist["Errormessage"] = errorlist["Errormessage"] + error_message
        return errorlist

    def check_state_completed(row, errorlist):
        # check price offer and pickup at
        error_code = "1"
        error_message = "Price_offer, but no pickup_at; "
        error_expression = (row["price_offer"] != NaN) and (row["pickup_at"] == NaN)

        errorlist = Detect.check_expression(
            error_code, error_message, error_expression, errorlist
        )

        # check pickup and dropoff address
        error_code = "2"
        error_message = "Pickup address equals dropoff address in completed ride; "
        error_expression = row["pickup_address"] == row["dropoff_address"]

        errorlist = Detect.check_expression(
            error_code, error_message, error_expression, errorlist
        )

        # check that vehicle_arrived_at is filled 
        error_code = "3"
        error_message = "Drive is completed, but there is not vehicle_arrived_at time filled; "
        error_expression = row["vehicle_arrived_at"] == NaN 

        errorlist = Detect.check_expression(
            error_code, error_message, error_expression, errorlist
        )

        # return final errorlist
        return errorlist

    def check_state_offer(row, errorlist):

        # check if arrived_at is filled 
        error_code = "11"
        error_message = "Although the ride wasn't completed, there is a time in arrived_at "
        error_expression = row["vehicle_arrived_at"] != NaN

        errorlist = Detect.check_expression(
            error_code, error_message, error_expression, errorlist
        )

        # return final errorlist
        return errorlist

    def check_state_canceled(row, errorlist):

        # check presence of all timestamps

        # return final errorlist
        return errorlist

    def check_timestamp_order(row, errorlist):

        # check presence of all timestamps
        error_code = "101"
        error_message = "Not all timestamps set; "
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

        errorlist = Detect.check_expression(
            error_code, error_message, error_expression, errorlist
        )

        # check if times are all after the creation time 
        error_code = "102"
        error_message = "Created is after any other time; "
        error_expression = row["created_at"] < row['scheduled_to'] or row['dispatched_at'] or row['arriving_push'] or row['vehicle_arrived_at'] or row['earliest_pickup_expectation'] or row['pickup_first_eta'] or row['pickup_eta'] or row['pickup_at'] or row['dropoff_first_eta'] or row['dropoff_eta'] or row['dropff_at'] or row['updated_at']
        
        errorlist = Detect.check_expression(
            error_code, error_message, error_expression, errorlist
        )

        # return final errorlist
        return errorlist

    def check_timestamp_calculations(row, errorlist):

        # check the dispatched_at calculation
        error_code = "201"
        error_message = "Dispatched_at is not 8 Minutes before scheduled_to; "
        error_expression = (row["scheduled_to"] != NaN) and (row["dispatched_at"] + pd.Timedelta(minutes = 8) != row["scheduled_to"]) 
        
        errorlist = Detect.check_expression(
            error_code, error_message, error_expression, errorlist
        )

        # return final errorlist
        return errorlist
