import pandas as pd
from numpy import NaN


class Detect:
    def check_state_completed(row, errorlist):
        # check price offer and pickup at
        error_expression = (row["price_offer"] != NaN) and (row["pickup_at"] == NaN)
        if ~error_expression:
            if errorlist["Errorcode"] == "0":
                errorlist["Errorcode"] = "1"
            else:
                errorlist["Errorcode"] = errorlist["Errorcode"] + "; " + "1"
            errorlist["Errormessage"] = (
                errorlist["Errormessage"] + "Price_offer, but no pickup_at; "
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
                + "Pickup address equals dropoff address in completed ride; "
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
                errorlist["Errormessage"] + "Not all timestamps set; "
            )

        # check order of all timestamps
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
                errorlist["Errorcode"] = "12"
            else:
                errorlist["Errorcode"] = errorlist["Errorcode"] + "; " + "12"
            errorlist["Errormessage"] = (
                errorlist["Errormessage"] + "Not all timestamps are in right order; "
            )

        # return final errorlist
        return errorlist

    def check_timestamp_calculations(row, errorlist):

        # return final errorlist
        return errorlist
