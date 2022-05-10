# in case of authentication error: pip install mysql-connector-python
# VPN Connection needs to established first
from datetime import datetime

import MySQLdb as my
import pandas as pd


class Database:
    # open connection to remote MySQL DB
    # cnx = my.connect(
    #     user="remote_user",
    #     password="TeamProject1",
    #     host="134.155.95.141",
    #     port=3306,
    #     database="teamproject",
    # )

    # connection can't be built up in class if pd.read_sql is used
    local_cnx_parameters = ("localhost", "root", "TeamProject1", "teamproject", 3306)
    cnx_parameters = (
        "134.155.95.141",
        "remote_user",
        "TeamProject1",
        "teamproject",
        3306,
    )

    # function for read_querys
    def query_dataframe(table, query="SELECT * FROM", local=False):

        # set up connection based on local or remote connection
        if local:
            cnx = my.connect(*Database.local_cnx_parameters)
        else:
            cnx = my.connect(*Database.cnx_parameters)

        # queries
        if query == "SELECT * FROM":
            query = f"{query} {table}"

        # alternative way with cursor.execute - works with cnx = Database.cnx - BUT non column names are read into df
        # cursor = cnx.cursor()
        # cursor.execute(query)
        # query_df = pd.DataFrame(cursor.fetchall())

        # create dataframes from query results
        query_df = pd.read_sql(query, cnx)

        # close connection to remote MySQL DB
        cnx.close()

        return query_df

    # function for insertion_querys
    def insert_dataframe(df, table, local=False):

        # set up connection based on local or remote connection
        if local:
            cnx = my.connect(*Database.local_cnx_parameters)
        else:
            cnx = my.connect(*Database.cnx_parameters)
        cursor = cnx.cursor()

        # add column 'db_updated_at'
        date_time = datetime.now()
        df["db_updated_at"] = date_time
        print(f"Added column 'db_updated_at' {date_time}")

        # drop columns in input df which dont exist in db
        query = f"DESCRIBE {table}"
        cursor.execute(query)
        describe_df = pd.DataFrame(cursor.fetchall())

        column_df = pd.DataFrame(df.columns, columns=["column_names"])
        column_df["column_check"] = column_df.column_names.isin(describe_df[0])
        check_column_df = column_df[column_df["column_check"] == False]

        if ~check_column_df.empty:
            drop_columns_list = check_column_df["column_names"].values.tolist()
            df.drop(labels=drop_columns_list, axis=1, inplace=True)
            print(f"Dropped columns {drop_columns_list} from input df")

        # check for NaN and replace by None
        if df.isnull().values.any():
            df = df.astype(object).where(pd.notnull(df), None)
            print("Replaced NaN by None")

        # df needs to be transformed into list of tuples for executemany
        df_list = df.values.tolist()

        # create query dynamically from column names of input df
        database_attributes = str(tuple(df.columns)).replace("'", "")
        value_placeholder = str(("%s",) * len(df.columns)).replace("'", "")
        query = f"INSERT INTO {table} {database_attributes} VALUES {value_placeholder}"

        # commit INSERT Query and close connection
        # executemany https://dev.mysql.com/doc/connector-python/en/connector-python-api-mysqlcursor-executemany.html
        cursor.executemany(query, df_list)
        cnx.commit()
        cnx.close()

        print("Commit Succesful")
