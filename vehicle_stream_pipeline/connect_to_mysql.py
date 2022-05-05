# in case of authentication error: pip install mysql-connector-python
import MySQLdb as my
import pandas as pd


class Database:
    # open connection to remote MySQL DB
    cnx = my.connect(
        user="remote_user",
        password="TeamProject1",
        host="134.155.95.141",
        port=3306,
        database="teamproject",
    )

    # parameter - custom attributes list, custom query
    def query_dataframe(table, query="SELECT * FROM ", table_attributes="*"):

        cnx = Database.cnx
        # queries
        query = "SELECT * FROM testtable "

        # create dataframes from query results
        query_df = pd.read_sql(query, cnx)
        # close connection to remote MySQL DB
        cnx.close()

        return query_df

    def insert_dataframe(df, table):
        cnx = Database.cnx
        cursor = cnx.cursor()

        df_list = df.values.tolist()

        # create query dynamically from columns names
        # REQUIRES: df.columns must be dropped upfrond so that df.columnnames have zhe same names as table column names
        database_attributes = str(tuple(df.columns)).replace("'", "")
        value_placeholder = str(("%s",) * len(df.columns)).replace("'", "")
        query = f"INSERT INTO {table} {database_attributes} VALUES {value_placeholder}"

        # executemany https://dev.mysql.com/doc/connector-python/en/connector-python-api-mysqlcursor-executemany.html
        cursor.executemany(query, df_list)
        cnx.commit()
