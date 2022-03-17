# in case of authentication error: pip install mysql-connector-python
import mysql.connector
import pandas as pd

# open connection to remote MySQL DB
teamproject_sql = mysql.connector.connect(
    user="remote_user",
    password="TeamProject1",
    host="134.155.95.141",
    port="3306",
    database="teamproject",
)

# queries
query = "SELECT * FROM testtable "

# create dataframes from query results
test_data = pd.read_sql(query, teamproject_sql)

print(test_data)

# close connection to remote MySQL DB
teamproject_sql.close()
