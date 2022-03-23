import pandas as pd
#in case of authentication error: pip install mysql-connector-python
import mysql.connector


# open connection to remote MySQL DB
cnx = mysql.connector.connect(
    user="root",
    password="TeamProject1",
    host="localhost",
    port="3306",
    database="teamproject",
)


# write locally to MySQL DB

cursor = cnx.cursor()


# data transformations
# needs to be executed in for loop if several data rows are updated 
area = 'insertionTest3'
free_seats = 'insertionTest3'
free_seats_int = 4
capacity = 4.5

data = (
    area,
    free_seats,
    free_seats_int,
    capacity,
)

query = (
    "INSERT INTO testtable " "(area, free_seats, free_seats_int, capacity) " "VALUES (%s, %s, %s , %s)"
)

cursor.execute(query, data)
cnx.commit()