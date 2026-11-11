#The goal is to create a db

import sqlite3
import pandas as pd


#Create a function to insert data into the database
def insert_into_db(csv_file, db, table):
    conn = sqlite3.connect(db)

    df = pd.read_csv(csv_file)

    df.to_sql(table, conn, if_exists='replace')

    conn.close()


conn = sqlite3.connect("documents/SupremeCourt.db")
cursor = conn.cursor()

cursor.execute("""
 CREATE TABLE IF NOT EXISTS Patient (
     patientID INTEGER PRIMARY KEY,
     firstName TEXT,
     lastName TEXT,
 );
 """)

cursor.execute("""
 CREATE TABLE IF NOT EXISTS Doctor (
     doctorID INTEGER PRIMARY KEY,
     firstName TEXT,
     lastName TEXT,
     speciality TEXT
 );
 """)
conn.commit()


conn.commit()

conn.close()