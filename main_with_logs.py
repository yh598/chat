# Databricks notebook source
# MAGIC %md
# MAGIC Package Installer

# COMMAND ----------

# MAGIC %run "./package_installer"

# COMMAND ----------

# MAGIC %md
# MAGIC Package imports

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col
from pyspark.sql.types import StructType, StructField, StringType, FloatType, LongType, DoubleType, IntegerType

import pandas as pd
import json, os, re
from copy import copy
from datetime import datetime, date
from dateutil import parser
import ast
from tqdm import tqdm
import zipfile

from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, MapType , BooleanType, ArrayType, DoubleType
import random
import shutil
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm


# COMMAND ----------

# MAGIC %md
# MAGIC Import Utility

# COMMAND ----------

# MAGIC %run "./utils"

# COMMAND ----------

# MAGIC %md
# MAGIC Import Config

# COMMAND ----------

# MAGIC %run "./config"

# COMMAND ----------

# MAGIC %md
# MAGIC Import Utility - Post Output file processing methods

# COMMAND ----------

# MAGIC %run "./utils_post_output_files"

# COMMAND ----------

# MAGIC %md
# MAGIC Create spark dataframe from working directory [filename, filepath]

# COMMAND ----------

to_process_path

# COMMAND ----------

from pyspark.sql import Row

# Initialize lists to store file paths and names
file_paths = []
file_names = []

# Walk through the directory tree
for root, dirs, files in os.walk(to_process_path):
    for file in files:
        if file.endswith(".json"):
            file_paths.append(os.path.join(root, file))
            file_names.append(file)

# Create a list of Rows
rows = [Row(transcript_path=path, transcript_name=name) for path, name in zip(file_paths, file_names)]

# Create a Spark DataFrame from the list of Rows
spark_df = spark.createDataFrame(rows)
row_count = spark_df.count()
print(f"Number of rows in spark_df: {row_count}")

# COMMAND ----------

if not spark_df.rdd.isEmpty():
    transcript_path = spark_df.select("transcript_path").first()["transcript_path"]
else:
    transcript_path = None

# COMMAND ----------

if spark_df.count() == 0:
    print("No files to process")
else:
    current_date = datetime.now().strftime("%Y-%m-%d")
    df = (spark_df
        .withColumn("start_date", f.lit(current_date))
        .withColumn("retry_count", f.lit(1)))

# COMMAND ----------

# MAGIC %md
# MAGIC Import LLM agent to process transcript

# COMMAND ----------

import sys
import os

sys.path.append(os.path.abspath('.')+"/CII Pipeline")
print(os.path.abspath('.')+"/CII Pipeline")

# COMMAND ----------

# MAGIC %run "./CII Pipeline/generic_pipeline/call_driver_main"

# COMMAND ----------

# MAGIC %md
# MAGIC Defining UDF

# COMMAND ----------

# expected return type from LLM agent
schema = StructType([
    StructField("processed", StringType()),
    StructField("error_message", StringType()),
    StructField("op_file_path", StringType()),
    StructField("processed_date", StringType())
])

# add ec
process_transcript_UDF = udf(process_transcript, returnType = schema )

# COMMAND ----------

df1 = df.withColumn("output",process_transcript_UDF(df.transcript_path))

# COMMAND ----------

print(df1.count())

# COMMAND ----------

df_final = df1.withColumn("status", df1["output"].getItem("processed")) \
              .withColumn("error_message", df1["output"].getItem("error_message")) \
              .withColumn("processed_date", df1["output"].getItem("processed_date"))\
              .withColumn("output_file_path", df1["output"].getItem("op_file_path"))\
              .drop("output")
df_final

# COMMAND ----------

op_file_path

# COMMAND ----------

import time

start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
try:
    #pd_df_final = df_final.toPandas()
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_stage("NLP Processing", start_time, end_time, "Success", "LLM Processing completed successfully.")
except Exception as e:
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_stage("NLP Processing", start_time, end_time, "Failed", str(e))
    print(f"An error occurred: {e}")

# COMMAND ----------

log_file_with_stages = workspace_path + "/" + datetime.now().strftime("%d%m%Y") + "_staged_logging_status.csv"
write_log_to_file(log_file_with_stages)
print(log_entries)

# COMMAND ----------

# MAGIC %md
# MAGIC Update log file and write to output

# COMMAND ----------

# Define the log file path based on the current date
log_file_path = workspace_path + "/" + datetime.now().strftime("%d%m%Y")+ "_log"

# Write df_final to the log file path in overwrite mode using coalesce to reduce the number of output files
df_final.coalesce(1).write.mode("overwrite").parquet(log_file_path)

# COMMAND ----------

# MAGIC %md
# MAGIC Output Processing Path

# COMMAND ----------

op_file_path

# COMMAND ----------

# MAGIC %md
# MAGIC Set the Job Parameter with Output Path

# COMMAND ----------

start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
try:
    dbutils.jobs.taskValues.set("op_file_path", op_file_path)
    print("Job parameter set for op_file_path")
except Exception as e:
    print("Error setting taskValues: " + str(e))

# COMMAND ----------

# MAGIC %md
# MAGIC Dataframe with Output path

# COMMAND ----------

# MAGIC %run "./utils_post_output_files"

# COMMAND ----------

# MAGIC %md
# MAGIC Define the Table Schema

# COMMAND ----------

# MAGIC %run "./expected_delta_table_schema"

# COMMAND ----------

# MAGIC %md
# MAGIC Initialize Spark Session

# COMMAND ----------

initialize_spark("FlattenedJSONSchema")

# COMMAND ----------

# flatten the JSON schema
op_json_df = read_output_json(op_file_path)
flattened_df = flatten_json(op_json_df)
display(flattened_df.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC For Snowflake fields

# COMMAND ----------

try:
    if SNOWFLAKE_CONNECTION == "ON":
        print("Snowflake connection is ON. Adding Member_ID, Tenure, LOB2, Plan and Geo_county and State")
        df_with_snowflake_values = process_member_details_with_snowflake(flattened_df)
    else:
        print("Snowflake connection is OFF. Passing empty values for snowflake fields.")
        df_with_snowflake_values = flattened_df  
except Exception as e:
    print(f"An error occurred: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC For Netezza Fields

# COMMAND ----------

try:
    if NETEZZA_CONNECTION == "ON":
        print("Netezza connection is ON. Adding Last_Online and Member_Registered")
        df_with_netezza_values = update_member_info_with_netezza(df_with_snowflake_values)
    else:
        print("Netezza connection is OFF. Adding dummy random values Last_Online and Member_Registered.")
        df_with_netezza_values = add_dummy_values(df_with_snowflake_values)
except Exception as e:
    print(f"An error occurred: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC For NPS

# COMMAND ----------

try:
    if NETEZZA_CONNECTION == "ON":
        print("Netezza connection is ON. Adding NPS")
        df_with_nps = update_member_info_with_netezza_nps(df_with_netezza_values)
    else:
        print("Netezza connection is OFF. Adding dummy random values Last_Online and Member_Registered.")
        df_with_nps = add_dummy_values(df_with_snowflake_values)
except Exception as e:
    print(f"An error occurred: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC For Verint Fields

# COMMAND ----------

try:
    if VERINT_CONNECTION == "ON":
        print("Verint connection is ON")
        df_with_verint_values = update_call_satisfaction_score(df_with_nps)
    else:
        print("Verint connection is OFF. Adding (Not Available) for NPS and CSAT.")
        df_with_verint_values = add_dummy_for_verint(df_with_nps)
except Exception as e:
    print(f"An error occurred: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC Update Tenure

# COMMAND ----------

try:
    df_with_tenure = update_tenure_spark(df_with_verint_values)
except Exception as e:
    print(f"An error occurred: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC Skill Names Update

# COMMAND ----------

excel_for_verint = './ALL_LOB_Skill_IDS.xlsx'

try:
    df_with_skill_name = update_skill_names_from_verint(df_with_tenure, excel_for_verint)
    display(df_with_skill_name)
except Exception as e:
    print(f"An error occurred: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC For IFP_Metals

# COMMAND ----------

try:
    df_with_metals=update_ifp_metal_column(df_with_skill_name)
    display(df_with_metals)
except Exception as e:
    print(f"An error occurred: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC Values Sanity Check (Not NULL and NOT Empty)

# COMMAND ----------

try:
    final_df=update_columns_if_any_nulls(df_with_metals)
    display(final_df)
except Exception as e:
    print(f"An error occurred: {e}")

# COMMAND ----------

end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
log_stage("Additional data points processing", start_time,end_time, "Success", "Processing additional data points completed.")

# COMMAND ----------

print(log_entries)
write_log_to_file(log_file_with_stages)

# COMMAND ----------

# MAGIC %md
# MAGIC Update Delta Table

# COMMAND ----------

TABLE_NAME

# COMMAND ----------

start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
try:
    update_delta_table(final_df, CATALOG_NAME, SCHEMA_NAME, TABLE_NAME)
    end_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_stage("Update Delta Table", start_time, end_time, "Success", "Updating Delta table completed successfully.")
except Exception as e:
    end_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_stage("Update Delta Table", start_time, end_time, "Failed", str(e))
    print(f"An error occurred: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC Update log file for Job Stages

# COMMAND ----------

write_log_to_file(log_file_with_stages)
print(log_entries)

# COMMAND ----------

# MAGIC %md
# MAGIC Set log file path Job parameter

# COMMAND ----------

try:
    dbutils.jobs.taskValues.set("log_file_path", log_file_with_stages)
    print("Job parameter set for log file path")
except Exception as e:
    print("Error setting taskValues: " + str(e))

# COMMAND ----------

# MAGIC %md
# MAGIC ETL LOG Table Updates

# COMMAND ----------

def update_llm_processed_count(delta_table_name, ETL_LOG_TABLE):
    try:
        # Load the metadata table into a DataFrame
        df = spark.read.format("delta").table(delta_table_name)
        print(f"Loading the delta table {delta_table_name}")

        # Check if the column exists in the DataFrame
        if 'CONVERSATION_PROCESSED_DATE' not in df.columns:
            raise ValueError("Column 'CONVERSATION_PROCESSED_DATE' does not exist in the table")
        print("Column 'CONVERSATION_PROCESSED_DATE' exists in the table")

        # Register the DataFrame as a temporary view
        df.createOrReplaceTempView("metadata_table")

        # Get the counts for each condition using SQL queries
        sagility_count = spark.sql("""
            SELECT COUNT(DISTINCT id) AS count
            FROM metadata_table
            WHERE org_id = 'Sagility'
              AND TO_DATE(CONVERSATION_PROCESSED_DATE, 'yyyy-MM-dd') = CURRENT_DATE()
        """).collect()[0]['count']

        verint_count = spark.sql("""
            SELECT COUNT(DISTINCT id) AS count
            FROM metadata_table
            WHERE org_id != 'Sagility'
              AND TO_DATE(CONVERSATION_PROCESSED_DATE, 'yyyy-MM-dd') = CURRENT_DATE()
        """).collect()[0]['count']

        print("Calculating the counts for each condition")
        print("Sagility count: " + str(sagility_count))
        print("Verint count: " + str(verint_count))

        # Load the Daily_Transcript_Logs_v3 Delta table
        daily_transcript_logs = spark.read.format("delta").table(ETL_LOG_TABLE)

        # Register the DataFrame as a temporary view
        daily_transcript_logs.createOrReplaceTempView("daily_transcript_logs")

        # Update the Delta table with the new counts using SQL queries
        spark.sql(f"""
            MERGE INTO daily_transcript_logs AS target
            USING (
                SELECT
                    LOB_Information,
                    Date_Created,
                    Vendor_Information,
                    CASE
                        WHEN LOB_Information = 'Core Premier' AND Vendor_Information = 'Verint-BSC' AND TO_DATE(Date_Created, 'yyyy-MM-dd') = CURRENT_DATE() THEN '{verint_count}'
                        WHEN LOB_Information = 'Core Premier' AND Vendor_Information = 'Sagility' AND TO_DATE(Date_Created, 'yyyy-MM-dd') = CURRENT_DATE() THEN '{sagility_count}'
                        ELSE LLM_Processed_Transcript_Count
                    END AS LLM_Processed_Transcript_Count
                FROM daily_transcript_logs
            ) AS source
            ON target.LOB_Information = source.LOB_Information
               AND target.Date_Created = source.Date_Created
               AND target.Vendor_Information = source.Vendor_Information
            WHEN MATCHED THEN
                UPDATE SET target.LLM_Processed_Transcript_Count = source.LLM_Processed_Transcript_Count
        """)

        print(f"LLM_Processed_Transcript_Count updated with counts: Sagility={sagility_count}, Verint={verint_count}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function with the required parameters
update_llm_processed_count(CATALOG_NAME + "." + SCHEMA_NAME + "." + TABLE_NAME, CATALOG_NAME + "." + SCHEMA_NAME + "." + ETL_LOG_TABLE_NAME)

# COMMAND ----------

import pandas as pd
from datetime import datetime

def update_processing_times(delta_table_name):
    try:
        # Load the log file
        log_file_with_stages = workspace_path + "/" + datetime.now().strftime("%d%m%Y") + "_staged_logging_status.csv"
        log_df = pd.read_csv(log_file_with_stages)

        # Extract relevant times from the log file
        llm_processing_start_time = log_df[log_df['Stage'] == 'NLP Processing']['Start Time'].values[0]
        llm_processing_end_time = log_df[log_df['Stage'] == 'NLP Processing']['End Time'].values[0]
        data_points_processing_start_time = log_df[log_df['Stage'] == 'Additional data points processing']['Start Time'].values[0]
        data_points_processing_end_time = log_df[log_df['Stage'] == 'Additional data points processing']['End Time'].values[0]

        # Load the Delta table into a DataFrame
        df = spark.read.format("delta").table(delta_table_name)
        df.createOrReplaceTempView("delta_table")

        # Check if columns exist and add them if they do not
        columns_to_add = [
            ("LLM_Processing_Start_Time", "STRING"),
            ("LLM_Processing_End_Time", "STRING"),
            ("Data_Points_Processing_Start_Time", "STRING"),
            ("Data_Points_Processing_End_Time", "STRING")
        ]

        for column_name, column_type in columns_to_add:
            if column_name not in df.columns:
                spark.sql(f"""
                    ALTER TABLE {delta_table_name}
                    ADD COLUMN {column_name} {column_type}
                """)

        # Update the Delta table with the new times
        spark.sql(f"""
            MERGE INTO {delta_table_name} AS target
            USING (
                SELECT
                    LOB_Information,
                    Date_Created,
                    Vendor_Information,
                    '{llm_processing_start_time}' AS LLM_Processing_Start_Time,
                    '{llm_processing_end_time}' AS LLM_Processing_End_Time,
                    '{data_points_processing_start_time}' AS Data_Points_Processing_Start_Time,
                    '{data_points_processing_end_time}' AS Data_Points_Processing_End_Time
                FROM delta_table
                WHERE Date_Created = CURRENT_DATE()
                  AND LOB_Information = 'Core Premier'
            ) AS source
            ON target.LOB_Information = source.LOB_Information
               AND target.Date_Created = source.Date_Created
               AND target.Vendor_Information = source.Vendor_Information
            WHEN MATCHED THEN
                UPDATE SET
                    target.LLM_Processing_Start_Time = source.LLM_Processing_Start_Time,
                    target.LLM_Processing_End_Time = source.LLM_Processing_End_Time,
                    target.Data_Points_Processing_Start_Time = source.Data_Points_Processing_Start_Time,
                    target.Data_Points_Processing_End_Time = source.Data_Points_Processing_End_Time
            WHEN NOT MATCHED THEN
                INSERT (LOB_Information, Date_Created, Vendor_Information,LLM_Processing_Start_Time, LLM_Processing_End_Time, Data_Points_Processing_Start_Time, Data_Points_Processing_End_Time)
                VALUES (source.LOB_Information, source.Date_Created, source.Vendor_Information,source.LLM_Processing_Start_Time, source.LLM_Processing_End_Time, source.Data_Points_Processing_Start_Time, source.Data_Points_Processing_End_Time)
        """)

        print("Processing times updated successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function with the required parameters
update_processing_times(CATALOG_NAME + "." + SCHEMA_NAME + "." + ETL_LOG_TABLE_NAME)