# Databricks notebook source
# MAGIC %md
# MAGIC ##### Config-Setup for Sagility

# COMMAND ----------

!pip install openpyxl
!pip install aiofiles
dbutils.library.restartPython() 

# COMMAND ----------

from datetime import datetime

processing_start_time_sagility = datetime.now()

# COMMAND ----------

source_directory_sagility = "/Volumes/prod_adb/default/synapse_volume/raw/surveyspeechextraction/Sagility/"
# source_directory_sagility = "/Volumes/dbc_adv_anlaytics_dev/surveyspeechextraction/call_driver_poc/common_filter_testing/source_file_sagility/"
target_directory_sagility ="/Volumes/dbc_adv_anlaytics_dev/surveyspeechextraction/call_intent_workspace/common_filter/"


import os
if not os.path.exists(target_directory_sagility):
    os.makedirs(target_directory_sagility)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Current Date and time

# COMMAND ----------

from datetime import datetime, timedelta
import os
def get_current_date_and_time():
        # current_date = datetime.now()-timedelta(days=6)
        current_date = datetime.now()-timedelta(days=5)
        # Format the date in "dd/mm/yyyy" format
        formatted_date = current_date.strftime("%Y-%m-%d")
        formatted_time = current_date.time().strftime("%H:%M:%S")
        return formatted_date, formatted_time

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Setup Workspaces

# COMMAND ----------


# Setting up workspace folders
folder_to_execute = get_current_date_and_time()[0]
workspace_path = target_directory_sagility

# Define the main folders
folders = ["IFP", "Provider","Core Premier"]

# Create paths for each main folder and their subfolders
paths = {}
for folder in folders:
    paths[folder] = {
        "to_process_path": os.path.join(workspace_path, folder, folder_to_execute, "to_process", "Sagility"),
        "op_file_path": os.path.join(workspace_path, folder, folder_to_execute, "output", "Sagility"),
        "archive_path": os.path.join(workspace_path, folder, folder_to_execute, "archive", "Sagility"),
        "log_file_path": os.path.join(workspace_path, folder, folder_to_execute, "logs", "Sagility")
    }

# Add paths for raw_unzip_files and Original_zip_files inside other_lobs
raw_unzip_path = os.path.join(workspace_path,"raw_unzip_files",folder_to_execute)
original_zip_files = os.path.join(workspace_path,"original_zip_files" ,folder_to_execute)

# Print paths for each main folder and their subfolders
for folder, subfolders in paths.items():
    print(f'Paths for {folder}:')
    for subfolder_name, subfolder_path in subfolders.items():
        print(f'{subfolder_name}: {subfolder_path}')
    print()

# Print paths for raw_unzip_files and original_zip_files
print(f'raw_unzip_path: {raw_unzip_path}')
print(f'original_zip_files: {original_zip_files}')

# COMMAND ----------


# Create the workspace directory if it doesn't exist
if not os.path.exists(workspace_path):
    os.makedirs(workspace_path)

# Create the subdirectories for each main folder
for folder, subfolders in paths.items():
    for subfolder_name, subfolder_path in subfolders.items():
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

# COMMAND ----------

# Print paths for each main folder and their subfolders
for folder, subfolders in paths.items():
    print(f'Paths for {folder}:')
    for subfolder_name, subfolder_path in subfolders.items():
        print(f'{subfolder_name}: {subfolder_path}')
    print()


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Setup configurations for global filter

# COMMAND ----------

# MAGIC %run "./Unzip_Filter"

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Unzip all transcripts to target path

# COMMAND ----------

huntgroup_excel_path="./ALL_LOB_Skill_IDS.xlsx"

# COMMAND ----------

sagility_unzip_path=unzip_sagility_files(source_directory_sagility, original_zip_files,raw_unzip_path, folder_to_execute)


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Global Common Filter

# COMMAND ----------

try:
    global_filter_sagility(sagility_unzip_path, folder_to_execute, huntgroup_excel_path, paths)
except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------

processing_end_time_sagility = datetime.now()

# COMMAND ----------

CATALOG_NAME='dbc_adv_anlaytics_dev'
SCHEMA_NAME='surveyspeechextraction'
ETL_LOG_TABLE='Daily_Transcript_Logs'

# COMMAND ----------


from pyspark.sql.functions import current_date, lit, col, when, count, sum, to_date

def update_daily_transcript_logs(metadata_table_name, processing_start_time, processing_end_time):
    try:
        # Check existing columns in the table
        existing_columns = spark.sql(f"DESCRIBE {CATALOG_NAME}.{SCHEMA_NAME}.{ETL_LOG_TABLE}").select("col_name").rdd.flatMap(lambda x: x).collect()

        # Columns to be added
        columns_to_add = [
            "LLM_Processing_Start_Time STRING",
            "LLM_Processing_End_Time STRING",
            "Data_Points_Processing_Start_Time STRING",
            "Data_Points_Processing_End_Time STRING"
        ]

        # Add columns if not present
        for column in columns_to_add:
            column_name = column.split()[0]
            if column_name not in existing_columns:
                spark.sql(f"""
                ALTER TABLE {CATALOG_NAME}.{SCHEMA_NAME}.{ETL_LOG_TABLE}
                ADD COLUMNS ({column})
                """)

        # Load the metadata table
        metadata_table = spark.table(metadata_table_name)

        # Filter the metadata_table for the required conditions and today's date
        filtered_metadata = metadata_table.filter(
            (metadata_table.CII_Source.like('%Sagility%')) &
            (to_date(metadata_table.CII_Current_date) == current_date())
        )

        # Check if the necessary columns exist
        columns = metadata_table.columns

        outbound_calls_expr = sum(when(col("Direction") == 2, 1).otherwise(0)).alias("Outbound_Calls_Count") if "Direction" in columns else lit(0).alias("Outbound_Calls_Count")
        inbound_calls_expr = sum(when(col("Direction") == 1, 1).otherwise(0)).alias("Inbound_Calls_Count") if "Direction" in columns else lit(0).alias("Inbound_Calls_Count")
        skill_id_present_expr = sum(when(col("CD10").isNotNull(), 1).otherwise(0)).alias("Skill_ID_Present") if "CD10" in columns else lit(0).alias("Skill_ID_Present")
        queue_name_present_expr = sum(when(col("CD69").isNotNull(), 1).otherwise(0)).alias("Queue_Name_Present") if "CD69" in columns else lit(0).alias("Queue_Name_Present")
        member_id_present_expr = sum(when((col("CD48_Member_ID").isNotNull()) | (col("CD93_Subscriber_ID").isNotNull()) | (col("CD6_Subscriber_ID").isNotNull()), 1).otherwise(0)).alias("Member_ID_Present") if any(col in columns for col in ["CD48_Member_ID", "CD93_Subscriber_ID", "CD6_Subscriber_ID"]) else lit(0).alias("Member_ID_Present")
        length_more_than_60_seconds_expr = sum(when(col("Duration_in_sec") > 60, 1).otherwise(0)).alias("Length_More_Than_60Seconds") if "Duration_in_sec" in columns else lit(0).alias("Length_More_Than_60Seconds")

        # Calculate the counts for each LOB and date
        lob_counts = filtered_metadata.groupBy("CII_Selected_LOB", "CII_Current_date").agg(
            count("*").alias("Total_Transcript_Count"),
            outbound_calls_expr,
            inbound_calls_expr,
            skill_id_present_expr,
            queue_name_present_expr,
            member_id_present_expr,
            length_more_than_60_seconds_expr
        )

        # Add the static columns and cast to match the target schema
        lob_counts = lob_counts.withColumn("Date_Created", current_date()) \
            .withColumn("Global_Filter_Start_Datetime", lit(processing_start_time)) \
            .withColumn("Global_Filter_End_Datetime", lit(processing_end_time)) \
            .withColumn("Vendor_Information", lit("Sagility")) \
            .withColumn("LLM_Processed_Transcript_Count", lit(None).cast("string")) \
            .withColumn("Total_Transcript_Count", col("Total_Transcript_Count").cast("string")) \
            .withColumn("Outbound_Calls_Count", col("Outbound_Calls_Count").cast("string")) \
            .withColumn("Inbound_Calls_Count", col("Inbound_Calls_Count").cast("string")) \
            .withColumn("Skill_ID_Present", col("Skill_ID_Present").cast("string")) \
            .withColumn("Queue_Name_Present", col("Queue_Name_Present").cast("string")) \
            .withColumn("Member_ID_Present", col("Member_ID_Present").cast("string")) \
            .withColumn("Length_More_Than_60Seconds", col("Length_More_Than_60Seconds").cast("string")) \
            .withColumn("LOB_Information", col("CII_Selected_LOB"))

        # Drop the additional columns
        lob_counts = lob_counts.drop("CII_Selected_LOB", "CII_Current_date")
        
        # Write the results to the Delta table with mergeSchema option
        lob_counts.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(f"{CATALOG_NAME}.{SCHEMA_NAME}.{ETL_LOG_TABLE}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function with the required parameters
update_daily_transcript_logs("metadata_table", processing_start_time_sagility, processing_end_time_sagility)