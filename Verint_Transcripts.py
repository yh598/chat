
import os, shutil
from datetime import datetime
from zipfile import ZipFile
from Unzip_Filter import global_filter
import json

# Configuration
source_directory_verint = "/Volumes/prod_adb/default/ext-data-volume-stmlz/surveyspeechextraction/Call_Transcripts/2025_verint_transcripts/"
target_directory_verint = "/Volumes/dbc_adv_anlaytics_dev/surveyspeechextraction/call_intent_workspace/common_filter/"
processed_log_file = os.path.join(target_directory_verint, "processed_folders.txt")

# LOB folders
folders = ["MA", "IFP", "Broker", "Small Group", "MediCal", "Core Premier"]

# Ensure target/log file exists
os.makedirs(target_directory_verint, exist_ok=True)
if not os.path.exists(processed_log_file):
    open(processed_log_file, 'w').close()

# Load processed folders
processed_folders = set()
with open(processed_log_file, 'r') as log:
    for line in log:
        processed_folders.add(line.strip())

# Detect new date folders
all_folders = sorted([f for f in os.listdir(source_directory_verint) if f.startswith("2025-")])
new_folders = [f for f in all_folders if f not in processed_folders]

def timestamp_prefix():
    return datetime.now().strftime('%Y%m%dT%H%M%S%f')[:-3]

processing_start_time = datetime.now()

for folder_to_execute in new_folders:
    print(f"Processing folder: {folder_to_execute}")
    src_path = os.path.join(source_directory_verint, folder_to_execute)
    workspace_path = os.path.join(target_directory_verint, folder_to_execute)

    if os.path.exists(workspace_path):
        shutil.rmtree(workspace_path)
    os.makedirs(workspace_path, exist_ok=True)

    # Create standard paths
    paths = {}
    for folder in folders:
        folder_path = os.path.join(workspace_path, folder, folder_to_execute)
        paths[folder] = {
            "to_process_path": os.path.join(folder_path, "to_process", "Verint"),
            "op_file_path": os.path.join(folder_path, "output", "Verint"),
            "archive_path": os.path.join(folder_path, "archive", "Verint"),
            "log_file_path": os.path.join(folder_path, "logs", "Verint")
        }
        for path in paths[folder].values():
            os.makedirs(path, exist_ok=True)

    # Unzip files
    raw_unzip_path = os.path.join(workspace_path, "raw_unzip_files", folder_to_execute)
    original_zip_path = os.path.join(workspace_path, "original_zip_files", folder_to_execute)
    os.makedirs(raw_unzip_path, exist_ok=True)
    os.makedirs(original_zip_path, exist_ok=True)

    zip_files = [z for z in os.listdir(src_path) if z.endswith(".zip")]
    for zip_file in zip_files:
        try:
            src_zip = os.path.join(src_path, zip_file)
            shutil.copy(src_zip, original_zip_path)
            with ZipFile(src_zip, 'r') as zip_ref:
                zip_ref.extractall(raw_unzip_path)
        except Exception as e:
            print(f"Failed to unzip {zip_file}: {e}")

    try:
        # Run global filter
        excel_path = "./ALL_LOB_Skill_IDS.xlsx"
        global_filter(raw_unzip_path, folder_to_execute, excel_path, paths)

        # Timestamped filenames
        for folder in folders:
            op_path = paths[folder]["op_file_path"]
            for file in os.listdir(op_path):
                if file.endswith(".json"):
                    old_path = os.path.join(op_path, file)
                    new_name = f"{timestamp_prefix()}.json"
                    new_path = os.path.join(op_path, new_name)
                    os.rename(old_path, new_path)

        # Log completion
        with open(processed_log_file, 'a') as log:
            log.write(folder_to_execute + "\n")
        print(f"Completed: {folder_to_execute}")

    except Exception as e:
        print(f"Error processing folder {folder_to_execute}: {e}")

processing_end_time = datetime.now()
print(f"Completed all folders. Total time: {processing_end_time - processing_start_time}")






# Verint Transcript Dynamic Folder Processing Pipeline

import os, shutil
from datetime import datetime
from zipfile import ZipFile

# Step 1: Config and Setup
source_directory_verint = "/Volumes/prod_adb/default/ext-data-volume-stmlz/surveyspeechextraction/Call_Transcripts/2025_verint_transcripts/"
target_directory_verint = "/Volumes/dbc_adv_anlaytics_dev/surveyspeechextraction/call_intent_workspace/common_filter/"

if not os.path.exists(target_directory_verint):
    os.makedirs(target_directory_verint, exist_ok=True)

processed_log_file = os.path.join(target_directory_verint, "processed_folders.txt")
processed_folders = set()
if os.path.exists(processed_log_file):
    with open(processed_log_file, 'r') as log:
        for line in log:
            folder_name = line.strip()
            if folder_name:
                processed_folders.add(folder_name)
else:
    open(processed_log_file, 'w').close()

processing_start_time = datetime.now()
print(f"Initialized processing. Found {len(processed_folders)} folders already processed.")

# Step 2: Identify New Folders
all_folders = sorted([d for d in os.listdir(source_directory_verint) 
                      if d.startswith("2025-") and os.path.isdir(os.path.join(source_directory_verint, d))])
new_folders = [d for d in all_folders if d not in processed_folders]

print(f"Total folders detected: {len(all_folders)}")
print(f"New folders to process: {new_folders}")

# Step 3: Process New Folders
from Unzip_filter import global_filter  # Replace with actual module if needed

for folder in new_folders:
    print(f"\nProcessing folder: {folder}")
    folder_to_execute = folder
    source_folder_path = os.path.join(source_directory_verint, folder_to_execute)
    workspace_path = os.path.join(target_directory_verint, folder_to_execute)
    
    if os.path.exists(workspace_path):
        shutil.rmtree(workspace_path)
    os.makedirs(workspace_path, exist_ok=True)

    categories = ["MA", "IFP", "Broker"]
    paths = {}
    for cat in categories:
        cat_dir = os.path.join(workspace_path, cat)
        paths[cat] = {
            "to_process_path": os.path.join(cat_dir, "to_process"),
            "op_file_path": os.path.join(cat_dir, "op_file"),
            "archive_path": os.path.join(cat_dir, "archive"),
            "log_file_path": os.path.join(cat_dir, "log")
        }
        for sub_path in paths[cat].values():
            os.makedirs(sub_path, exist_ok=True)

    raw_unzip_path = os.path.join(workspace_path, "raw_unzipped")
    original_zip_path = os.path.join(workspace_path, "raw_zips")
    os.makedirs(raw_unzip_path, exist_ok=True)
    os.makedirs(original_zip_path, exist_ok=True)

    zip_files = [f for f in os.listdir(source_folder_path) if f.lower().endswith(".zip")]
    folder_error = False
    for zip_file in zip_files:
        try:
            print(f" Unzipping file: {zip_file}")
            src_zip_path = os.path.join(source_folder_path, zip_file)
            dest_zip_path = os.path.join(original_zip_path, zip_file)
            shutil.copy(src_zip_path, dest_zip_path)
            with ZipFile(src_zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_unzip_path)
        except Exception as e:
            folder_error = True
            print(f"  Failed to process {zip_file}: {e}")

    if folder_error:
        print(f" Skipping global filter for {folder_to_execute} due to extraction errors.")
        continue

    try:
        print(f" Running global_filter for folder {folder_to_execute} ...")
        global_filter(raw_unzip_path, folder_to_execute, paths)
        with open(processed_log_file, 'a') as log:
            log.write(folder_to_execute + "\n")
        processed_folders.add(folder_to_execute)
        print(f" Completed global_filter for folder: {folder_to_execute}")
    except Exception as e:
        print(f" Error during global_filter for {folder_to_execute}: {e}")

# Step 4: Completion
processing_end_time = datetime.now()
elapsed_time = processing_end_time - processing_start_time
print(f"\nProcessed {len(new_folders)} new folder(s). Elapsed time: {elapsed_time}")




# Databricks notebook source
# MAGIC %md
# MAGIC Config setup for Verint

# COMMAND ----------

!pip install openpyxl
!pip install aiofiles
dbutils.library.restartPython() 

# COMMAND ----------

from datetime import datetime

processing_start_time_verint = datetime.now()

# COMMAND ----------

source_directory_verint ="/Volumes/prod_adb/default/ext-data-volume-stmlz/surveyspeechextraction/Call_Transcripts/2025_verint_transcripts/ "

target_directory_verint = "/Volumes/dbc_adv_anlaytics_dev/surveyspeechextraction/call_intent_workspace/common_filter/"

import os
if not os.path.exists(target_directory_verint):
    os.makedirs(target_directory_verint)

# COMMAND ----------

# MAGIC %md
# MAGIC Current Date and time

# COMMAND ----------

from datetime import datetime, timedelta
import os
def get_current_date_and_time():
        current_date = datetime.now()-timedelta(days=4)
        # current_date = datetime.now()-timedelta(days=12)
        # print("Current date and time: ", current_date)
        # Format the date in "dd/mm/yyyy" format
        formatted_date = current_date.strftime("%Y-%m-%d")
        formatted_time = current_date.time().strftime("%H:%M:%S")
        return formatted_date, formatted_time

# COMMAND ----------

# MAGIC %md
# MAGIC Setup Workspaces

# COMMAND ----------


# Setting up workspace folders
folder_to_execute = get_current_date_and_time()[0]
workspace_path = target_directory_verint

# Define the main folders
folders = ["MA", "IFP", "Broker","Small Group","MediCal","Core Premier"]

# Create paths for each main folder and their subfolders
paths = {}
for folder in folders:
    paths[folder] = {
        "to_process_path": os.path.join(workspace_path, folder, folder_to_execute, "to_process", "Verint"),
        "op_file_path": os.path.join(workspace_path, folder, folder_to_execute, "output", "Verint"),
        "archive_path": os.path.join(workspace_path, folder, folder_to_execute, "archive", "Verint"),
        "log_file_path": os.path.join(workspace_path, folder, folder_to_execute, "logs", "Verint")
    }

# Add paths for raw_unzip_files and Original_zip_files inside other_lobs
raw_unzip_path = os.path.join(workspace_path,  "raw_unzip_files",folder_to_execute)
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
# MAGIC Setup configurations

# COMMAND ----------

# MAGIC %run "./Unzip_Filter"

# COMMAND ----------

# MAGIC %md
# MAGIC Step1: Unzip All files to target directory

# COMMAND ----------

try:
    verint_unzip=unzip_verint_files(source_directory_verint, folder_to_execute, raw_unzip_path, original_zip_files)
    # for unzip all files
    # verint_unzip=asyncio.run(unzip_verint_files(source_directory_verint, raw_unzip_path, original_zip_files))
except Exception as e:
    print(f'Error in Unzipping files: {e}')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Global Common Filter

# COMMAND ----------

# # Load the Excel files into DataFrames
all_lob_excel_path = "./ALL_LOB_Skill_IDS.xlsx"

# COMMAND ----------

try:
    global_filter(verint_unzip, folder_to_execute, all_lob_excel_path, paths)
except Exception as e:
    print(f'Error in global_filter: {e}')

# COMMAND ----------

processing_end_time_verint = datetime.now()

# COMMAND ----------

# MAGIC %md
# MAGIC ETL LOG table Update

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
            (metadata_table.CII_Source.like('%Verint%')) &
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
            .withColumn("Vendor_Information", lit("Verint-BSC")) \
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
update_daily_transcript_logs("metadata_table", processing_start_time_verint, processing_end_time_verint)