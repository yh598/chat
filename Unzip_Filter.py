# Databricks notebook source
# MAGIC %md
# MAGIC ##### Common Methods

# COMMAND ----------

CATALOG_NAME = "dbc_adv_anlaytics_dev"
SCHEMA_NAME = "surveyspeechextraction"
METADATA_TABLE = "metadata_table"

# COMMAND ----------

# MAGIC %md
# MAGIC Read Json Method

# COMMAND ----------

import ujson
import asyncio
import aiofiles


async def read_json(json_path: str, queue: asyncio.Queue):
    """
    Reads a JSON file from the given path asynchronously and puts the content into the queue.
    """
    try:
        async with aiofiles.open(json_path, 'r') as file:
            data = await file.read()
            await queue.put((json_path, ujson.loads(data)))
    except Exception as e:
        print(f"Error reading JSON file {json_path}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC Unzip and Copy Zip Method

# COMMAND ----------

def copy_zip_contents(source_path, zip_file, original_zip_path):
    shutil.copy(os.path.join(source_path, zip_file), os.path.join(original_zip_path, zip_file))

def unzip_file(zip_file_path, unzip_dir_path):
    shutil.unpack_archive(zip_file_path, unzip_dir_path)

# COMMAND ----------

# MAGIC %md
# MAGIC Create record and Update Delta table Method

# COMMAND ----------

import pandas as pd
import os
import shutil
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pyspark.sql.functions import col as spark_col, lit
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType

def get_finance_lobs_batch(member_ids):
    member_ids_str = "', '".join(member_ids)
    query = f"""
    WITH ranked_data AS (
        SELECT  
            SUBSCRIBER_ID,
            MBR_SFX,
            FINANCE_LOB,
            ROW_NUMBER() OVER (PARTITION BY SUBSCRIBER_ID ORDER BY STUDY_PERIOD DESC) AS rn
        FROM ec_sbx_adv_anltcs_prd1.mbrdna_prd.tbl_tgt_mbrdna_dnrmlzd_wide 
        WHERE LEFT(SUBSCRIBER_ID, 9) IN ('{member_ids_str}')
        OR LEFT(SUBSCRIBER_ID || MBR_SFX, 11) IN ('{member_ids_str}')
    )
    SELECT 
        ranked_data.SUBSCRIBER_ID || ranked_data.MBR_SFX AS ranked_MEMBER_ID,
        ranked_data.FINANCE_LOB AS ranked_FINANCE_LOB
    FROM ranked_data
    WHERE ranked_data.rn = 1
    """
    result_df = spark.sql(query)
    finance_lobs_dict = result_df.rdd.map(lambda row: (row['ranked_MEMBER_ID'], row['ranked_FINANCE_LOB'])).collectAsMap()
    return finance_lobs_dict

def create_record(metadata, content, filepath, filter_condition, CII_is_selected, source, lob, finance_lobs_dict):
    cd_value = metadata.get('CD48') or metadata.get('CD93') or metadata.get('CD6')
    finance_lob = finance_lobs_dict.get(cd_value, "Not Available")  # Replace None values with "Not Available"

    record = {
        "filename": os.path.basename(filepath),
        "CII_file_location": filepath,
        "CII_Finance_LOB": finance_lob,
        "language": content.get("language", "en-US"),
        "mediaId": content.get("mediaId", ""),
        "status": content.get("status", ""),
        "dateCreated": content.get("dateCreated", ""),
        "contentType": content.get("contentType", ""),
        "channels": content.get("channels", ""),
        "length": metadata.get("length", 0),
        "CD8_EDUID": metadata.get("CD8", ""),
        "CD10_Skill_ID": metadata.get("CD10", ""),
        "CD6_Subscriber_ID": metadata.get("CD6", ""),
        "CD48_Member_ID": metadata.get("CD48", ""),
        "CD93_Subscriber_ID": metadata.get("CD93", ""),
        "CD69_Queue_Name": metadata.get("CD69", ""),
        "CII_Source": source,
        "CII_is_selected": "Yes" if CII_is_selected else "No",
        "CII_Selection_condition": filter_condition,
        "CII_Selected_LOB": lob,
        "CII_Current_date": datetime.now().strftime("%Y-%m-%d"),
        "Direction": metadata.get("Direction", 0),
        "Duration_seconds": metadata.get("Duration_seconds", 0)
    }
    # Include all the rest metadata fields
    record.update({k: v for k, v in metadata.items() if k not in record})
    return record

# COMMAND ----------



from pyspark.sql import DataFrame


def update_delta_table(df: DataFrame, CATALOG_NAME: str, SCHEMA_NAME: str, TABLE_NAME: str) -> str:
    '''
    Function to update delta table

    Args:
    df (DataFrame): dataframe to load into delta table
    CATALOG_NAME: catalog name accessed from config notebook
    SCHEMA_NAME: schema name accessed from config notebook
    TABLE_NAME: table name accessed from config notebook

    Returns:
    status of update operation
    '''
    spark.sql(f"USE CATALOG {CATALOG_NAME}")
    spark.sql(f"USE SCHEMA {SCHEMA_NAME}")
    
    try:
        if isinstance(df, pd.DataFrame):
            df = spark.createDataFrame(df)
        
        # Rename columns to remove invalid characters
        new_columns = []
        for column in df.columns:
            new_col = column.replace(" ", "_").replace(",", "_").replace(";", "_").replace("{", "_").replace("}", "_").replace("(", "_").replace(")", "_").replace("\n", "_").replace("\t", "_").replace("=", "_")
            if new_col in new_columns:
                new_col += "_dup"
            new_columns.append(new_col)
        
        df = df.toDF(*new_columns)
        
        if spark.catalog.tableExists(f"{CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}"):
            # Get the schema of the existing table
            table_schema = spark.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}").schema
            
            # Create a dictionary of column names and their data types
            table_columns = {field.name: field.dataType for field in table_schema}
            
            # Ensure all known columns are present in the DataFrame and cast them to the correct type
            for col_name, col_type in table_columns.items():
                if col_name in df.columns:
                    df = df.withColumn(col_name, spark_col(col_name).cast(col_type))
                else:
                    df = df.withColumn(col_name, lit(None).cast(col_type))
            
            # Select columns in the order of the table schema
            df = df.select(*table_columns.keys())
            
            df.write.format("delta")\
                .mode("append")\
                .option("overwriteSchema", "True")\
                .option("mergeSchema", "True")\
                .saveAsTable(f"{CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}")
            return f"successfully updated {CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}"
        else:
            print(f"Table doesn't exist. Creating {CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME} table")
            df.write.format("delta")\
                .saveAsTable(f"{CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}")
            return f"created new table {CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}"
    except Exception as e:
        print(f"Error updating delta table: {e}")

def create_dataframe_and_update_delta(records, catalog_name, schema_name, table_name):
    """
    Creates a DataFrame from records and updates the Delta table.

    Args:
    records (list): List of records to be converted into a DataFrame.
    catalog_name (str): Catalog name for the Delta table.
    schema_name (str): Schema name for the Delta table.
    table_name (str): Table name for the Delta table.

    Returns:
    str: Status of the update operation.
    """
    # Create DataFrame from records
    df = pd.DataFrame(records)
    print("Number of records: ", len(records))
    display(df)
    
    # Update Delta table
    update_status = update_delta_table(df, catalog_name, schema_name, table_name)
    return update_status


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Global Filter Logic - Verint

# COMMAND ----------

# MAGIC %md
# MAGIC Unzip Verint files

# COMMAND ----------

import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

def unzip_verint_files(source_dir, file_name, unzip_file_path, original_zip_path):
    unzip_dir_path = os.path.join(unzip_file_path, "verint_unzip")
    if not os.path.exists(unzip_dir_path):
        os.makedirs(unzip_dir_path)
        
    source_path = os.path.join(source_dir, file_name)
    
    print("Starting to unzip the contents!!!")
    if os.path.isdir(source_path):
        os.makedirs(original_zip_path, exist_ok=True)
        
        zip_files = [f for f in os.listdir(source_path) if f.endswith('.zip')]
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Copy zip files in parallel
            copy_futures = [executor.submit(copy_zip_contents, source_path, zip_file, original_zip_path) for zip_file in zip_files]
            for future in copy_futures:
                future.result()
            
            # Unzip files in parallel
            unzip_futures = [executor.submit(unzip_file, os.path.join(original_zip_path, zip_file), unzip_dir_path) for zip_file in zip_files]
            for future in unzip_futures:
                future.result()
    else:
        print(f"{source_path} is not a directory")
    
    print(f"Unzip process complete !!! {unzip_dir_path}")
    print(f"Total unzip files at unzip path: {len(os.listdir(unzip_dir_path))}")
    return unzip_dir_path


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Unip all from source

# COMMAND ----------

# %python
# import os
# import shutil
# import asyncio
# from concurrent.futures import ThreadPoolExecutor

# async def copy_zip_contents_verint(zip_file, original_zip_path):
#     loop = asyncio.get_event_loop()
#     await loop.run_in_executor(None, shutil.copy, zip_file, os.path.join(original_zip_path, os.path.basename(zip_file)))

# async def unzip_file_verint(zip_file_path, unzip_dir_path):
#     loop = asyncio.get_event_loop()
#     await loop.run_in_executor(None, shutil.unpack_archive, zip_file_path, unzip_dir_path)

# async def unzip_verint_files(source_dir, unzip_file_path, original_zip_path):
#     # Create the subfolder inside the raw_unzip_files path
#     verint_unzip_path = os.path.join(unzip_file_path, 'verint_unzip_files')
#     if not os.path.exists(verint_unzip_path):
#         os.makedirs(verint_unzip_path)
        
#     print("Starting to unzip the contents!!!")
#     if os.path.isdir(source_dir):
#         os.makedirs(original_zip_path, exist_ok=True)
        
#         # Get all zip files in the source directory
#         zip_files = [os.path.join(source_dir, file) for file in os.listdir(source_dir) if file.endswith('.zip')]
#         print(f"Total zip files: {len(zip_files)}")
        
#         # Copy zip files in parallel
#         copy_tasks = [copy_zip_contents_verint(zip_file, original_zip_path) for zip_file in zip_files]
#         await asyncio.gather(*copy_tasks)
        
#         # Unzip files in parallel
#         unzip_tasks = [unzip_file_verint(os.path.join(original_zip_path, os.path.basename(zip_file)), verint_unzip_path) for zip_file in zip_files]
#         await asyncio.gather(*unzip_tasks)
#     else:
#         print(f"{source_dir} is not a directory")
#     print(f"Unzip process complete !!! {verint_unzip_path}")
#     print(f"Total unzip files at unzip path: {len(os.listdir(verint_unzip_path))}")
    
#     return verint_unzip_path


# COMMAND ----------

# MAGIC %md
# MAGIC With Batch updates

# COMMAND ----------

import pandas as pd
import os
import shutil
import asyncio
import nest_asyncio
import aiofiles
import ujson
from concurrent.futures import ThreadPoolExecutor

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

def load_excel_sheets_verint(all_lob_excel_path):
    print("Loading Excel sheets...")
    skill_dfs = {
        'MA': pd.read_excel(all_lob_excel_path, sheet_name='MA_BSC'),
        'IFP': pd.read_excel(all_lob_excel_path, sheet_name='IFP_BSC'),
        'Broker': pd.read_excel(all_lob_excel_path, sheet_name='Broker_BSC'),
        'Small Group': pd.read_excel(all_lob_excel_path, sheet_name='SmallGroup_BSC'),
        'MediCal': pd.read_excel(all_lob_excel_path, sheet_name='Medical_BSC'),
        'Core Premier': pd.read_excel(all_lob_excel_path, sheet_name='CorePremier_BSC')
    }
    # Add '106' prefix to the Skill column in each DataFrame
    for lob, df in skill_dfs.items():
        df['Skill'] = '106' + df['Skill'].astype(str)
        
    org_df = pd.read_excel(all_lob_excel_path, sheet_name='MA-Org-IDS')
    medical_queue_df = pd.read_excel(all_lob_excel_path, sheet_name='Medical_BSC_Queue')
    print("Excel sheets loaded.")
    return skill_dfs, org_df, medical_queue_df

def get_medicare_transcript_ids(org_df):
    print("Getting Medicare transcript IDs...")
    ids = ['80500' + str(id) for id in org_df[org_df['NAME'].isin([
        'Member Medicare', 
        'Member Medicare Care Navigation', 
        'Member Medicare Rx Tech'
    ])]['ID'].values]
    print(f"Medicare transcript IDs: {ids}")
    return ids

def get_finance_lobs_batch(member_ids):
    member_ids_str = "', '".join(member_ids)
    query = f"""
    WITH ranked_data AS (
        SELECT  
            SUBSCRIBER_ID,
            MBR_SFX,
            FINANCE_LOB,
            ROW_NUMBER() OVER (PARTITION BY SUBSCRIBER_ID ORDER BY STUDY_PERIOD DESC) AS rn
        FROM ec_sbx_adv_anltcs_prd1.mbrdna_prd.tbl_tgt_mbrdna_dnrmlzd_wide 
        WHERE LEFT(SUBSCRIBER_ID, 9) IN ('{member_ids_str}')
        OR LEFT(SUBSCRIBER_ID || MBR_SFX, 11) IN ('{member_ids_str}')
    )
    SELECT 
        ranked_data.SUBSCRIBER_ID || ranked_data.MBR_SFX AS ranked_MEMBER_ID,
        ranked_data.FINANCE_LOB AS ranked_FINANCE_LOB
    FROM ranked_data
    WHERE ranked_data.rn = 1
    """
    result_df = spark.sql(query)
    finance_lobs_dict = result_df.rdd.map(lambda row: (row['ranked_MEMBER_ID'], row['ranked_FINANCE_LOB'])).collectAsMap()
    return finance_lobs_dict

def should_move_file_verint(metadata, skill_dfs, medicare_transcript_ids, lob, medical_queue_df, finance_lobs_dict):
    cd10_value = metadata.get('CD10')
    org_id = metadata.get('org_id')
    
    skill_id_match = False
    org_id_match = False
    finance_lob_values = []

    if lob in skill_dfs:
        skill_id_match = cd10_value and cd10_value in skill_dfs[lob]['Skill'].values
        if lob == 'MA':
            org_id_match = org_id and org_id in medicare_transcript_ids
            finance_lob_values = ["GMAPD", "Total MAPD", "", None, ' ']
        elif lob == 'IFP':
            org_id_match = True  # No org_id filter for IFP
            finance_lob_values = ["Individual (IFP)", "", None, ' ']
        elif lob == 'MediCal':
            cd69_value = metadata.get('CD69')
            queue_match = cd69_value and cd69_value in medical_queue_df['Verint_Queue_Name'].values
            skill_id_match = skill_id_match or queue_match
            org_id_match = True  # No org_id filter for MediCal
        else:
            org_id_match = True  # No org_id filter for other LOBs
    
    cd_value = metadata.get('CD48') or metadata.get('CD93') or metadata.get('CD6')
    if cd_value and lob in ['MA', 'IFP']:
        finance_lobs = finance_lobs_dict.get(cd_value, [])
        finance_lob_match = any(lob in finance_lob_values for lob in finance_lobs) or not finance_lobs
        
        if (skill_id_match or org_id_match) and finance_lob_match:
            print(f"File matches with skill_id: {cd10_value} or org_id: {org_id} and finance_lob: {finance_lobs}")
            return True, "Filtered for skill id, org id, and finance lob"
        else:
            if not skill_id_match:
                return False, "Not filtered due to skill id not matched"
            if not org_id_match:
                return False, "Not filtered due to org id not matched"
            if not finance_lob_match:
                return False, "Not filtered due to finance lob not matched"
    else:
        if skill_id_match or org_id_match:
            print(f"File matches with skill_id: {cd10_value} or org_id: {org_id}")
            return True, "Filtered for skill id or org id"
        else:
            if not skill_id_match:
                return False, "Not filtered due to skill id not matched"
            if not org_id_match:
                return False, "Not filtered due to org id not matched"
    
    print(f"No match for file with skill_id: {cd10_value}, org_id: {org_id}, and finance lob: {finance_lobs if cd_value else 'N/A'}")
    return False, "Not filtered due to skill id, org id, and finance lob not matched"

async def process_file_verint(json_data, file_path, skill_dfs, medicare_transcript_ids, paths, records, lob_counts, medical_queue_df, finance_lobs_dict):
    try:
        if json_data is None:
            return
        
        metadata = json_data.get('metadata', {})
        if not metadata:
            print(f"No metadata found in file: {file_path}")
            return
        
        # Check direction condition
        selectedCallDirection = 1
        if int(metadata.get("Direction", 0)) != selectedCallDirection:
            condition = "Not filtered due to direction condition"
            record = create_record(metadata, json_data, file_path, condition, False, "Verint-BSC", "", finance_lobs_dict)
            records.append(record)
            print(f"Skipping file {file_path} due to direction condition.")
            return
        
        # Check duration condition
        min_duration = 60  # Example duration in seconds
        if int(metadata.get("Duration_seconds", 0)) <= min_duration:
            condition = "Not filtered due to duration condition"
            record = create_record(metadata, json_data, file_path, condition, False, "Verint-BSC", "", finance_lobs_dict)
            records.append(record)
            print(f"Skipping file {file_path} due to duration condition.")
            return
        
        cd10_value = metadata.get('CD10')
        cd69_value = metadata.get('CD69')
        
        # Check and copy file for each LOB
        for lob in skill_dfs.keys():
            if (cd10_value and cd10_value in skill_dfs[lob]['Skill'].values) or (lob == 'MediCal' and cd69_value and cd69_value in medical_queue_df['Verint_Queue_Name'].values):
                is_filtered, condition = should_move_file_verint(metadata, skill_dfs, medicare_transcript_ids, lob, medical_queue_df, finance_lobs_dict)
                if is_filtered:
                    target_path = paths[lob]['to_process_path']
                    print(f"Copying file {file_path} to {target_path} for {lob}")
                    shutil.copy(file_path, target_path)
                    record = create_record(metadata, json_data, file_path, condition, True, "Verint-BSC", lob, finance_lobs_dict)
                    records.append(record)
                    lob_counts[lob] += 1
                else:
                    record = create_record(metadata, json_data, file_path, condition, False, "Verint-BSC", "", finance_lobs_dict)
                    records.append(record)
                    print(f"File {file_path} did not match any LOB conditions.")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

async def read_files_to_queue(json_files, queue):
    for file_path in json_files:
        await read_json(file_path, queue)

async def process_queue(queue, skill_dfs, medicare_transcript_ids, paths, records, lob_counts, medical_queue_df, finance_lobs_dict):
    while True:
        file_path, json_data = await queue.get()
        if json_data is None and file_path is None:
            break
        await process_file_verint(json_data, file_path, skill_dfs, medicare_transcript_ids, paths, records, lob_counts, medical_queue_df, finance_lobs_dict)
        queue.task_done()

async def process_batch_verint(batch_files, skill_dfs, medicare_transcript_ids, paths, records, lob_counts, medical_queue_df, finance_lobs_dict):
    queue = asyncio.Queue()
    read_task = asyncio.create_task(read_files_to_queue(batch_files, queue))
    process_task = asyncio.create_task(process_queue(queue, skill_dfs, medicare_transcript_ids, paths, records, lob_counts, medical_queue_df, finance_lobs_dict))
    await read_task
    await queue.put((None, None))  # Signal the end of the queue
    await process_task

def global_filter(raw_unzip_files, folder_to_execute, all_lob_excel_path, paths, batch_size=500):
    try:
        print("Starting global filter process...")
        # Load Excel sheets
        skill_dfs, org_df, medical_queue_df = load_excel_sheets_verint(all_lob_excel_path)
        print(f"Medical df:{medical_queue_df}")
        
        # Get Medicare transcript IDs
        medicare_transcript_ids = get_medicare_transcript_ids(org_df)
        
        # List all JSON files in the raw_unzip_files directory
        json_files = [os.path.join(raw_unzip_files, f) for f in os.listdir(raw_unzip_files) if f.endswith('.json')]
    
        # Use sample of 1000 files
        # json_files = json_files[:1000]
        total_files = len(json_files)
        print(f"Found {total_files} JSON files to process.")
        
        # Collect all member_ids
        member_ids = set()
        for file in json_files:
            try:
                content = ujson.loads(open(file, 'r').read())
                metadata = content.get("metadata", {})
                if metadata:
                    member_id = metadata.get('CD48') or metadata.get('CD93') or metadata.get('CD6')
                    if member_id:
                        member_ids.add(member_id)
            except FileNotFoundError:
                print(f"File not found: {file}")
            except Exception as e:
                print(f"Error reading file {file}: {e}")
        
        # Get finance_lobs in batch
        finance_lobs_dict = get_finance_lobs_batch(list(member_ids))
        
        records = []
        lob_counts = {lob: 0 for lob in skill_dfs.keys()}
        
        # Process files in batches
        loop = asyncio.get_event_loop()
        for i in range(0, total_files, batch_size):
            batch_files = json_files[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1} with {len(batch_files)} files.")
            loop.run_until_complete(process_batch_verint(batch_files, skill_dfs, medicare_transcript_ids, paths, records, lob_counts, medical_queue_df, finance_lobs_dict))
        
        # Print the counts of each LOB filtered files
        for lob, count in lob_counts.items():
            print(f"LOB {lob} filtered files count: {count}")
        
        # Create DataFrame from records and update Delta table
        update_status = create_dataframe_and_update_delta(records, CATALOG_NAME, SCHEMA_NAME, METADATA_TABLE)
        print(update_status)
        
        print("Global filter process completed.")
    except Exception as e:
        print(f"Error in global filter: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Global filter logic- Sagility

# COMMAND ----------

# MAGIC %md
# MAGIC Unzip Sagility Files

# COMMAND ----------


import os
import shutil
from concurrent.futures import ThreadPoolExecutor

def unzip_sagility_files(source, original_zip_path, raw_unzip_path,folder_to_execute):
    # Ensure the target directories exist
    os.makedirs(original_zip_path, exist_ok=True)
    os.makedirs(raw_unzip_path, exist_ok=True)
    
    # Create new paths under original_zip_path and raw_unzip_path
    sagility_zip_path = os.path.join(original_zip_path, "sagility_zip_files")
    sagility_unzip_path = os.path.join(raw_unzip_path, "sagility_raw_unzip")
    os.makedirs(sagility_zip_path, exist_ok=True)
    os.makedirs(sagility_unzip_path, exist_ok=True)
    
    # List all zip files in the source directory that contain the folder_to_execute date
    zip_files = [f for f in os.listdir(source) if f.endswith('.zip') and folder_to_execute in f]
    print(f"Zip files: {zip_files}")
    
    # Copy and unzip files in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Copy zip files to sagility_zip_path
        copy_futures = [executor.submit(copy_zip_contents, source, zip_file, sagility_zip_path) for zip_file in zip_files]
        for future in copy_futures:
            future.result()
        
        # Unzip files to sagility_unzip_path
        unzip_futures = [executor.submit(unzip_file, os.path.join(sagility_zip_path, zip_file), sagility_unzip_path) for zip_file in zip_files]
        for future in unzip_futures:
            future.result()
    
    # Collect all JSON files from the unzipped directories
    raw_json_files = []
    for root, _, files in os.walk(sagility_unzip_path):
        for file in files:
            if file.endswith('.json'):
                raw_json_files.append(os.path.join(root, file))
    
    print(f"Unzipping and moving files from {source} to {sagility_unzip_path} completed.")
    print(f"Total unzipped JSON files: {len(raw_json_files)}")
    
    return sagility_unzip_path

# COMMAND ----------

# MAGIC %md
# MAGIC Unzip All sagility files

# COMMAND ----------

# %python
# import os
# import shutil
# from concurrent.futures import ThreadPoolExecutor
# import zipfile

# def copy_zip_contents_sagility(source, zip_file, destination):
#     shutil.copy(os.path.join(source, zip_file), destination)

# def unzip_file_sagility(zip_path, extract_to):
#     try:
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall(extract_to)
#     except FileNotFoundError as e:
#         print(f"File not found: {zip_path}. Error: {e}")
#     except Exception as e:
#         print(f"Error unzipping file: {zip_path}. Error: {e}")

# def unzip_sagility_files(source, original_zip_path, raw_unzip_path, folder_to_execute):
#     # Ensure the target directories exist
#     os.makedirs(original_zip_path, exist_ok=True)
#     os.makedirs(raw_unzip_path, exist_ok=True)
    
#     # Create new paths under original_zip_path and raw_unzip_path
#     sagility_zip_path = os.path.join(original_zip_path, "sagility_zip_files")
#     sagility_unzip_path = os.path.join(raw_unzip_path, "sagility_raw_unzip")
#     os.makedirs(sagility_zip_path, exist_ok=True)
#     os.makedirs(sagility_unzip_path, exist_ok=True)
    
#     # List all zip files in the source directory
#     zip_files = [f for f in os.listdir(source) if f.endswith('.zip')]
#     print(f"Zip files: {zip_files}")
    
#     # Copy and unzip files in parallel
#     with ThreadPoolExecutor(max_workers=10) as executor:
#         # Copy zip files to sagility_zip_path
#         copy_futures = [executor.submit(copy_zip_contents_sagility, source, zip_file, sagility_zip_path) for zip_file in zip_files]
#         for future in copy_futures:
#             future.result()
        
#         # Unzip files to sagility_unzip_path
#         unzip_futures = [executor.submit(unzip_file_sagility, os.path.join(sagility_zip_path, zip_file), sagility_unzip_path) for zip_file in zip_files]
#         for future in unzip_futures:
#             future.result()
    
#     print(f"Unzipping and moving files from {source} to {sagility_unzip_path} completed.")
    
#     return sagility_unzip_path


# COMMAND ----------

# MAGIC %md
# MAGIC With Batch updated

# COMMAND ----------

import pandas as pd
import os
import shutil
import ujson
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor

def load_excel_sheets_sagility(all_lob_excel_path):
    try:
        print("Loading Excel sheets...")
        skill_dfs = {
            'IFP': pd.read_excel(all_lob_excel_path, sheet_name='IFP-HuntGroup'),
            'Provider': pd.read_excel(all_lob_excel_path, sheet_name='Provider-HuntGroup'),
            'Core Premier': pd.read_excel(all_lob_excel_path, sheet_name='IFP-HuntGroup')
        }
        print("Excel sheets loaded.")
        return skill_dfs
    except Exception as e:
        print(f"Error loading Excel sheets: {e}")
        return {}

def get_finance_lobs_batch(member_ids):
    try:
        member_ids_str = "', '".join(member_ids)
        query = f"""
        WITH ranked_data AS (
            SELECT  
                SUBSCRIBER_ID,
                MBR_SFX,
                FINANCE_LOB,
                ROW_NUMBER() OVER (PARTITION BY SUBSCRIBER_ID ORDER BY STUDY_PERIOD DESC) AS rn
            FROM ec_sbx_adv_anltcs_prd1.mbrdna_prd.tbl_tgt_mbrdna_dnrmlzd_wide 
            WHERE LEFT(SUBSCRIBER_ID, 9) IN ('{member_ids_str}')
            OR LEFT(SUBSCRIBER_ID || MBR_SFX, 11) IN ('{member_ids_str}')
        )
        SELECT 
            ranked_data.SUBSCRIBER_ID || ranked_data.MBR_SFX AS ranked_MEMBER_ID,
            ranked_data.FINANCE_LOB AS ranked_FINANCE_LOB
        FROM ranked_data
        WHERE ranked_data.rn = 1
        """
        result_df = spark.sql(query)
        finance_lobs_dict = result_df.rdd.map(lambda row: (row['ranked_MEMBER_ID'], row['ranked_FINANCE_LOB'])).collectAsMap()
        return finance_lobs_dict
    except Exception as e:
        print(f"Error fetching finance LOBs: {e}")
        return {}

def should_move_file_provider(file_path, metadata, skill_dfs):
    try:
        cd10_value = metadata.get('CD10')
        lob = metadata.get('LOB')
        if cd10_value:
            huntgroup_row = skill_dfs['Provider'][skill_dfs['Provider']['HuntGroup'] == int(cd10_value)]
            if not huntgroup_row.empty:
                return True, "Filtered for Provider"
            else:
                return False, "Not filtered due to skill id not matched"
        return False, "Not filtered due to missing CD10"
    except Exception as e:
        print(f"Error in should_move_file_provider: {e}")
        return False, "Error in processing"

def should_move_file_ifp(metadata, skill_dfs):
    try:
        cd10_value = metadata.get('CD10')
        lob = metadata.get('LOB')
        if cd10_value:
            huntgroup_row = skill_dfs['IFP'][skill_dfs['IFP']['HuntGroup'] == int(cd10_value)]
            if not huntgroup_row.empty:
                return True, "Filtered for IFP"
            else:
                return False, "Not filtered due to skill id not matched"
        return False, "Not filtered due to missing CD10"
    except Exception as e:
        print(f"Error in should_move_file_ifp: {e}")
        return False, "Error in processing"

def should_move_file_premier(metadata, skill_dfs):
    try:
        cd10_value = metadata.get('CD10')
        lob = metadata.get('LOB')
        if cd10_value:
            huntgroup_row = skill_dfs['Core Premier'][skill_dfs['Core Premier']['HuntGroup'] == int(cd10_value)]
            if not huntgroup_row.empty:
                return True, "Filtered for Core Premier"
            else:
                return False, "Not filtered due to skill id not matched"
        return False, "Not filtered due to missing CD10"
    except Exception as e:
        print(f"Error in should_move_file_premier: {e}")
        return False, "Error in processing"

def process_file_sagility(file_data, skill_dfs, paths, records, lob_counts, finance_lobs_dict):
    file_path, json_data = file_data
    try:
        metadata = json_data.get('metadata', {})
        
        # Check direction condition
        selectedCallDirection = 1
        if int(metadata.get("Direction")) != selectedCallDirection:
            condition = "Not filtered due to direction condition"
            record = create_record(metadata, json_data, file_path, condition, False, "Sagility", "", finance_lobs_dict)
            records.append(record)
            print(f"Skipping file {file_path} due to direction condition.")
            return
        
        # Check duration condition
        min_duration = 60  # Example duration in seconds
        if int(metadata.get("Duration_seconds", 0)) <= min_duration:
            condition = "Not filtered due to duration condition"
            record = create_record(metadata, json_data, file_path, condition, False, "Sagility", "", finance_lobs_dict)
            records.append(record)
            print(f"Skipping file {file_path} due to duration condition.")
            return
        
        # Check filename for Provider
        if 'provider' in file_path.lower():
            is_filtered, condition = should_move_file_provider(file_path, metadata, skill_dfs)
            if is_filtered:
                target_path = paths['Provider']['to_process_path']
                print(f"Copying file {file_path} to {target_path} for Provider")
                shutil.copy(file_path, target_path)
                record = create_record(metadata, json_data, file_path, condition, True, "Sagility", "Provider", finance_lobs_dict)
                records.append(record)
                lob_counts['Provider'] += 1
                return
            else:
                condition = "Not filtered due to not provider transcripts."
                record = create_record(metadata, json_data, file_path, condition, False, "Sagility", "", finance_lobs_dict)
                records.append(record)
                print(f"File {file_path} did not match Provider conditions.")
                return
        
        # Check filename for Member
        if 'member' in file_path.lower():
            # Check and copy file for IFP
            is_filtered, condition = should_move_file_ifp(metadata, skill_dfs)
            if is_filtered:
                target_path = paths['IFP']['to_process_path']
                print(f"Copying file {file_path} to {target_path} for IFP")
                shutil.copy(file_path, target_path)
                record = create_record(metadata, json_data, file_path, condition, True, "Sagility", "IFP", finance_lobs_dict)
                records.append(record)
                lob_counts['IFP'] += 1
                return
            
            # Check and copy file for Core Premier
            is_filtered, condition = should_move_file_premier(metadata, skill_dfs)
            if is_filtered:
                target_path = paths['Core Premier']['to_process_path']
                print(f"Copying file {file_path} to {target_path} for Core Premier")
                shutil.copy(file_path, target_path)
                record = create_record(metadata, json_data, file_path, condition, True, "Sagility", "Core Premier", finance_lobs_dict)
                records.append(record)
                lob_counts['Core Premier'] += 1
                return
            
            # If no conditions matched
            condition = "Not filtered due to skill id not matched"
            record = create_record(metadata, json_data, file_path, condition, False, "Sagility", "", finance_lobs_dict)
            records.append(record)
            print(f"File {file_path} did not match any Member conditions.")
            return
        
        # If no filename conditions matched
        condition = "Not filtered due to filename conditions"
        record = create_record(metadata, json_data, file_path, condition, False, "Sagility", "", finance_lobs_dict)
        records.append(record)
        print(f"File {file_path} did not match any filename conditions.")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

async def process_batch_sagility(batch_files, skill_dfs, paths, records, lob_counts, finance_lobs_dict):
    queue = asyncio.Queue()
    read_tasks = [read_json(file_path, queue) for file_path in batch_files]
    await asyncio.gather(*read_tasks)
    
    file_data_list = []
    while not queue.empty():
        file_data_list.append(await queue.get())
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_file_sagility, file_data, skill_dfs, paths, records, lob_counts, finance_lobs_dict) for file_data in file_data_list]
        for future in futures:
            future.result()

def global_filter_sagility(sagility_unzip_path, folder_to_execute, all_lob_excel_path, paths, batch_size=500):
    try:
        print("Starting Sagility filter process...")
        # Load Excel sheets
        skill_dfs = load_excel_sheets_sagility(all_lob_excel_path)
        
        # List all JSON files in the sagility_unzip_path directory
        json_files = []
        for root, _, files in os.walk(sagility_unzip_path):
            for filename in files:
                if filename.endswith('.json'):
                    json_files.append(os.path.join(root, filename))
        
        # Limit to a sample of 1000 files
        # json_files = json_files[:1000]
        total_files = len(json_files)
        print(f"Found {total_files} JSON files to process.")
        
        # Collect all member_ids
        member_ids = set()
        for file in json_files:
            try:
                content = ujson.loads(open(file, 'r').read())
                metadata = content.get("metadata", {})
                if metadata:
                    member_id = metadata.get('CD48') or metadata.get('CD93') or metadata.get('CD6')
                    if member_id:
                        member_ids.add(member_id)
            except Exception as e:
                print(f"Error reading file {file}: {e}")
        
        # Get finance_lobs in batch
        finance_lobs_dict = get_finance_lobs_batch(list(member_ids))
        
        records = []
        lob_counts = {'IFP': 0, 'Provider': 0, 'Core Premier': 0}
        
        # Process files in batches
        loop = asyncio.get_event_loop()
        for i in range(0, total_files, batch_size):
            batch_files = json_files[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1} with {len(batch_files)} files.")
            loop.run_until_complete(process_batch_sagility(batch_files, skill_dfs, paths, records, lob_counts, finance_lobs_dict))
        
        # Print the counts of each LOB filtered files
        for lob, count in lob_counts.items():
            print(f"LOB {lob} filtered files count: {count}")
        
        # Create DataFrame from records and update Delta table
        update_status = create_dataframe_and_update_delta(records, CATALOG_NAME, SCHEMA_NAME, METADATA_TABLE)
        print(update_status)
        
        print("Sagility filter process completed.")
    except Exception as e:
        print(f"Error in global_filter_sagility: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Global Filter Logic - TTec

# COMMAND ----------

# MAGIC %md
# MAGIC Unzip TTec files- Daywise

# COMMAND ----------


import os
import shutil
from concurrent.futures import ThreadPoolExecutor

def unzip_ttec_files(source_path, original_zip_path, raw_unzip_path, folder_to_execute):
    # Ensure the target directories exist
    os.makedirs(original_zip_path, exist_ok=True)
    os.makedirs(raw_unzip_path, exist_ok=True)
    
    # Create a new path under raw_unzip_path
    ttec_unzip_path = os.path.join(raw_unzip_path, "ttec_unzip")
    os.makedirs(ttec_unzip_path, exist_ok=True)
    
    # Format the folder_to_execute date to 'YYYYMMDD'
    folder_to_execute_date = folder_to_execute.replace('-', '')
    
    # List all zip files in the source directory that contain the folder_to_execute_date
    zip_files = [f for f in os.listdir(source_path) if f.endswith('.zip') and folder_to_execute_date in f]
    print("Zip files to process:", zip_files)
    # Copy and unzip files in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Copy zip files to original_zip_path
        copy_futures = [executor.submit(copy_zip_contents, source_path, zip_file, original_zip_path) for zip_file in zip_files]
        for future in copy_futures:
            future.result()
        
        # Unzip files to ttec_unzip_path
        unzip_futures = [executor.submit(unzip_file, os.path.join(original_zip_path, zip_file), ttec_unzip_path) for zip_file in zip_files]
        for future in unzip_futures:
            future.result()
    
    print(f"Unzipping and moving files from {source_path} to {ttec_unzip_path} completed.")
    print(f"Total unzip files at unzip path: {len(os.listdir(ttec_unzip_path))}")
    
    return ttec_unzip_path

# COMMAND ----------

# MAGIC %md
# MAGIC Unzip Ttec all files

# COMMAND ----------


# import os
# import shutil
# from concurrent.futures import ThreadPoolExecutor

# def copy_zip_contents_ttec(source_path, zip_file, original_zip_path):
#     shutil.copy(os.path.join(source_path, zip_file), os.path.join(original_zip_path, zip_file))

# def unzip_file(zip_file_path, unzip_dir_path):
#     try:
#         shutil.unpack_archive(zip_file_path, unzip_dir_path)
#     except (shutil.ReadError, OSError) as e:
#         print(f"Error unzipping file {zip_file_path}: {e}")

# def unzip_ttec_files(source_path, original_zip_path, raw_unzip_path, folder_to_execute):
#     # Ensure the target directories exist
#     os.makedirs(original_zip_path, exist_ok=True)
#     os.makedirs(raw_unzip_path, exist_ok=True)
    
#     # Create a new path under raw_unzip_path
#     ttec_unzip_path = os.path.join(raw_unzip_path, "ttec_unzip")
#     os.makedirs(ttec_unzip_path, exist_ok=True)
    
#     # List all zip files in the source directory
#     zip_files = [f for f in os.listdir(source_path) if f.endswith('.zip')]
#     print("Zip files to process:", zip_files)
    
#     # Copy and unzip files in parallel
#     with ThreadPoolExecutor(max_workers=10) as executor:
#         # Copy zip files to original_zip_path
#         copy_futures = [executor.submit(copy_zip_contents_ttec, source_path, zip_file, original_zip_path) for zip_file in zip_files]
#         for future in copy_futures:
#             future.result()
        
#         # Unzip files to ttec_unzip_path
#         unzip_futures = [executor.submit(unzip_file, os.path.join(original_zip_path, zip_file), ttec_unzip_path) for zip_file in zip_files]
#         for future in unzip_futures:
#             future.result()
    
#     print(f"Unzipping and moving files from {source_path} to {ttec_unzip_path} completed.")
#     print(f"Total unzipped files at unzip path: {len(os.listdir(ttec_unzip_path))}")
    
#     return ttec_unzip_path

# COMMAND ----------

# %python
# import os
# import shutil
# import zipfile
# from concurrent.futures import ThreadPoolExecutor

# def copy_zip_contents_ttec(source_path, zip_file, original_zip_path):
#     try:
#         shutil.copy(os.path.join(source_path, zip_file), os.path.join(original_zip_path, zip_file))
#     except Exception as e:
#         print(f"Error copying file {zip_file}: {e}")

# def unzip_file(zip_file_path, unzip_dir_path):
#     try:
#         with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#             zip_ref.extractall(unzip_dir_path)
#     except (zipfile.BadZipFile, OSError) as e:
#         print(f"Error unzipping file {zip_file_path}: {e}")

# def unzip_ttec_files(source_path, original_zip_path, raw_unzip_path, folder_to_execute):
#     # Ensure the target directories exist
#     os.makedirs(original_zip_path, exist_ok=True)
#     os.makedirs(raw_unzip_path, exist_ok=True)
    
#     # Create a new path under raw_unzip_path
#     ttec_unzip_path = os.path.join(raw_unzip_path, "ttec_unzip_v2")
#     os.makedirs(ttec_unzip_path, exist_ok=True)
    
#     # List all zip files in the source directory
#     zip_files = [f for f in os.listdir(source_path) if f.endswith('.zip')]
#     print("Zip files to process:", zip_files)
    
#     # Copy and unzip files in parallel
#     with ThreadPoolExecutor(max_workers=10) as executor:
#         # Copy zip files to original_zip_path
#         copy_futures = [executor.submit(copy_zip_contents_ttec, source_path, zip_file, original_zip_path) for zip_file in zip_files]
#         for future in copy_futures:
#             future.result()
        
#         # Unzip files to ttec_unzip_path
#         unzip_futures = [executor.submit(unzip_file, os.path.join(original_zip_path, zip_file), ttec_unzip_path) for zip_file in zip_files]
#         for future in unzip_futures:
#             future.result()
    
#     print(f"Unzipping and moving files from {source_path} to {ttec_unzip_path} completed.")
#     print(f"Total unzipped files at unzip path: {len(os.listdir(ttec_unzip_path))}")
    
#     return ttec_unzip_path

# COMMAND ----------

# %python
# import os
# import zipfile
# import asyncio
# import aiofiles
# import ujson
# from concurrent.futures import ThreadPoolExecutor

# async def copy_zip_contents_ttec(source_path, zip_file, original_zip_path):
#     try:
#         shutil.copy(os.path.join(source_path, zip_file), os.path.join(original_zip_path, zip_file))
#     except Exception as e:
#         print(f"Error copying file {zip_file}: {e}")

# async def unzip_file(zip_file_path, unzip_dir_path):
#     try:
#         with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#             for member in zip_ref.namelist():
#                 if member.endswith('.json'):
#                     member_path = os.path.join(unzip_dir_path, os.path.basename(member))
#                     with zip_ref.open(member) as source:
#                         async with aiofiles.open(member_path, 'wb') as target:
#                             await target.write(source.read())
#     except (zipfile.BadZipFile, OSError) as e:
#         print(f"Error unzipping file {zip_file_path}: {e}")

# async def unzip_ttec_files(source_path, original_zip_path, raw_unzip_path, folder_to_execute):
#     os.makedirs(original_zip_path, exist_ok=True)
#     os.makedirs(raw_unzip_path, exist_ok=True)
    
#     ttec_unzip_path = os.path.join(raw_unzip_path, "ttec_unzip")
#     os.makedirs(ttec_unzip_path, exist_ok=True)
    
#     zip_files = [f for f in os.listdir(source_path) if f.endswith('.zip')]
#     print("Zip files to process:", zip_files)
    
#     copy_tasks = [copy_zip_contents_ttec(source_path, zip_file, original_zip_path) for zip_file in zip_files]
#     await asyncio.gather(*copy_tasks)
    
#     unzip_tasks = [unzip_file(os.path.join(original_zip_path, zip_file), ttec_unzip_path) for zip_file in zip_files]
#     await asyncio.gather(*unzip_tasks)
    
#     print(f"Unzipping and moving files from {source_path} to {ttec_unzip_path} completed.")
#     print(f"Total unzipped files at unzip path: {len(os.listdir(ttec_unzip_path))}")
    
#     return ttec_unzip_path

# # To run the async function in a synchronous context
# def run_unzip_ttec_files(source_path, original_zip_path, raw_unzip_path, folder_to_execute):
#     loop = asyncio.get_event_loop()
#     return loop.run_until_complete(unzip_ttec_files(source_path, original_zip_path, raw_unzip_path, folder_to_execute))

# COMMAND ----------

# MAGIC %md
# MAGIC With batch Update

# COMMAND ----------

import pandas as pd
import os
import shutil
import ujson
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from glob import glob


def should_move_file_ttec(metadata, skill_df, lob, finance_lobs_dict):
    cd10_value = metadata.get('CD10')
    if cd10_value:
        skill_row = skill_df[skill_df['Skill_ID'] == cd10_value]
        if not skill_row.empty and skill_row.iloc[0]['BSC_LOB'].startswith(lob):
            cd_value = metadata.get('CD48') or metadata.get('CD93') or metadata.get('CD6')
            if cd_value:
                finance_lobs = finance_lobs_dict.get(cd_value, [])
                if any(lob in finance_lobs for lob in ["Individual (IFP)", "", None, ' ']):
                    return True, "Filtered for skill id and finance lob"
            return True, "Filtered for skill id"
        else:
            return False, "Not filtered due to skill id not matched"
    return False, "Not filtered due to LOB conditions"


async def filter_ttec_transcripts(read_queue: asyncio.Queue, skill_df, paths, records, finance_lobs_dict, filtered_json_files):
    while True:
        file_path, json_data = await read_queue.get()
        if file_path is None:
            break
        metadata = json_data.get('metadata', {})
        selectedCallDirection = 1
        
        # Check direction condition
        if int(metadata.get("Direction", 0)) != selectedCallDirection:
            condition = "Not filtered due to direction condition"
            record = create_record(metadata, json_data, file_path, condition, False, "TTec", "", finance_lobs_dict)
            records.append(record)
            print(f"Skipping file {file_path} due to direction condition.")
            read_queue.task_done()
            continue
        
        # Check duration condition
        if int(metadata.get("Duration_seconds", 0)) <= 60:
            condition = "Not filtered due to duration condition"
            record = create_record(metadata, json_data, file_path, condition, False, "TTec", "", finance_lobs_dict)
            records.append(record)
            print(f"Skipping file {file_path} due to duration condition.")
            read_queue.task_done()
            continue
        
        for lob in paths:
            is_filtered, condition = should_move_file_ttec(metadata, skill_df, lob, finance_lobs_dict)
            if is_filtered:
                target_path = os.path.join(paths[lob]["to_process_path"], os.path.basename(file_path))
                shutil.copy(file_path, target_path)
                record = create_record(metadata, json_data, file_path, condition, True, "TTec", lob, finance_lobs_dict)
                records.append(record)
                filtered_json_files[lob] += 1
                print(f"Moved {file_path} to {target_path} for {lob}")
                break
        else:
            record = create_record(metadata, json_data, file_path, condition, False, "TTec", "", finance_lobs_dict)
            records.append(record)
            print(f"No filter match for file: {file_path}")
        
        read_queue.task_done()

async def process_batch_ttec(batch_files, skill_df, paths, records, finance_lobs_dict, filtered_json_files):
    read_queue = asyncio.Queue()
    
    # Start the file processing tasks
    processing_tasks = [asyncio.create_task(filter_ttec_transcripts(read_queue, skill_df, paths, records, finance_lobs_dict, filtered_json_files)) for _ in range(10)]
    
    # Start the file reading tasks
    reading_tasks = [asyncio.create_task(read_json(file_path, read_queue)) for file_path in batch_files]
    
    # Wait for all reading tasks to complete
    await asyncio.gather(*reading_tasks)
    
    # Signal the processing tasks to stop
    for _ in range(10):
        await read_queue.put((None, None))
    
    # Wait for all processing tasks to complete
    await asyncio.gather(*processing_tasks)

def filter_ttec_files(unzip_path_ttec, all_lob_excel_path, paths, batch_size=1000):
    # Load the "TTec-ALL" sheet from the Excel file into a DataFrame
    skill_df = pd.read_excel(all_lob_excel_path, sheet_name="TTec-ALL")
    
    # Ensure the target directories exist
    for lob in paths:
        os.makedirs(paths[lob]["to_process_path"], exist_ok=True)
        print(lob)
    
    # List all JSON files in the raw_unzip_path
    total_json_files = 0
    filtered_json_files = {lob: 0 for lob in paths}
    records = []
    
    json_files = []
    # List all JSON files in the specified directory only
    json_files = glob(os.path.join(unzip_path_ttec, "*.json"))
    total_files = len(json_files)
    print(f"Found {total_files} JSON files to process.")
    
    # Collect all member_ids
    member_ids = set()
    for file_path in json_files:
        content = ujson.loads(open(file_path, 'r').read())
        metadata = content.get("metadata", {})
        if metadata:
            member_id = metadata.get('CD48') or metadata.get('CD93') or metadata.get('CD6')
            if member_id:
                member_ids.add(member_id)
    
    # Get finance_lobs in batch
    finance_lobs_dict = get_finance_lobs_batch(list(member_ids))
    
    # Process files in batches
    loop = asyncio.get_event_loop()
    for i in range(0, total_files, batch_size):
        batch_files = json_files[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} with {len(batch_files)} files.")
        loop.run_until_complete(process_batch_ttec(batch_files, skill_df, paths, records, finance_lobs_dict, filtered_json_files))
    
    print(f"Filtering and moving JSON files from {unzip_path_ttec} completed.")
    print(f"Number of files in raw unzip directory: {total_json_files}")
    for lob in paths:
        print(f"Number of filtered files in {lob} target directory: {filtered_json_files[lob]}")
    
    # Create DataFrame from records and update Delta table
    update_status = create_dataframe_and_update_delta(records, CATALOG_NAME, SCHEMA_NAME, METADATA_TABLE)
    print(update_status)