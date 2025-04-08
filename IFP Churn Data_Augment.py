# Databricks notebook source
# MAGIC %md
# MAGIC  
# MAGIC # IFP Data for CatBoost Model

# COMMAND ----------

import pandas as pd
import numpy as np
import pyspark.pandas as ps
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, count, when, rand, isnan, year, month, expr
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors
from math import radians, sin, cos, sqrt, atan2
from collections import Counter

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COALESCE(churn.open_enrollment_survival_dt,0) AS survival_flag, '2021' AS open_enrollment_cycle, m.*
# MAGIC FROM ec_sbx_adv_anltcs_prd1.mbrdna_prd.tbl_tgt_mbrdna_dnrmlzd_wide m
# MAGIC LEFT JOIN ( --self join to grab churn flag from future
# MAGIC       SELECT MEMBER_ID AS join_key, max(1) as open_enrollment_survival_dt
# MAGIC       FROM ec_sbx_adv_anltcs_prd1.mbrdna_prd.tbl_tgt_mbrdna_dnrmlzd_wide
# MAGIC       --WHERE MARKETS_LOB LIKE '%IFP%' --include members who switched plans
# MAGIC       WHERE STUDY_END_DT = '2022-01-31' --grab future study period
# MAGIC       --ensure elig is active following open enrollment
# MAGIC       AND FIRST_ELIG_MONTH_START_DT <= '2022-01-01'
# MAGIC       AND LAST_ELIG_MONTH_END_DT >= '2022-01-01'
# MAGIC       GROUP BY 1
# MAGIC       ) AS churn ON churn.join_key = m.MEMBER_ID  
# MAGIC WHERE MARKETS_LOB LIKE '%IFP%'
# MAGIC AND STUDY_END_DT = '2021-12-31'
# MAGIC --AND LAST_ELIG_MONTH_END_DT > '2021-12-31'
# MAGIC UNION
# MAGIC SELECT COALESCE(churn.open_enrollment_survival_dt,0) AS survival_flag, '2022' AS open_enrollment_cycle, m.*
# MAGIC FROM ec_sbx_adv_anltcs_prd1.mbrdna_prd.tbl_tgt_mbrdna_dnrmlzd_wide m
# MAGIC LEFT JOIN ( --self join to grab churn flag from future
# MAGIC       SELECT MEMBER_ID AS join_key, max(1) as open_enrollment_survival_dt
# MAGIC       FROM ec_sbx_adv_anltcs_prd1.mbrdna_prd.tbl_tgt_mbrdna_dnrmlzd_wide
# MAGIC       --WHERE MARKETS_LOB LIKE '%IFP%' --include members who switched plans
# MAGIC       WHERE STUDY_END_DT = '2023-01-31' --grab future study period
# MAGIC       --ensure elig is active following open enrollment
# MAGIC       AND FIRST_ELIG_MONTH_START_DT <= '2023-01-01'
# MAGIC       AND LAST_ELIG_MONTH_END_DT >= '2023-01-01'
# MAGIC       GROUP BY 1
# MAGIC       ) AS churn ON churn.join_key = m.MEMBER_ID  
# MAGIC WHERE MARKETS_LOB LIKE '%IFP%'
# MAGIC AND STUDY_END_DT = '2022-12-31'
# MAGIC --AND LAST_ELIG_MONTH_END_DT > '2022-12-31'
# MAGIC UNION
# MAGIC SELECT COALESCE(churn.open_enrollment_survival_dt,0) AS survival_flag, '2023' AS open_enrollment_cycle, m.*
# MAGIC FROM ec_sbx_adv_anltcs_prd1.mbrdna_prd.tbl_tgt_mbrdna_dnrmlzd_wide m
# MAGIC LEFT JOIN ( --self join to grab churn flag from future
# MAGIC       SELECT MEMBER_ID AS join_key, max(1) as open_enrollment_survival_dt
# MAGIC       FROM ec_sbx_adv_anltcs_prd1.mbrdna_prd.tbl_tgt_mbrdna_dnrmlzd_wide
# MAGIC       --WHERE MARKETS_LOB LIKE '%IFP%' --include members who switched plans
# MAGIC       WHERE STUDY_END_DT = '2024-01-31' --grab future study
# MAGIC       --ensure elig is active following open enrollment
# MAGIC       AND FIRST_ELIG_MONTH_START_DT <= '2024-01-01'
# MAGIC       AND LAST_ELIG_MONTH_END_DT >= '2024-01-01'
# MAGIC       GROUP BY 1
# MAGIC       ) AS churn ON churn.join_key = m.MEMBER_ID  
# MAGIC WHERE MARKETS_LOB LIKE '%IFP%'
# MAGIC AND STUDY_END_DT = '2023-12-31'
# MAGIC --AND LAST_ELIG_MONTH_END_DT > '2023-12-31'
# MAGIC UNION
# MAGIC SELECT COALESCE(churn.open_enrollment_survival_dt,0) AS survival_flag, '2024' AS open_enrollment_cycle, m.*
# MAGIC FROM ec_sbx_adv_anltcs_prd1.mbrdna_prd.tbl_tgt_mbrdna_dnrmlzd_wide m
# MAGIC LEFT JOIN ( --self join to grab churn flag from future
# MAGIC       SELECT MEMBER_ID AS join_key, max(1) as open_enrollment_survival_dt
# MAGIC       FROM ec_sbx_adv_anltcs_prd1.mbrdna_prd.tbl_tgt_mbrdna_dnrmlzd_wide
# MAGIC       --WHERE MARKETS_LOB LIKE '%IFP%' --include members who switched plans
# MAGIC       WHERE STUDY_END_DT = '2025-01-31' --grab future study
# MAGIC       --ensure elig is active following open enrollment
# MAGIC       AND FIRST_ELIG_MONTH_START_DT <= '2025-01-01'
# MAGIC       AND LAST_ELIG_MONTH_END_DT >= '2025-01-01'
# MAGIC       GROUP BY 1
# MAGIC       ) AS churn ON churn.join_key = m.MEMBER_ID  
# MAGIC WHERE MARKETS_LOB LIKE '%IFP%'
# MAGIC AND STUDY_END_DT = '2024-12-31'
# MAGIC --AND LAST_ELIG_MONTH_END_DT > '2024-12-31'

# COMMAND ----------

_sqldf.write.mode("overwrite").saveAsTable('dev_adb.ifp_churn.data_new')

# COMMAND ----------

# MAGIC %sql
# MAGIC select open_enrollment_cycle, survival_flag,count(*) 
# MAGIC from dev_adb.ifp_churn.data_new
# MAGIC group by 1,2
# MAGIC order by 1,2

# COMMAND ----------

from pyspark.sql import functions as F

# Aggregate churned and total users per year
df_churn = spark.sql(f'''select * from dev_adb.ifp_churn.data_new''').groupBy("open_enrollment_cycle").agg(
    F.sum(F.when(F.col("survival_flag") == 0, 1).otherwise(0)).alias("churned_users"),
    F.count("*").alias("total_users")
)

# Compute churn rate
df_churn = df_churn.withColumn("churn_rate", (F.col("churned_users") / F.col("total_users")) * 100)

# Convert to Pandas for visualization
df_churn_pd = df_churn.toPandas()

import matplotlib.pyplot as plt

# Sort by year
df_churn_pd = df_churn_pd.sort_values(by="open_enrollment_cycle")

# Plot churn rate
plt.figure(figsize=(10, 5))
plt.bar(df_churn_pd["open_enrollment_cycle"], df_churn_pd["churn_rate"], color="red", alpha=0.7)

# Labels and title
plt.xlabel("Year")
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate Per Year")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Show the plot
plt.show()

# COMMAND ----------

# Compute overall churn rate
average_churn_rate = df_churn_pd["churn_rate"].mean()

print(f"Average Churn Rate: {average_churn_rate:.2f}%")

# COMMAND ----------

df = spark.sql(f'''select * from dev_adb.ifp_churn.data_new''').withColumn("churn_flag", expr("1 - survival_flag"))
call_intent = spark.sql(f'''select * from dbc_adv_anlaytics_dev.surveyspeechextraction.ifp_work_call_intent''')
income = spark.sql(f'''select * from prd_dbc_curated.mysidewalk.tbl_gold_msw_zip_code_income''')
poverty = spark.sql(f'''select * from prd_dbc_curated.mysidewalk.tbl_gold_msw_zip_code_poverty''')
provider_ipa = spark.sql(f'''select * from prd_dbc_curated.provider.tbl_gold_ipa''')
provider_group = spark.sql(f'''select * from prd_dbc_curated.provider.tbl_gold_provider_group''')
provider = provider_ipa.unionByName(provider_group)

df = df.join(call_intent, ["MEMBER_ID"], "left")
df = df.join(income, ["ZIPCODE"], "left")
df = df.join(poverty, ["ZIPCODE"], "left")
df = df.join(provider, df["PROV_IPA_ID"] == provider_ipa["FACETS_PROV_ID"], "left")

# Add provider_flag column (1 if PROV_IPA_ID is not null, else 0)
df = df.withColumn("provider_flag", F.when(F.col("PROV_IPA_ID").isNotNull(), 1).otherwise(0))

# COMMAND ----------

# Check the Shape of the DataFrame
num_rows = df.count()
num_cols = len(df.columns)
print(f"Original Shape of DataFrame: ({num_rows}, {num_cols})")

# Identify Duplicate Columns
column_counts = Counter(df.columns)
duplicate_columns = [col for col, count in column_counts.items() if count > 1]

# Drop Redundant Duplicate Columns
columns_to_drop = [col for col in duplicate_columns]

df1 = df.drop(*columns_to_drop)

# Check the New Shape After Removing Duplicates
num_rows = df1.count()
num_cols = len(df1.columns)
print(f"Shape after merging and cleaning: ({num_rows}, {num_cols})")

# COMMAND ----------

# Identify Numerical and Categorical Columns
numeric_cols = [col_name for col_name, dtype in df1.dtypes if dtype in ('int', 'double', 'float') or dtype.startswith('decimal')]
categorical_cols = [col_name for col_name, dtype in df1.dtypes if dtype == 'string']

# Convert Decimal Columns to Double for Imputation
for col_name, dtype in df1.dtypes:
    if dtype.startswith("decimal"):
        df1 = df1.withColumn(col_name, col(col_name).cast("double"))

# Fill NoneType values with a placeholder before filtering
df1 = df1.fillna({col_name: -1 for col_name in numeric_cols})

# Filter out columns that contain only null or NaN values
non_null_numeric_cols = [
    col_name for col_name in numeric_cols 
    if df1.select(col_name)
          .filter(~col(col_name).isNull() & ~isnan(col(col_name)))
          .count() > 0
]

# Handle Missing Values for Numerical Columns WITHOUT Renaming Columns
num_imputer = Imputer(strategy="median", inputCols=non_null_numeric_cols, outputCols=non_null_numeric_cols) 
df2 = num_imputer.fit(df1).transform(df1)

# Handle Missing Values for Categorical Columns WITHOUT Encoding (CatBoost can handle them as-is)
df2 = df2.fillna({col_name: "missing" for col_name in categorical_cols})

# COMMAND ----------

df2.write.mode("overwrite").saveAsTable('dev_adb.ifp_churn.data_augment_preprocessed')