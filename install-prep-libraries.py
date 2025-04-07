# Databricks notebook source
# Potentially used: unstructured[local-inference] 

# COMMAND ----------

# MAGIC %pip install -U transformers 
# MAGIC %pip install --upgrade sentence-transformers
# MAGIC %pip install --upgrade mlflow
# MAGIC %pip install --upgrade langchain
# MAGIC %pip install --upgrade databricks-vectorsearch==0.45 
# MAGIC %pip install --upgrade pycryptodome
# MAGIC %pip install --upgrade accelerate
# MAGIC %pip install --upgrade sacremoses
# MAGIC %pip install --upgrade ninja
# MAGIC %pip install --upgrade tiktoken
# MAGIC %pip install --upgrade nltk
# MAGIC %pip install --upgrade databricks
# MAGIC %pip install --upgrade databricks-agents
# MAGIC %pip install --upgrade databricks-feature-engineering==0.8.1a2
# MAGIC %pip install --upgrade pypdf
# MAGIC %pip install --upgrade langchain_community
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import nltk
nltk.download('punkt')