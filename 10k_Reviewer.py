# Databricks notebook source
# MAGIC %md
# MAGIC ## Document Ingestion and Preparation
# MAGIC
# MAGIC <img style="float: right" width="800px" src="https://raw.githubusercontent.com/databricks-industry-solutions/hls-llm-doc-qa/basic-qa-LLM-HLS/images/data-prep.jpeg?token=GHSAT0AAAAAACBNXSB4IK2XJS37QU6HCJCEZEBL3TA">
# MAGIC
# MAGIC
# MAGIC #
# MAGIC 1. Organize your documents into a Unity Catalog Volume
# MAGIC     * In this demo we have preuploaded a set of PDFs from PubMed on S3, but your own documents will work the same way
# MAGIC 2. Use LangChain to ingest those documents and split them into manageable chunks using a text splitter
# MAGIC 3. Use a sentence transformer NLP model to create embeddings of those text chunks and store them in a vectorstore
# MAGIC     * Embeddings are basically creating a high-dimension vector encoding the semantic meaning of a chunk of text
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Start with required Python libraries for data preparation.

# COMMAND ----------

import os, sys, json

# COMMAND ----------

# MAGIC %run ./util/install-prep-libraries

# COMMAND ----------

# MAGIC %md
# MAGIC Creating a dropdown widget for model selection, as well as defining the file paths where our PDFs are stored, where we want to cache the HuggingFace model downloads, and where we want to persist our vectorstore.

# COMMAND ----------

# where you want the PDFs to be saved in your environment
#dbutils.widgets.text("UC_Volume_Path", "hls_llm_qa_demo.data.pdf_docs")
dbutils.widgets.text("UC_Volume_Path", "/Volumes/dev_adb/yh_agent_resume/data_volume")

# which embeddings model we want to use. We are going to use the foundation model API, but you can use custom models (i.e. from HuggingFace), External Models (Azure OpenAI), etc.
dbutils.widgets.text("Embeddings_Model", "databricks-bge-large-en")

# publicly accessible bucket with PDFs for this demo
#dbutils.widgets.text("Source_Documents", "s3a://db-gtm-industry-solutions/data/hls/llm_qa/")
dbutils.widgets.text("Source_Documents", "/Workspace/Users/yhu01@blueshieldca.com/adb_rag_resume/data")

# Location for the split documents to be saved  
#dbutils.widgets.text("Persisted_UC_Table_Location", "hls_llm_qa_demo.vse.hls_llm_qa_raw_docs")
dbutils.widgets.text("Persisted_UC_Table_Location", "dev_adb.yh_agent_resume.data_chunked")

# Vector Search Endpoint Name - , hls_llm_qa_demo_vse
#dbutils.widgets.text("Vector_Search_Endpoint", "VS_ENDPOINT")
dbutils.widgets.text("Vector_Search_Endpoint", "yhu01_resume_search")

# Vector Index Name 
#dbutils.widgets.text("Vector_Index", "hls_llm_qa_demo.vse.hls_llm_qa_embeddings")
dbutils.widgets.text("Vector_Index", "dev_adb.yh_agent_resume.data_chunked_index")

# Target Catalog Name
#dbutils.widgets.text("Catalog_Name", "hls_llm_qa_demo")
dbutils.widgets.text("Catalog_Name", "dev_adb")

# Target VSE Schema Name
dbutils.widgets.text("Vse_Schema_Name", "yh_agent_resume")

# Target UC_CATALOG Schema Name
dbutils.widgets.text("UC_CATALOG", "dev_adb")

# Target UC_SCHEMA Name
dbutils.widgets.text("UC_SCHEMA", "yh_agent_resume")

# Target CHUNKS_DELTA_TABLE Name
dbutils.widgets.text("CHUNKS_DELTA_TABLE", "data_chunked")



# COMMAND ----------

#get widget values
pdf_path = dbutils.widgets.get("UC_Volume_Path")
source_pdfs = dbutils.widgets.get("Source_Documents")
embeddings_model = dbutils.widgets.get("Embeddings_Model")
vector_search_endpoint_name = dbutils.widgets.get("Vector_Search_Endpoint")
vector_index_name = dbutils.widgets.get("Vector_Index")
UC_table_save_location = dbutils.widgets.get("Persisted_UC_Table_Location")
Persisted_UC_Table_Location = dbutils.widgets.get("Persisted_UC_Table_Location")
Vse_Schema_Name = dbutils.widgets.get("Vse_Schema_Name")
UC_CATALOG = dbutils.widgets.get("UC_CATALOG")
UC_SCHEMA = dbutils.widgets.get("UC_SCHEMA")
CHUNKS_DELTA_TABLE = dbutils.widgets.get("CHUNKS_DELTA_TABLE")


# TEMPORARY - NEED TO ADD STRING LOGIC TO GENERATE:
volume_path = pdf_path

# COMMAND ----------

print(f"pdf_path = {pdf_path}")
print(f"source_pdfs = {source_pdfs}")
print(f"embeddings_model = {embeddings_model}")
print(f"vector_search_endpoint_name = {vector_search_endpoint_name}")
print(f"vector_index_name = {vector_index_name}")
print(f"UC_table_save_location = {UC_table_save_location}")
print(f"Persisted_UC_Table_Location = {Persisted_UC_Table_Location}")
print(f"Vse_Schema_Name = {Vse_Schema_Name}")
print(f"volume_path = {volume_path}")
print(f"UC_CATALOG = {UC_CATALOG}")
print(f"UC_SCHEMA = {UC_SCHEMA}")
print(f"CHUNKS_DELTA_TABLE = {CHUNKS_DELTA_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC Create Unity schema if it does not exist

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create Unity schema if it does not exist in the Unity catalog
# MAGIC -- Use IF NOT EXISTS clause to avoid errors if the schema already exists
# MAGIC CREATE SCHEMA IF NOT EXISTS ${Catalog_Name}.yh_agent_resume

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Prep
# MAGIC
# MAGIC This data preparation need only happen one time to create data sets that can then be reused in later sections without re-running this part.
# MAGIC
# MAGIC - Grab the set of PDFs (ex: Arxiv papers allow curl, PubMed does not)
# MAGIC - We have are providing a set of PDFs from PubMedCentral relating to Cystic Fibrosis (all from [PubMedCentral Open Access](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/), all with the CC BY license), but any topic area would work
# MAGIC - If you already have a repository of PDFs then you can skip this step, just organize them all in an accessible Unity Catalog Volume

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create an external volume under the newly created directory
# MAGIC CREATE SCHEMA IF NOT EXISTS ${Catalog_Name}.${Vse_Schema_Name}

# COMMAND ----------

source_pdfs

# COMMAND ----------

print('file:'+source_pdfs)

# COMMAND ----------

volume_path

# COMMAND ----------

# MAGIC %md
# MAGIC copy the files to the volume

# COMMAND ----------

dbutils.fs.ls(f'file:'+source_pdfs)

# COMMAND ----------

# MAGIC %md
# MAGIC clean up the volume

# COMMAND ----------

# Source directory in workspace
source_dir = 'file:'+source_pdfs

# Destination volume path
destination_volume = volume_path

#clean up the volume

destination_dir = volume_path
# List files in the source directory
files = dbutils.fs.ls(destination_dir)

# Remove eachj file in the volume
for file_info in files:
    source_path = file_info.path
    destination_path = f"{destination_volume}/{file_info.name}"
    dbutils.fs.rm(destination_path)

# COMMAND ----------

#Move the documents to volume on unity catalog

# List files in the source directory
files = dbutils.fs.ls(source_dir)

# Copy each file to the volume
for file_info in files:
    source_path = file_info.path
    destination_path = f"{destination_volume}/{file_info.name}"
    dbutils.fs.cp(source_path, destination_path)

# COMMAND ----------

# MAGIC %md
# MAGIC All of the PDFs should now be accessible in the `Unity Catalog Volume` now; you can run the below command to check if you want.
# MAGIC
# MAGIC `dbutils.fs.ls(volume_path)`

# COMMAND ----------

dbutils.fs.ls(volume_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Document DB
# MAGIC
# MAGIC Now it's time to load the texts that have been generated, and create a searchable database of text for use in the `langchain` pipeline. 
# MAGIC These documents are embedded, so that later queries can be embedded too, and matched to relevant text chunks by embedding.
# MAGIC
# MAGIC - Use `langchain` to reading directly from PDFs, although LangChain also supports txt, HTML, Word docs, GDrive, PDFs, etc.
# MAGIC - Create a Databricks Vector Search endpoint to have a persistent vector index.
# MAGIC - Use the Foundation Model APIs to generate the embeddings to sync against the vector index.
# MAGIC - Sync the vector index to populate for our rag implementation.

# COMMAND ----------

# MAGIC %md
# MAGIC Create the document database:
# MAGIC - Here we are using the `PyPDFDirectoryLoader` loader from LangChain ([docs page](https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/pdf.html#using-pypdf)) to form `documents`; `langchain` can also form doc collections directly from PDFs, GDrive files, etc.

# COMMAND ----------

pdf_path

# COMMAND ----------

# Load the volume of pdf's into a list of text.  

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFDirectoryLoader

# Load directly from Unity Catalog Volumes
loader_path = volume_path

pdf_loader = PyPDFDirectoryLoader(loader_path)
docs = pdf_loader.load()
len(docs)

# COMMAND ----------

loader_path

# COMMAND ----------

type(docs[0])

# COMMAND ----------

# MAGIC %md
# MAGIC Here we are using a text splitter from LangChain to split our PDFs into manageable chunks. This is for a few reasons, primarily:
# MAGIC - LLMs (currently) have a limited context length. DRBX by default has a context length of 32k tokens. tokens (roughly words) in the prompt.
# MAGIC - When we create embeddings for these documents, an NLP model (sentence transformer) creates a numerical representation (a high-dimensional vector) of that chunk of text that captures the semantic meaning of what is being embedded. If we were to embed large documents, the NLP model would need to capture the meaning of the entire document in one vector; by splitting the document, we can capture the meaning of chunks throughout that document and retrieve only what is most relevant.
# MAGIC - In this case, the embeddings model we use can except a very limited number of tokens. 
# MAGIC - More info on embeddings: [Hugging Face: Getting Started with Embeddings](https://huggingface.co/blog/getting-started-with-embeddings)

# COMMAND ----------

# For PDFs we need to split them for embedding:
from langchain.text_splitter import TokenTextSplitter

# this is splitting into chunks based on a fixed number of tokens
# the embeddings model we use below can take a maximum of 128 tokens (and truncates beyond that) so we keep our chunks at that max size
text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=32)
documents = text_splitter.split_documents(docs)

# COMMAND ----------

# MAGIC %md
# MAGIC Drop the chunked table
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create Unity schema if it does not exist in the Unity catalog
# MAGIC -- Use IF NOT EXISTS clause to avoid errors if the schema already exists
# MAGIC drop table IF EXISTS dev_adb.yh_agent_resume.data_chunked

# COMMAND ----------

# MAGIC %md
# MAGIC Load the pandas data into a spark dataframe and then write to the chunked table

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from pyspark.sql.functions import monotonically_increasing_id

# Workspace URL for printing links to the delta table/vector index
workspace_url = spark.conf.get("spark.databricks.workspaceUrl", None)

# Vector Search client
vsc = VectorSearchClient(disable_notice=True)

from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql import Row

# Define the schema explicitly
schema = StructType([
    StructField("text", StringType(), True)
])

# Assuming 'docs' is a list of strings
rows = [Row(text=doc) for doc in documents]

# Create a DataFrame with the defined schema
chunked_docs_df = spark.createDataFrame(rows, schema)

# Rename columns and add an 'id' column for the primary key
documents_with_id = chunked_docs_df.withColumnRenamed("text", "chunked_text") \
                            .withColumn("chunk_id", monotonically_increasing_id())

# Proceed with your operations on chunked_docs_df

# Now you can proceed with saving chunked_docs_df to a Delta table for further processing
documents_with_id.write.format("delta").mode("overwrite").saveAsTable("dev_adb.yh_agent_resume.data_chunked")

# Enable change-data capture for the Delta table
spark.sql("ALTER TABLE dev_adb.yh_agent_resume.data_chunked SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

print(f"View Delta Table at: https://{workspace_url}/explore/data/dev_adb/yh_agent_resume/data_chunked")

# COMMAND ----------


display(documents_with_id)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we split the documents into more manageable chunks. We will now set up **Databricks Vector Search** with a **Direct Vector Access Index** which will be used with Langchain in our RAG architecture. 
# MAGIC - We first need to create a dataframe with an id column to be used with Vector Search.
# MAGIC - We will then calculate the embeddings using BGE
# MAGIC - Finally we will save this in our Vector Search as an index to be used for RAG.

# COMMAND ----------

UC_CATALOG = f'dev_adb'
UC_SCHEMA = f'yh_agent_resume'

# COMMAND ----------



CHUNKS_DELTA_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.data_chunked"
CHUNKS_VECTOR_INDEX = f"{UC_CATALOG}.{UC_SCHEMA}.data_chunked_index"

# COMMAND ----------

VECTOR_SEARCH_ENDPOINT = vector_search_endpoint_name

# COMMAND ----------

CHUNKS_VECTOR_INDEX

# COMMAND ----------

# MAGIC %md
# MAGIC This is setting up the catalog for the chunked data as well as establishes the Vector Search endpoint

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointStatusState, EndpointType
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointStateReady
from databricks.sdk.errors import ResourceDoesNotExist, NotFound, PermissionDenied
import os
w = WorkspaceClient()

# Create UC Catalog if it does not exist, otherwise, raise an exception
try:
    _ = w.catalogs.get(UC_CATALOG)
    print(f"PASS: UC catalog `{UC_CATALOG}` exists")
except NotFound as e:
    print(f"`{UC_CATALOG}` does not exist, trying to create...")
    try:
        _ = w.catalogs.create(name=UC_CATALOG)
    except PermissionDenied as e:
        print(f"FAIL: `{UC_CATALOG}` does not exist, and no permissions to create.  Please provide an existing UC Catalog.")
        raise ValueError(f"Unity Catalog `{UC_CATALOG}` does not exist.")
        
# Create UC Schema if it does not exist, otherwise, raise an exception
try:
    _ = w.schemas.get(full_name=f"{UC_CATALOG}.{UC_SCHEMA}")
    print(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` exists")
except NotFound as e:
    print(f"`{UC_CATALOG}.{UC_SCHEMA}` does not exist, trying to create...")
    try:
        _ = w.schemas.create(name=UC_SCHEMA, catalog_name=UC_CATALOG)
        print(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` created")
    except PermissionDenied as e:
        print(f"FAIL: `{UC_CATALOG}.{UC_SCHEMA}` does not exist, and no permissions to create.  Please provide an existing UC Schema.")
        raise ValueError("Unity Catalog Schema `{UC_CATALOG}.{UC_SCHEMA}` does not exist.")

# Create the Vector Search endpoint if it does not exist
vector_search_endpoints = w.vector_search_endpoints.list_endpoints()
if sum([VECTOR_SEARCH_ENDPOINT == ve.name for ve in vector_search_endpoints]) == 0:
    print(f"Please wait, creating Vector Search endpoint `{VECTOR_SEARCH_ENDPOINT}`.  This can take up to 20 minutes...")
    w.vector_search_endpoints.create_endpoint_and_wait(VECTOR_SEARCH_ENDPOINT, endpoint_type=EndpointType.STANDARD)

# Make sure vector search endpoint is online and ready.
w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(VECTOR_SEARCH_ENDPOINT)

print(f"PASS: Vector Search endpoint `{VECTOR_SEARCH_ENDPOINT}` exists")

# COMMAND ----------

CHUNKS_DELTA_TABLE

# COMMAND ----------

# MAGIC %md
# MAGIC THIS is the end that has written to the chunk table now we need to run the code to create the index

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC 1/ Create the Vector Search Index
# MAGIC First, we copy the sample data to a Delta Table and sync to a Vector Search index. Here, we use the gte-large-en-v1.5 embedding model hosted on Databricks Foundational Model APIs.

# COMMAND ----------


index_name=vector_index_name
index_name

# Attempt to delete the existing index if it exists
try:
    vsc.delete_index(index_name=index_name)
    print(f"Existing index '{index_name}' deleted successfully.")
except Exception as e:
    print(f"No existing index named '{index_name}' was found or an error occurred: {e}")


# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

# Embed and sync chunks to a vector index
print(
    f"Embedding docs & creating Vector Search Index, this will take ~5 - 10 minutes.\nView Index Status at: https://{workspace_url}/explore/data/{UC_CATALOG}/{UC_SCHEMA}/{CHUNKS_VECTOR_INDEX.split('.')[-1]}"
)

# Initialize the Vector Search client
vsc = VectorSearchClient()

# Define the index name and other parameters
index_name = CHUNKS_VECTOR_INDEX
source_table_name = CHUNKS_DELTA_TABLE
primary_key_column = "chunk_id"  # Adjust based on your table's primary key
embedding_column = "chunked_text"  # Adjust based on your embedding column



# Proceed to create the Vector Search index without invoking serverless compute
# This example creates a Delta Sync Index with embeddings computed by Databricks
try:
    index = vsc.create_delta_sync_index_and_wait(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=index_name,
        primary_key=primary_key_column,
        source_table_name=source_table_name,
        pipeline_type="TRIGGERED",  # Using TRIGGERED to avoid continuous compute costs
        embedding_source_column=embedding_column,
        embedding_model_endpoint_name="databricks-gte-large-en",  # Specify your model endpoint
    )
    print(f"Vector Search index '{index_name}' created successfully.")
except Exception as e:
    print(f"Error creating Vector Search index '{index_name}': {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC 2/ Deploy to the review application
# MAGIC
# MAGIC Now that our Vector Search index is ready, let's prepare the RAG chain and deploy it to the review application backed by a scalable-production ready REST API on Model serving.

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1/ Configuring our Chain parameters
# MAGIC Databricks makes it easy to parameterize your chain with MLflow Model Configurations. Later, you can tune application quality by adjusting these parameters, such as the system prompt or retrieval settings. Most applications will include many more parameters, but for this demo, we'll keep the configuration to a minimum.

# COMMAND ----------

chain_config = {
    "llm_model_serving_endpoint_name": "poc-openai-completions-endpoint",  # the foundation model we want to use
    "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT,  # Endoint for vector search
    "vector_search_index": f"{CHUNKS_VECTOR_INDEX}",
    "llm_prompt_template": """You are an investment manager. Use the following files to answer the question. \n\nContext: {context}""", # LLM Prompt template
}

# Here, we define an input example in the schema required by Agent Framework
input_example = {"messages": [ {"role": "user", "content": "Which stock performs best?"}]}

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1/ Log the application & view trace
# MAGIC We first register the chain as an MLflow model and inspect the MLflow Trace to understand what is happening inside the chain.
# MAGIC
# MAGIC MLflow trace

# COMMAND ----------

import mlflow


# COMMAND ----------

mlflow.__version__

# COMMAND ----------

QUICK_START_REPO_SAVE_FOLDER = "genai-resume"

import os
os.getcwd()
lc_model=os.path.join(
            os.getcwd(),
            f"rag_langchain_runner",
        ),  # Chain code file from the quick start repo

# COMMAND ----------

lc_model

# COMMAND ----------

import mlflow

# Log the model to MLflow
with mlflow.start_run(run_name="databricks-docs-bot"):
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(),
            f"rag_langchain_runner",
        ),  # Chain code file from the quick start repo
        model_config=chain_config,  # Chain configuration set above
        artifact_path="chain",  # Required by MLflow
        input_example=input_example,  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
    )

# Test the chain locally to see the MLflow Trace
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(input_example)

# COMMAND ----------

chain_config = {
    "llm_model_serving_endpoint_name": "poc-openai-completions-endpoint",  # the foundation model we want to use
    "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT,  # Endoint for vector search
    "vector_search_index": f"{CHUNKS_VECTOR_INDEX}",
    "llm_prompt_template": """you are a portfolio manager. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.\n\nContext: {context}""", # LLM Prompt template
}

# Here, we define an input example in the schema required by Agent Framework
input_example = {"messages": [ {"role": "user", "content": "which stock has the highest information value?"}]}

# COMMAND ----------

# Test the chain locally to see the MLflow Trace
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(input_example)

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1/ Deploy the application
# MAGIC Now, we:
# MAGIC
# MAGIC Register the application in Unity Catalog
# MAGIC Use Agent Framework to deploy to the Quality Lab review application
# MAGIC Along side the review ap, a scalable, production-ready Model Serving endpoint is also deployed.

# COMMAND ----------

# Use the current user name to create any necesary resources
w = WorkspaceClient()
user_name = w.current_user.me().user_name.split("@")[0].replace(".", "")

# UC Catalog & Schema where outputs tables/indexs are saved
# If this catalog/schema does not exist, you need create catalog/schema permissions.
UC_CATALOG = f'dev_adb'
UC_SCHEMA = f'yh_agent_resume'

# UC Model name where the POC chain is logged
UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.chat_bot"

# Vector Search endpoint where index is loaded
# If this does not exist, it will be created
VECTOR_SEARCH_ENDPOINT = f'{user_name}_resume_search'

# COMMAND ----------

from databricks import agents

# COMMAND ----------

UC_MODEL_NAME

# COMMAND ----------

from databricks import agents
import time
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate

# Use Unity Catalog to log the chain
mlflow.set_registry_uri('databricks-uc')

# Register the chain to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=UC_MODEL_NAME)

# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(model_name=UC_MODEL_NAME, model_version=uc_registered_model_info.version)

# Wait for the Review App to be ready
print("\nWaiting for endpoint to deploy.  This can take 10 - 20 minutes.", end="")
while w.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY or w.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
    print(".", end="")
    time.sleep(30)