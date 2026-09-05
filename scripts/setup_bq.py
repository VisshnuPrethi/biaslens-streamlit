#!/usr/bin/env python
"""
BigQuery Setup Script for BiasLens.
Creates the dataset and table schema for storing loan application test pairs.
"""
import os
import sys
from google.cloud import bigquery
from google.api_core.exceptions import NotFound

# Load environment variables if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def setup_bigquery():
    # 1. Load project configurations
    project_id = os.environ.get("GCP_PROJECT_ID")
    dataset_id = os.environ.get("BQ_DATASET_ID", "biaslens")
    table_id = os.environ.get("BQ_TABLE_ID", "loan_applications")

    print("Initializing BigQuery Client...")
    try:
        # If project_id is provided, bind it to client, otherwise rely on default credentials environment config.
        if project_id:
            client = bigquery.Client(project=project_id)
        else:
            client = bigquery.Client()
            project_id = client.project
        
        print(f"Using Google Cloud Project ID: {project_id}")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize BigQuery Client. Details:\n{e}", file=sys.stderr)
        print("\nPlease make sure that:", file=sys.stderr)
        print("1. GCP_PROJECT_ID is correctly configured in your environment or .env file.", file=sys.stderr)
        print("2. Google application credentials are set up (e.g. via GOOGLE_APPLICATION_CREDENTIALS environment variable or running `gcloud auth application-default login`).", file=sys.stderr)
        sys.exit(1)

    # 2. Check/Create Dataset
    dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
    try:
        dataset = client.get_dataset(dataset_ref)
        print(f"Dataset '{dataset_id}' already exists.")
    except NotFound:
        print(f"Dataset '{dataset_id}' not found. Creating dataset...")
        dataset = bigquery.Dataset(dataset_ref)
        # Default to US multi-region if not specified, but check environment
        dataset.location = os.environ.get("BQ_LOCATION", "US")
        try:
            dataset = client.create_dataset(dataset, timeout=30)
            print(f"Dataset '{dataset_id}' successfully created in location '{dataset.location}'.")
        except Exception as e:
            print(f"Failed to create dataset '{dataset_id}': {e}", file=sys.stderr)
            sys.exit(1)

    # 3. Define Table Schema
    schema = [
        bigquery.SchemaField("eval_run_id", "STRING", mode="REQUIRED", description="Unique ID for the audit run"),
        bigquery.SchemaField("pair_id", "STRING", mode="REQUIRED", description="Unique ID linking control & counterfactual cases together"),
        bigquery.SchemaField("case_type", "STRING", mode="REQUIRED", description="Whether this is a control or counterfactual case"),
        bigquery.SchemaField("demographic_attribute", "STRING", mode="REQUIRED", description="Attribute tested (e.g., gender, race, age)"),
        bigquery.SchemaField("demographic_value", "STRING", mode="REQUIRED", description="Value of demographic attribute (e.g., male, female)"),
        bigquery.SchemaField("credit_score", "INTEGER", mode="REQUIRED", description="Non-demographic attribute: Credit score of applicant"),
        bigquery.SchemaField("income", "NUMERIC", mode="REQUIRED", description="Non-demographic attribute: Annual income"),
        bigquery.SchemaField("loan_amount", "NUMERIC", mode="REQUIRED", description="Non-demographic attribute: Requested loan amount"),
        bigquery.SchemaField("employment_status", "STRING", mode="NULLABLE", description="Non-demographic attribute: Employment status"),
        bigquery.SchemaField("prompt", "STRING", mode="REQUIRED", description="Full prompt text sent to the LLM agent"),
        bigquery.SchemaField("raw_response", "STRING", mode="NULLABLE", description="Raw text response returned by the LLM agent"),
        bigquery.SchemaField("approved", "BOOLEAN", mode="NULLABLE", description="Extracted decision (True/False)"),
        bigquery.SchemaField("confidence_score", "FLOAT", mode="NULLABLE", description="Confidence or probability score if available"),
        bigquery.SchemaField("model_name", "STRING", mode="REQUIRED", description="Name/version of the Gemini model used"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED", description="Timestamp of evaluation"),
    ]

    # 4. Check/Create Table
    table_ref = dataset_ref.table(table_id)
    try:
        table = client.get_table(table_ref)
        print(f"Table '{table_id}' already exists in dataset '{dataset_id}'.")
        
        # Verify schema match or advise
        existing_fields = {field.name for field in table.schema}
        new_fields = {field.name for field in schema}
        missing_fields = new_fields - existing_fields
        if missing_fields:
            print(f"Warning: Existing table is missing fields: {missing_fields}. You may want to drop or migrate the table.")
    except NotFound:
        print(f"Table '{table_id}' not found. Creating table with defined schema...")
        table = bigquery.Table(table_ref, schema=schema)
        try:
            table = client.create_table(table, timeout=30)
            print(f"Table '{table_id}' successfully created in dataset '{dataset_id}'.")
        except Exception as e:
            print(f"Failed to create table '{table_id}': {e}", file=sys.stderr)
            sys.exit(1)

    print("\nBigQuery setup completed successfully!")


if __name__ == "__main__":
    setup_bigquery()
