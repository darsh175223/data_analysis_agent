"""Data ingestion module for retrieving metrics from Hadoop or Hive,
processing them via Apache Spark, and returning results as JSON.

This script uses placeholder connection details and sample ETL steps. Replace
placeholders with real configurations in production.
"""
from typing import List, Dict
import json

from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

# ---------------------------------------------------------------------------
# Mapping of business KPIs to the preferred storage layer.
# ---------------------------------------------------------------------------
# "hadoop"  -> Stored in a data lake (raw, high-volume, semi-structured data)
# "hive"    -> Stored in a data warehouse (curated, structured, query-ready data)
metric_storage_map: Dict[str, str] = {
    "customer_churn_rate": "hive",
    "daily_active_users": "hadoop",
    "avg_session_length": "hadoop",
    "revenue": "hive",
    "cost_of_goods_sold": "hive",
    "inventory_level": "hive",
    "click_through_rate": "hadoop",
    "conversion_rate": "hive",
    "net_promoter_score": "hive",
    "bounce_rate": "hadoop",
    "time_to_resolution": "hive",
    "support_ticket_volume": "hadoop",
    "marketing_spend": "hive",
    "email_open_rate": "hadoop",
    "page_views": "hadoop",
    "average_order_value": "hive",
    "refund_rate": "hive",
    "server_response_time": "hadoop",
    "subscription_growth": "hive",
    "product_defect_rate": "hive",
}


def _init_spark() -> SparkSession:
    """Initialise (or retrieve) a SparkSession with Hive support enabled."""
    return (
        SparkSession.builder
        .appName("MetricETL")
        .master("local[*]")  # Change to Spark cluster master URL in production
        .enableHiveSupport()  # Required for querying Hive tables
        .getOrCreate()
    )


def _read_from_hadoop(spark: SparkSession) -> DataFrame:
    """Placeholder logic for reading data from a Hadoop-backed data lake.

    Adjust the file format and path to suit real HDFS layouts.
    """
    hdfs_path = "hdfs://localhost:9000/path/to/sample_metric.json"  
    return spark.read.json(hdfs_path)


def _read_from_hive(spark: SparkSession) -> DataFrame:
    """Placeholder logic for reading data from a Hive warehouse table."""
    table_name = "sample_db.sample_metric_table" 
    query = f"SELECT * FROM {table_name} LIMIT 100"  # Fixed sample size
    return spark.sql(query)


def _transform(df: DataFrame) -> DataFrame:
    """Sample ETL transformation stage (placeholder)."""
    # Example: drop duplicate records and fill nulls
    return (
        df.dropDuplicates()
        .fillna("N/A")
    )


def get_metric(metric: str) -> List[str]:
    """Retrieve a metric, process it via Spark ETL, and return JSON rows.

    Parameters
    ----------
    metric : str
        The business KPI to retrieve.

    Returns
    -------
    List[str]
        Each element is a JSON-stringified row of the transformed metric
        dataset.

    Raises
    ------
    KeyError
        If the requested metric is not registered in `metric_storage_map`.
    """
    metric_key = metric.lower().strip()
    if metric_key not in metric_storage_map:
        raise KeyError(f"Metric '{metric}' is not defined in the storage map.")

    storage_layer = metric_storage_map[metric_key]

    spark = _init_spark()

    # -------------------------------------------------------------------
    # Data extraction phase (placeholders for demo purposes).
    # -------------------------------------------------------------------
    if storage_layer == "hadoop":
        raw_df = _read_from_hadoop(spark)
    else:  # storage_layer == "hive"
        raw_df = _read_from_hive(spark)

    # -------------------------------------------------------------------
    # Transformation phase (sample transformations shown).
    # -------------------------------------------------------------------
    transformed_df = _transform(raw_df)

    # -------------------------------------------------------------------
    # Load phase: collect to driver as JSON.
    # -------------------------------------------------------------------
    json_rows: List[str] = transformed_df.toJSON().collect()

    # Clean up Spark resources (optional but recommended in scripts)
    spark.stop()

    return json_rows


if __name__ == "__main__":
    # Example usage for manual testing
    sample_metric = "revenue"
    try:
        result = get_metric(sample_metric)
        print(json.dumps(result, indent=2))
    except KeyError as exc:
        print(exc)