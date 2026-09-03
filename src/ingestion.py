# Databricks notebook source

from pyspark.sql import functions as F
from delta.tables import DeltaTable

dbutils.widgets.text("catalog", "fraud_demo")
dbutils.widgets.text("batch_id", "manual")

catalog = dbutils.widgets.get("catalog")
batch_id = dbutils.widgets.get("batch_id")

raw = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(f"/Volumes/{catalog}/bronze/raw_data/creditcard.csv")
)

# No natural primary key in this dataset, so derive a deterministic
# one from the transaction's own values. Any row with identical
# Time/V1..V28/Amount/Class is the same transaction, no matter how
# many times ingestion runs against the same source file.
df = (
    raw
    .withColumn("row_hash", F.sha2(F.concat_ws("||", *raw.columns), 256))
    .withColumn("ingest_ts", F.current_timestamp())
    .withColumn("batch_id", F.lit(batch_id))
)

table_name = f"{catalog}.bronze.transactions"

if spark.catalog.tableExists(table_name):
    target = DeltaTable.forName(spark, table_name)
    (target.alias("t")
        .merge(df.alias("s"), "t.row_hash = s.row_hash")
        .whenNotMatchedInsertAll()
        .execute())
    print(f"Merged batch_id={batch_id} — only new rows inserted.")
else:
    df.write.mode("overwrite").saveAsTable(table_name)
    print(f"Created {table_name} with initial load.")
