# Databricks notebook source

from pyspark.sql import functions as F
dbutils.widgets.text("catalog", "fraud_demo")
catalog = dbutils.widgets.get("catalog")
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(f"/Volumes/{catalog}/bronze/raw_data/creditcard.csv")
    .withColumn("ingest_ts", F.current_timestamp())
    .withColumn("batch_id", F.lit(dbutils.widgets.get("batch_id")))
)

df.write.mode("append").saveAsTable(f"{catalog}.bronze.transactions")
