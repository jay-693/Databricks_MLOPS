# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering
# MAGIC %restart_python
from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.sql import Window

fe = FeatureEngineeringClient()

dbutils.widgets.text("catalog", "fraud_demo_test")

catalog = dbutils.widgets.get("catalog")

df_raw = spark.table(f"{catalog}.bronze.creditcarddata")
print("Rows:", df_raw.count())
print("Columns:", len(df_raw.columns))
print(df_raw.columns)

df_raw.printSchema()

display(
    df_raw.groupBy("isFraud")
    .count()
    .withColumn("percentage", F.round(F.col("count") / F.sum("count").over(Window.partitionBy()) * 100, 4))
    .orderBy("isFraud")
)

display(
    df_raw.groupBy("type", "isFraud")
    .count()
    .orderBy("type", "isFraud")
)

# --Data quality checks
# --Before feature engineering, check nulls, duplicate rows, invalid amounts, transaction types, and the target distribution.

# Null count for every column
null_counts = df_raw.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in df_raw.columns
])

display(null_counts)

# Duplicate rows
duplicate_count = (
    df_raw.count()
    - df_raw.dropDuplicates().count()
)

print("Duplicate rows:", duplicate_count)

# Basic validity checks
print("Negative amounts:", df_raw.filter(F.col("amount") < 0).count())
print("Zero amounts:", df_raw.filter(F.col("amount") == 0).count())

display(
    df_raw.groupBy("type")
    .agg(
        F.count("*").alias("transactions"),
        F.round(F.avg("amount"), 2).alias("avg_amount"),
        F.round(F.max("amount"), 2).alias("max_amount")
    )
    .orderBy(F.desc("transactions"))
)

display(
    df_raw.select(
        F.substring("nameOrig", 1, 1).alias("orig_type"),
        F.substring("nameDest", 1, 1).alias("dest_type")
    )
    .groupBy("orig_type", "dest_type")
    .count()
    .orderBy(F.desc("count"))
)

# Feature engineering — transaction and time features
df_fe = (
    df_raw
    .withColumn("log_amount", F.log1p(F.col("amount")))
    .withColumn("amount_is_zero", (F.col("amount") == 0).cast("int"))
    .withColumn("hour_of_day", ((F.col("step") - 1) % 24).cast("int"))
    .withColumn("simulation_day", F.floor((F.col("step") - 1) / 24).cast("int") + 1)
)

display(
    df_fe.select(
        "step", "type", "amount", "log_amount",
        "hour_of_day", "simulation_day", "isFraud",
    ).limit(20)
)

# Feature engineering — account/entity type

df_fe = (
    df_fe
    .withColumn("orig_is_customer",
                (F.substring("nameOrig", 1, 1) == "C").cast("int"))
    .withColumn("dest_is_customer",
                (F.substring("nameDest", 1, 1) == "C").cast("int"))
    .withColumn("dest_is_merchant",
                (F.substring("nameDest", 1, 1) == "M").cast("int"))
)

display(
    df_fe.groupBy(
        "orig_is_customer",
        "dest_is_customer",
        "dest_is_merchant",
        "isFraud"
    )
    .count()
    .orderBy(F.desc("count"))
)

# Train/test split BEFORE target-dependent feature engineering
# For a production fraud system, a time-based split is often more realistic than a
# random split because we want to predict future transactions using past transactions.

max_step = df_fe.agg(F.max("step").alias("max_step")).first()["max_step"]
split_step = int(max_step * 0.80)

train_df = df_fe.filter(F.col("step") <= split_step)
test_df = df_fe.filter(F.col("step") > split_step)

print("Maximum step:", max_step)
print("Split step:", split_step)
print("Train rows:", train_df.count())
print("Test rows:", test_df.count())

display(
    train_df.groupBy("isFraud").count().withColumn("dataset", F.lit("train"))
    .unionByName(
        test_df.groupBy("isFraud").count().withColumn("dataset", F.lit("test"))
    )
    .orderBy("dataset", "isFraud")
)

# Historical transaction frequency features
# Instead of using the raw customer IDs, we create:
# orig_txn_count_train: number of historical transactions by sender
# dest_txn_count_train: number of historical transactions received by destination

orig_counts = (
    train_df
    .groupBy("nameOrig")
    .agg(F.count("*").alias("orig_txn_count_train"))
)

dest_counts = (
    train_df
    .groupBy("nameDest")
    .agg(F.count("*").alias("dest_txn_count_train"))
)

train_df = (
    train_df
    .join(orig_counts, on="nameOrig", how="left")
    .join(dest_counts, on="nameDest", how="left")
)

test_df = (
    test_df
    .join(orig_counts, on="nameOrig", how="left")
    .join(dest_counts, on="nameDest", how="left")
    .fillna({
        "orig_txn_count_train": 0,
        "dest_txn_count_train": 0
    })
)

# Log versions reduce the impact of very high-frequency entities.
train_df = (
    train_df
    .withColumn("log_orig_txn_count", F.log1p("orig_txn_count_train"))
    .withColumn("log_dest_txn_count", F.log1p("dest_txn_count_train"))
)

test_df = (
    test_df
    .withColumn("log_orig_txn_count", F.log1p("orig_txn_count_train"))
    .withColumn("log_dest_txn_count", F.log1p("dest_txn_count_train"))
)

display(
    train_df.select(
        "nameOrig", "orig_txn_count_train",
        "nameDest", "dest_txn_count_train",
        "isFraud"
    ).limit(20)
)

# Transaction-type encoding

one_hot_cols = []

for value in ["TRANSFER", "CASH_OUT"]:
    new_col_name = f"type_{value}"

    train_df = train_df.withColumn(
        new_col_name,
        F.when(F.col("type") == value, 1).otherwise(0)
    )

    one_hot_cols.append(new_col_name)

# Final Model Features

label_col = "isFraud"

numeric_feature_cols = [
    "log_amount",
    "amount_is_zero",
    "hour_of_day",
    "simulation_day",
    "orig_is_customer",
    "dest_is_customer",
    "dest_is_merchant",
    "log_orig_txn_count",
    "log_dest_txn_count",
]

all_feature_cols = numeric_feature_cols + one_hot_cols
numeric_assembler = VectorAssembler(
    inputCols=all_feature_cols, outputCol="All_features")
scaler = StandardScaler(inputCol="All_features",
                        outputCol="scaled_features", withMean=True, withStd=True)

pipeline = Pipeline(
    stages=[
        numeric_assembler,
        scaler
    ]
)

# Fit on the TRAINING period only, then apply the same fitted pipeline to the test period.
pipeline_model = pipeline.fit(train_df)

final_train_df = pipeline_model.transform(train_df)
final_test_df = pipeline_model.transform(test_df)

train_table = "fraud_demo.bronze.creditcardtransactions"
test_table = "fraud_demo.bronze.creditcardtransactions_test"

# Same columns, same order, for both periods.
output_cols = [*numeric_feature_cols, label_col, *one_hot_cols]

# Add one-hot type columns to test_df (only added to train_df in the encoding step)
for col_name in one_hot_cols:
    type_value = col_name.replace("type_", "")
    test_df = test_df.withColumn(col_name, F.when(
        F.col("type") == type_value, 1).otherwise(0))

final_test_df = pipeline_model.transform(test_df)

final_train_features = final_train_df.select(*output_cols)
final_test_features = final_test_df.select(*output_cols)

assert final_train_features.columns == final_test_features.columns, "train/test columns differ"

(
    final_train_features.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(train_table)
)

(
    final_test_features.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(test_table)
)

# Sanity check: rows and frauds in what was actually written
for name in (train_table, test_table):
    t = spark.table(name)
    total = t.count()
    frauds = t.filter(F.col(label_col) == 1).count()
    print(f"{name}: {total:,} rows | {frauds:,} frauds ({100 * frauds / total:.4f}%) | {len(t.columns)} columns")

sources = {
    "fraud_features":      f"{catalog}.bronze.creditcardtransactions",
    "fraud_features_test": f"{catalog}.bronze.creditcardtransactions_test",
}

for target, source in sources.items():
    df = spark.table(source).withColumn(
        "txn_id", F.monotonically_increasing_id())
    fe.create_table(
        name=f"{catalog}.silver.{target}",
        primary_keys=["txn_id"],
        df=df,
        description=f"Engineered features for fraud model ({target})",
    )
