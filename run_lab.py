import os
import sys

# Set JAVA_HOME before importing pyspark
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-24'
os.environ['PATH'] = os.environ['JAVA_HOME'] + '\\bin;' + os.environ.get('PATH', '')

# Now import and run the anomaly detection script
sys.path.insert(0, r'C:\Users\USER\Machine-Learning-in-Streaming\spark')

import argparse
from typing import List

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


def build_schema() -> T.StructType:
    fields: List[T.StructField] = [T.StructField("Time", T.DoubleType(), True)]
    for index in range(1, 29):
        fields.append(T.StructField(f"V{index}", T.DoubleType(), True))
    fields.extend(
        [
            T.StructField("Amount", T.DoubleType(), True),
            T.StructField("Class", T.IntegerType(), True),
        ]
    )
    return T.StructType(fields)


def main() -> None:
    spark = (
        SparkSession.builder.appName("fraud-anomaly-detection")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print("✓ Spark session created successfully")

    schema = build_schema()
    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", "localhost:9092")
        .option("subscribe", "transactions")
        .option("startingOffsets", "latest")
        .load()
    )
    print("✓ Connected to Kafka topic 'transactions'")

    events = (
        raw_stream.select(F.col("timestamp"), F.col("value").cast("string").alias("json"))
        .select("timestamp", F.from_json("json", schema).alias("event"))
        .select("timestamp", "event.*")
    )

    def detect_in_batch(batch_df, batch_id: int) -> None:
        if batch_df.isEmpty():
            return

        amount_stats = batch_df.select(
            F.mean("Amount").alias("mean_amount"),
            F.stddev_samp("Amount").alias("std_amount"),
        ).first()

        mean_amount = amount_stats["mean_amount"]
        std_amount = amount_stats["std_amount"] or 1.0

        anomalies = (
            batch_df.withColumn(
                "z_score_amount", F.abs((F.col("Amount") - F.lit(mean_amount)) / F.lit(std_amount))
            )
            .withColumn("is_anomaly", F.col("z_score_amount") >= F.lit(3.0))
            .filter(F.col("is_anomaly"))
            .select("timestamp", "Time", "Amount", "z_score_amount", "Class")
        )

        if not anomalies.isEmpty():
            print(f"\n==== Batch {batch_id} ANOMALIES DETECTED ====")
            anomalies.show(truncate=False)

    query = (
        events.writeStream.foreachBatch(detect_in_batch)
        .outputMode("append")
        .option("checkpointLocation", "checkpoints/anomaly_detection")
        .start()
    )

    print("✓ Streaming anomaly detection started. Listening for transactions...")
    query.awaitTermination()


if __name__ == "__main__":
    main()
