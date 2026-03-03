import argparse
from typing import List

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Drift monitoring on streaming predictions")
	parser.add_argument("--bootstrap-servers", default="localhost:9092")
	parser.add_argument("--topic", default="transactions")
	parser.add_argument("--model-path", default="models/fraud_pipeline")
	parser.add_argument("--window-duration", default="1 minute")
	parser.add_argument("--slide-duration", default="30 seconds")
	parser.add_argument("--checkpoint", default="checkpoints/monitoring")
	return parser.parse_args()


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
	args = parse_args()
	spark = (
		SparkSession.builder.appName("fraud-drift-monitoring")
		.config("spark.sql.shuffle.partitions", "2")
		.getOrCreate()
	)
	spark.sparkContext.setLogLevel("WARN")

	schema = build_schema()
	raw_stream = (
		spark.readStream.format("kafka")
		.option("kafka.bootstrap.servers", args.bootstrap_servers)
		.option("subscribe", args.topic)
		.option("startingOffsets", "latest")
		.load()
	)

	events = (
		raw_stream.select(F.col("timestamp"), F.col("value").cast("string").alias("json"))
		.select("timestamp", F.from_json("json", schema).alias("event"))
		.select("timestamp", "event.*")
		.withWatermark("timestamp", "2 minutes")
	)

	model = PipelineModel.load(args.model_path)
	scored = model.transform(events)

	monitored = (
		scored.groupBy(
			F.window("timestamp", args.window_duration, args.slide_duration).alias("window")
		)
		.agg(
			F.count("*").alias("events"),
			F.avg(F.col("prediction")).alias("predicted_fraud_rate"),
			F.avg(F.col("Class").cast("double")).alias("true_fraud_rate"),
			F.avg("Amount").alias("avg_amount"),
			F.stddev_samp("Amount").alias("std_amount"),
		)
		.select(
			F.col("window.start").alias("window_start"),
			F.col("window.end").alias("window_end"),
			"events",
			F.round("predicted_fraud_rate", 4).alias("predicted_fraud_rate"),
			F.round("true_fraud_rate", 4).alias("true_fraud_rate"),
			F.round("avg_amount", 2).alias("avg_amount"),
			F.round("std_amount", 2).alias("std_amount"),
		)
	)

	query = (
		monitored.writeStream.format("console")
		.outputMode("update")
		.option("truncate", "false")
		.option("checkpointLocation", args.checkpoint)
		.start()
	)
	query.awaitTermination()


if __name__ == "__main__":
	main()