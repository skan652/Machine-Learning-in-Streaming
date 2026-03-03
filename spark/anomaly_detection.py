import argparse
from typing import List

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Streaming anomaly detection")
	parser.add_argument("--bootstrap-servers", default="localhost:9092")
	parser.add_argument("--topic", default="transactions")
	parser.add_argument("--checkpoint", default="checkpoints/anomaly_detection")
	parser.add_argument("--threshold", type=float, default=3.0)
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
		SparkSession.builder.appName("fraud-anomaly-detection")
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
			.withColumn("is_anomaly", F.col("z_score_amount") >= F.lit(args.threshold))
			.filter(F.col("is_anomaly"))
			.select("timestamp", "Time", "Amount", "z_score_amount", "Class")
		)

		if not anomalies.isEmpty():
			print(f"\n==== Batch {batch_id} anomalies ====")
			anomalies.show(truncate=False)

	query = (
		events.writeStream.foreachBatch(detect_in_batch)
		.outputMode("append")
		.option("checkpointLocation", args.checkpoint)
		.start()
	)

	query.awaitTermination()


if __name__ == "__main__":
	main()