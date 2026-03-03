import argparse
from typing import List

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Streaming inference with Spark + Kafka")
	parser.add_argument("--bootstrap-servers", default="localhost:9092")
	parser.add_argument("--topic", default="transactions")
	parser.add_argument("--model-path", default="models/fraud_pipeline")
	parser.add_argument("--checkpoint", default="checkpoints/stream_inference")
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
		SparkSession.builder.appName("fraud-stream-inference")
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

	parsed = (
		raw_stream.select(F.col("timestamp"), F.col("value").cast("string").alias("json"))
		.select("timestamp", F.from_json("json", schema).alias("event"))
		.select("timestamp", "event.*")
	)

	model = PipelineModel.load(args.model_path)
	scored = model.transform(parsed)

	output = scored.select(
		"timestamp",
		"Time",
		"Amount",
		F.col("prediction").cast("int").alias("prediction"),
		F.col("probability").cast("string").alias("probability"),
		"Class",
	)

	query = (
		output.writeStream.format("console")
		.outputMode("append")
		.option("truncate", "false")
		.option("checkpointLocation", args.checkpoint)
		.start()
	)
	query.awaitTermination()


if __name__ == "__main__":
	main()