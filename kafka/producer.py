import argparse
import json
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Kafka producer for credit-card events")
	parser.add_argument(
		"--bootstrap-servers",
		default="localhost:9092",
		help="Kafka bootstrap servers",
	)
	parser.add_argument("--topic", default="transactions", help="Kafka topic")
	parser.add_argument(
		"--csv-path",
		default="data/creditcard.csv",
		help="Path to credit-card CSV data",
	)
	parser.add_argument(
		"--rows",
		type=int,
		default=0,
		help="Number of rows to stream (0 = all rows)",
	)
	parser.add_argument(
		"--delay-seconds",
		type=float,
		default=0.02,
		help="Delay between events in seconds",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	csv_file = Path(args.csv_path)
	if not csv_file.exists():
		raise FileNotFoundError(
			f"CSV file not found: {csv_file}. Download the Kaggle dataset and place it at this path."
		)

	dataframe = pd.read_csv(csv_file)
	if args.rows > 0:
		dataframe = dataframe.head(args.rows)

	producer = KafkaProducer(
		bootstrap_servers=args.bootstrap_servers,
		value_serializer=lambda record: json.dumps(record).encode("utf-8"),
		acks="all",
	)

	print(
		f"Streaming {len(dataframe)} events to topic '{args.topic}' on {args.bootstrap_servers}"
	)
	for _, row in dataframe.iterrows():
		payload = row.to_dict()
		producer.send(args.topic, payload)
		time.sleep(args.delay_seconds)

	producer.flush()
	producer.close()
	print("Producer finished.")


if __name__ == "__main__":
	main()