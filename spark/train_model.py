import argparse
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession


FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fraud model and save Spark pipeline")
    parser.add_argument("--csv-path", default="data/creditcard.csv")
    parser.add_argument("--model-path", default="models/fraud_pipeline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}. Download the Kaggle dataset and place it at this path."
        )

    spark = SparkSession.builder.appName("fraud-model-training").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(csv_path))
        .dropna()
    )

    train_df, _ = df.randomSplit([0.8, 0.2], seed=42)

    assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="features_raw")
    scaler = StandardScaler(
        inputCol="features_raw", outputCol="features", withStd=True, withMean=False
    )
    classifier = LogisticRegression(featuresCol="features", labelCol="Class", maxIter=30)

    pipeline = Pipeline(stages=[assembler, scaler, classifier])
    model = pipeline.fit(train_df)

    model.write().overwrite().save(args.model_path)
    print(f"Model saved to: {args.model_path}")

    spark.stop()


if __name__ == "__main__":
    main()
