"""Fail preflight unless both workers are alive and a Spark action succeeds."""
import json
from urllib.request import urlopen
from airline.common.spark_session import create_spark_session


def main():
    with urlopen("http://spark-master:8080/json/", timeout=10) as response:
        status = json.load(response)
    alive = [worker for worker in status.get("workers", []) if worker.get("state") == "ALIVE"]
    if len(alive) < 2:
        raise RuntimeError("Spark preflight requires two registered live workers")
    spark = create_spark_session("airline-preflight")
    try:
        if spark.range(4).repartition(4).count() != 4:
            raise RuntimeError("Spark execution health check failed")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
