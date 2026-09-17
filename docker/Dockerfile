# Exact package versions are pinned in the lock; PySpark is base-supplied.
FROM apache/spark:3.5.9-python3@sha256:af02a459c8706e031c835c16f9db3c463816776e2543dd0f828af65606bcf392
USER root
COPY requirements.lock /tmp/requirements.lock
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.lock && python3 -m pip check
# Use the image's Python and inherit the notebook service's Spark/worker settings.
RUN python3 -m ipykernel install --prefix=/usr/local --name=spark --display-name=spark
USER spark
