# run.py verifies the production base's immutable ID before this offline build.
FROM airline-revision-spark:3.5.9
USER root
COPY requirements.lock /tmp/xgb-study.lock
COPY wheels /tmp/xgb-wheels
RUN python3 -m pip install --no-index --find-links=/tmp/xgb-wheels --no-deps \
    --require-hashes --target=/opt/xgb-study -r /tmp/xgb-study.lock \
    && rm -rf /tmp/xgb-wheels
ENV PYTHONPATH=/opt/xgb-study:/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ENV PYTHONDONTWRITEBYTECODE=1
RUN python3 -m pip check
