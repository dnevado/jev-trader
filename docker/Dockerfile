# Scheduled paper-trading job (ECS Fargate). Build from the repository root:
#   docker build --platform linux/amd64 -f docker/Dockerfile -t jevbt-paper .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app JEVBT_DATA_DIR=/tmp/data
WORKDIR /app

COPY docker/requirements-job.txt /tmp/requirements-job.txt
RUN pip install --no-cache-dir -r /tmp/requirements-job.txt && rm /tmp/requirements-job.txt

# Only the modules the job needs (no LLM / Jev / research / API code).
COPY src/jevbt/__init__.py src/jevbt/config.py src/jevbt/strategy.py src/jevbt/paper.py src/jevbt/aws_job.py jevbt/
COPY src/jevbt/backtest/__init__.py src/jevbt/backtest/engine.py jevbt/backtest/
COPY src/jevbt/features/__init__.py src/jevbt/features/technical.py jevbt/features/
COPY src/jevbt/ingest/__init__.py src/jevbt/ingest/alpaca.py src/jevbt/ingest/cache.py jevbt/ingest/
COPY src/jevbt/broker/__init__.py src/jevbt/broker/alpaca_paper.py jevbt/broker/

RUN useradd --create-home --uid 10001 job
USER job

ENTRYPOINT ["python", "-m", "jevbt.aws_job"]
CMD ["trade"]
