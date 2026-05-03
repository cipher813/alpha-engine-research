FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.12

# Install dependencies. alpha-engine-lib is installed from public git+https
# (lib was flipped public 2026-05-03; previous versions vendored a local
# copy via deploy.sh staging). [arcticdb] pulls arcticdb (used by data/
# fetchers/price_fetcher.py); [flow_doctor] pulls flow-doctor for the
# handler's setup_logging call. Excludes pytest / python-dotenv /
# pre-installed Lambda runtime deps (boto3 etc.).
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir "alpha-engine-lib[arcticdb,flow_doctor] @ git+https://github.com/cipher813/alpha-engine-lib@v0.3.0" && \
    grep -vE "^#|^$|^pytest|^python-dotenv|^boto3|^botocore|^s3transfer|^alpha-engine-lib" requirements.txt > /tmp/req-lambda.txt && \
    pip install --no-cache-dir -r /tmp/req-lambda.txt && \
    rm -rf /root/.cache/pip /tmp/req-lambda.txt

# Copy application code
COPY agents/ ${LAMBDA_TASK_ROOT}/agents/
COPY config/ ${LAMBDA_TASK_ROOT}/config/
COPY config.py ${LAMBDA_TASK_ROOT}/
COPY data/ ${LAMBDA_TASK_ROOT}/data/
COPY emailer/ ${LAMBDA_TASK_ROOT}/emailer/
COPY graph/ ${LAMBDA_TASK_ROOT}/graph/
COPY scoring/ ${LAMBDA_TASK_ROOT}/scoring/
COPY thesis/ ${LAMBDA_TASK_ROOT}/thesis/
COPY archive/ ${LAMBDA_TASK_ROOT}/archive/
COPY evals/ ${LAMBDA_TASK_ROOT}/evals/
COPY memory/ ${LAMBDA_TASK_ROOT}/memory/
COPY rag/ ${LAMBDA_TASK_ROOT}/rag/
# scripts/ holds aggregate_costs.py — imported by lambda/handler.py at the
# end of every successful run to write the daily cost parquet (PR #81 SF-
# wire-up). Without this COPY the import raises ModuleNotFoundError at
# runtime; the handler's try/except catches it (non-fatal — Backtester
# renders an empty cost section), but the parquet never gets written.
COPY scripts/ ${LAMBDA_TASK_ROOT}/scripts/
COPY flow-doctor.yaml ${LAMBDA_TASK_ROOT}/
COPY preflight.py ${LAMBDA_TASK_ROOT}/
COPY retry.py ${LAMBDA_TASK_ROOT}/
COPY health_status.py ${LAMBDA_TASK_ROOT}/
COPY ssm_secrets.py ${LAMBDA_TASK_ROOT}/
COPY dry_run.py ${LAMBDA_TASK_ROOT}/
COPY strict_mode.py ${LAMBDA_TASK_ROOT}/

# Main Lambda handler
COPY lambda/handler.py ${LAMBDA_TASK_ROOT}/handler.py

# Eval-judge Lambda handler — same image, separate Lambda function in
# AWS that overrides CMD to ["eval_judge_handler.handler"] via
# --image-config at deploy time. Sharing the image with the main
# function avoids a parallel ECR repo + duplicate Docker build for a
# handler that needs the exact same dependency set.
COPY lambda/eval_judge_handler.py ${LAMBDA_TASK_ROOT}/eval_judge_handler.py

# Rolling-4-week-mean Lambda handler (PR 4b) — same image, separate
# Lambda overriding CMD to ["eval_rolling_mean_handler.handler"].
COPY lambda/eval_rolling_mean_handler.py ${LAMBDA_TASK_ROOT}/eval_rolling_mean_handler.py

CMD ["handler.handler"]
