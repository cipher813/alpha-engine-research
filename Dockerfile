FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.12

# Install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir \
    $(grep -vE "^#|^$|^pytest|^python-dotenv|^boto3|^botocore|^s3transfer|^flow-doctor" requirements.txt) \
    && rm -rf /root/.cache/pip

# Install flow-doctor from local source (copied into build context)
COPY flow-doctor-pkg/ /tmp/flow-doctor-pkg/
RUN pip install --no-cache-dir --no-deps /tmp/flow-doctor-pkg/ \
    && rm -rf /tmp/flow-doctor-pkg /root/.cache/pip

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
COPY flow-doctor.yaml ${LAMBDA_TASK_ROOT}/
COPY retry.py ${LAMBDA_TASK_ROOT}/

# Main Lambda handler
COPY lambda/handler.py ${LAMBDA_TASK_ROOT}/handler.py

CMD ["handler.handler"]
