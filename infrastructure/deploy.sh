#!/usr/bin/env bash
# deploy.sh — Build and deploy Lambda functions to AWS.
#
# Main function uses container image (10 GB limit) because dependencies
# exceed the 250 MB zip size limit (numpy + pandas + curl_cffi + yfinance).
# Alerts function uses zip (lightweight, no heavy deps).
#
# Prerequisites:
#   1. AWS CLI configured with appropriate credentials
#   2. IAM role created (alpha-engine-research-role)
#   3. S3 bucket created (alpha-engine-research)
#   4. ECR repository created: alpha-engine-research-runner
#   5. Docker installed and running
#
# Usage: ./infrastructure/deploy.sh [main|alerts|both]

set -euo pipefail

FUNCTION_MAIN="alpha-engine-research-runner"
FUNCTION_ALERTS="alpha-engine-research-alerts"
REGION="${AWS_REGION:-us-east-1}"
BUCKET="alpha-engine-research"
RUNTIME="python3.12"
BUILD_DIR="lambda/package"
ZIP_ALERTS="lambda-alerts.zip"
S3_PREFIX="deployments"

# ECR repository for container image deployment
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$REGION" 2>/dev/null || echo "ACCOUNT_ID")
ROLE_ARN="${LAMBDA_ROLE_ARN:-arn:aws:iam::${ACCOUNT_ID}:role/alpha-engine-research-role}"
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${FUNCTION_MAIN}"

TARGET="${1:-both}"

# ── Lambda env vars from lambda.env ──────────────────────────────────────────
# Reads lambda.env (gitignored) and builds the JSON for --environment.

LAMBDA_ENV_FILE=".env"

build_lambda_env_json() {
  if [ ! -f "$LAMBDA_ENV_FILE" ]; then
    echo "WARNING: $LAMBDA_ENV_FILE not found — Lambda will have no env vars configured." >&2
    echo ""
    return
  fi
  # Parse KEY=VALUE lines from .env, stopping at LAMBDA_SKIP marker.
  # Vars after LAMBDA_SKIP are local-only (AWS creds etc).
  python3 -c "
import json
env = {}
with open('$LAMBDA_ENV_FILE') as f:
    for line in f:
        line = line.strip()
        if line == '# LAMBDA_SKIP':
            break
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        key, val = line.split('=', 1)
        key, val = key.strip(), val.strip()
        # Strip surrounding quotes (single or double)
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('\"', \"'\"):
            val = val[1:-1]
        if key and val:
            env[key] = val
if env:
    print(json.dumps({'Variables': env}))
else:
    print('')
"
}

LAMBDA_ENV_JSON=$(build_lambda_env_json)

# ── Main function: container image deployment ────────────────────────────────

build_and_deploy_main() {
  echo "=== Building container image for $FUNCTION_MAIN ==="

  # Copy flow-doctor source into build context (Docker can't COPY from outside context)
  FLOW_DOCTOR_DIR="${FLOW_DOCTOR_DIR:-$(dirname "$(pwd)")/flow-doctor}"
  rm -rf flow-doctor-pkg
  if [ -d "$FLOW_DOCTOR_DIR" ]; then
    echo "Copying flow-doctor from $FLOW_DOCTOR_DIR..."
    mkdir -p flow-doctor-pkg
    cp -r "$FLOW_DOCTOR_DIR/flow_doctor" flow-doctor-pkg/
    cp "$FLOW_DOCTOR_DIR/pyproject.toml" flow-doctor-pkg/
    [ -f "$FLOW_DOCTOR_DIR/README.md" ] && cp "$FLOW_DOCTOR_DIR/README.md" flow-doctor-pkg/ || true
  else
    echo "ERROR: flow-doctor not found at $FLOW_DOCTOR_DIR"
    exit 1
  fi

  # Build Docker image
  echo "Building Docker image..."
  docker build --platform linux/amd64 --provenance=false -t "$FUNCTION_MAIN:latest" .
  rm -rf flow-doctor-pkg

  # Authenticate with ECR
  echo "Authenticating with ECR..."
  aws ecr get-login-password --region "$REGION" | \
    docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

  # Ensure ECR repository exists
  aws ecr describe-repositories --repository-names "$FUNCTION_MAIN" --region "$REGION" &>/dev/null || \
    aws ecr create-repository --repository-name "$FUNCTION_MAIN" --region "$REGION" > /dev/null

  # Tag and push
  echo "Pushing image to ECR..."
  docker tag "$FUNCTION_MAIN:latest" "$ECR_REPO:latest"
  docker push "$ECR_REPO:latest"
  IMAGE_URI="$ECR_REPO:latest"

  # Update or create Lambda function
  echo "Deploying $FUNCTION_MAIN..."

  # Build env var args from lambda.env
  ENV_ARGS=()
  if [ -n "$LAMBDA_ENV_JSON" ]; then
    ENV_ARGS=(--environment "$LAMBDA_ENV_JSON")
    echo "  Env vars from lambda.env: $(echo "$LAMBDA_ENV_JSON" | python3 -c "import sys,json; print(', '.join(json.load(sys.stdin).get('Variables',{}).keys()))")"
  fi

  if aws lambda get-function --function-name "$FUNCTION_MAIN" --region "$REGION" &>/dev/null; then
    # Check if existing function is zip-based (can't switch to image in-place)
    EXISTING_PKG=$(aws lambda get-function-configuration \
      --function-name "$FUNCTION_MAIN" --region "$REGION" \
      --query "PackageType" --output text 2>/dev/null || echo "Zip")

    if [ "$EXISTING_PKG" = "Image" ]; then
      # Already container-based — update the image and env vars
      aws lambda update-function-code \
        --function-name "$FUNCTION_MAIN" \
        --image-uri "$IMAGE_URI" \
        --region "$REGION" > /dev/null
      # Sync env vars from lambda.env
      if [ -n "$LAMBDA_ENV_JSON" ]; then
        echo "  Waiting for code update to complete..."
        aws lambda wait function-updated --function-name "$FUNCTION_MAIN" --region "$REGION" 2>/dev/null || sleep 5
        aws lambda update-function-configuration \
          --function-name "$FUNCTION_MAIN" \
          --environment "$LAMBDA_ENV_JSON" \
          --region "$REGION" > /dev/null
      fi
    else
      # Zip → Image migration: delete and recreate
      echo "  Migrating from zip to container image..."
      aws lambda delete-function --function-name "$FUNCTION_MAIN" --region "$REGION"
      sleep 2

      aws lambda create-function \
        --function-name "$FUNCTION_MAIN" \
        --package-type Image \
        --code "ImageUri=$IMAGE_URI" \
        --role "$ROLE_ARN" \
        --timeout 900 \
        --memory-size 1024 \
        "${ENV_ARGS[@]}" \
        --region "$REGION" > /dev/null

      echo "  NOTE: EventBridge triggers were removed with the old function."
      echo "  Re-run setup-eventbridge.sh to restore schedules."
    fi
  else
    # Fresh create
    aws lambda create-function \
      --function-name "$FUNCTION_MAIN" \
      --package-type Image \
      --code "ImageUri=$IMAGE_URI" \
      --role "$ROLE_ARN" \
      --timeout 900 \
      --memory-size 1024 \
      "${ENV_ARGS[@]}" \
      --region "$REGION" > /dev/null
  fi
  echo "  $FUNCTION_MAIN deployed (container image)."
}

# ── Alerts function: zip deployment (lightweight) ────────────────────────────

build_and_deploy_alerts() {
  echo "=== Building zip package for $FUNCTION_ALERTS ==="

  # Clean build dir
  rm -rf "$BUILD_DIR"
  mkdir -p "$BUILD_DIR"

  echo "Installing dependencies (linux x86_64, python 3.12)..."
  BUILD_VENV="/tmp/lambda-build-venv"
  WHEEL_CACHE="/tmp/lambda-wheels"
  python3 -m venv "$BUILD_VENV"
  "$BUILD_VENV/bin/pip" install --upgrade pip --quiet

  # Pre-build pure-Python packages that lack platform-specific wheels
  rm -rf "$WHEEL_CACHE" && mkdir -p "$WHEEL_CACHE"
  "$BUILD_VENV/bin/pip" wheel sgmllib3k -w "$WHEEL_CACHE" --quiet

  # flow-doctor wheel
  FLOW_DOCTOR_DIR="${FLOW_DOCTOR_DIR:-$(dirname "$(pwd)")/flow-doctor}"
  if [ -d "$FLOW_DOCTOR_DIR" ]; then
    "$BUILD_VENV/bin/pip" wheel "$FLOW_DOCTOR_DIR" --no-deps -w "$WHEEL_CACHE" --quiet
  fi

  grep -vE "^#|^pytest|^python-dotenv|^boto3|^botocore|^s3transfer" requirements.txt \
    > /tmp/lambda-requirements.txt
  "$BUILD_VENV/bin/pip" install -r /tmp/lambda-requirements.txt -t "$BUILD_DIR" \
    --platform manylinux2014_x86_64 \
    --implementation cp \
    --python-version 312 \
    --abi cp312 \
    --only-binary=:all: \
    --find-links "$WHEEL_CACHE" \
    --quiet

  # Strip to reduce size
  find "$BUILD_DIR" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
  find "$BUILD_DIR" -type d -name "test" -exec rm -rf {} + 2>/dev/null || true
  find "$BUILD_DIR" -type d -name "*.dist-info" ! -name "curl_cffi-*" -exec rm -rf {} + 2>/dev/null || true
  find "$BUILD_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

  echo "Building $ZIP_ALERTS..."
  cp lambda/alerts_handler.py "$BUILD_DIR/alerts_handler.py"
  (cd "$BUILD_DIR" && zip -r "../../$ZIP_ALERTS" . -x "*.pyc" -x "*/__pycache__/*") > /dev/null
  rm -f "$BUILD_DIR/alerts_handler.py"
  echo "  Package: $ZIP_ALERTS ($( du -sh "$ZIP_ALERTS" | cut -f1 ))"

  echo "Deploying $FUNCTION_ALERTS..."
  S3_KEY="$S3_PREFIX/$ZIP_ALERTS"
  aws s3 cp "$ZIP_ALERTS" "s3://$BUCKET/$S3_KEY" --region "$REGION" --quiet
  if aws lambda get-function --function-name "$FUNCTION_ALERTS" --region "$REGION" &>/dev/null; then
    aws lambda update-function-code \
      --function-name "$FUNCTION_ALERTS" \
      --s3-bucket "$BUCKET" \
      --s3-key "$S3_KEY" \
      --region "$REGION" > /dev/null
  else
    aws lambda create-function \
      --function-name "$FUNCTION_ALERTS" \
      --runtime "$RUNTIME" \
      --role "$ROLE_ARN" \
      --handler "alerts_handler.handler" \
      --code "S3Bucket=$BUCKET,S3Key=$S3_KEY" \
      --timeout 60 \
      --memory-size 256 \
      --environment "Variables={S3_BUCKET=$BUCKET}" \
      --region "$REGION" > /dev/null
  fi
  echo "  $FUNCTION_ALERTS deployed (zip)."
}

# ── Dispatch ─────────────────────────────────────────────────────────────────

case "$TARGET" in
  main)    build_and_deploy_main ;;
  alerts)  build_and_deploy_alerts ;;
  both)    build_and_deploy_main; build_and_deploy_alerts ;;
  *)       echo "Usage: $0 [main|alerts|both]"; exit 1 ;;
esac

echo ""
echo "Deployment complete."
echo ""
