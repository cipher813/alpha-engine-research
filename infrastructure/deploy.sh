#!/usr/bin/env bash
# deploy.sh — Build and deploy both Lambda functions to AWS.
#
# OPEN ITEM: Run after all AWS prerequisites are set up:
#   1. AWS CLI configured with appropriate credentials
#   2. IAM role created (alpha-engine-research-role)
#   3. S3 bucket created (alpha-engine-research)
#   4. SES sender verified
#
# Usage: ./infrastructure/deploy.sh [main|alerts|both]

set -euo pipefail

FUNCTION_MAIN="alpha-engine-research-runner"
FUNCTION_ALERTS="alpha-engine-research-alerts"
ROLE_ARN="${LAMBDA_ROLE_ARN:-arn:aws:iam::ACCOUNT_ID:role/alpha-engine-research-role}"
REGION="${AWS_REGION:-us-east-1}"
BUCKET="alpha-engine-research"
RUNTIME="python3.12"
BUILD_DIR="lambda/package"
ZIP_MAIN="lambda-main.zip"
ZIP_ALERTS="lambda-alerts.zip"
S3_PREFIX="deployments"

TARGET="${1:-both}"

build_package() {
  # Clean build dir to avoid stale/wrong-platform files from previous builds
  echo "Cleaning $BUILD_DIR..."
  rm -rf "$BUILD_DIR"
  mkdir -p "$BUILD_DIR"

  echo "Installing dependencies (linux x86_64, python 3.12)..."
  # Use a temp venv with an up-to-date pip so cross-platform installs work.
  BUILD_VENV="/tmp/lambda-build-venv"
  WHEEL_CACHE="/tmp/lambda-wheels"
  python3 -m venv "$BUILD_VENV"
  "$BUILD_VENV/bin/pip" install --upgrade pip --quiet

  # Pre-build pure-Python packages that lack platform-specific wheels into local .whl files.
  # --only-binary=:all: (required for cross-platform installs) rejects sdists, so we build
  # universal wheels locally first; --find-links then satisfies those deps from the cache.
  rm -rf "$WHEEL_CACHE" && mkdir -p "$WHEEL_CACHE"
  "$BUILD_VENV/bin/pip" wheel sgmllib3k -w "$WHEEL_CACHE" --quiet

  # Install lambda-only deps:
  #   - skip dev tools (pytest, python-dotenv) — not needed at runtime
  #   - skip boto3/botocore/s3transfer — pre-installed in Lambda Python 3.12 runtime
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

  # Strip test files, dist-info, and __pycache__ to reduce package size
  find "$BUILD_DIR" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
  find "$BUILD_DIR" -type d -name "test" -exec rm -rf {} + 2>/dev/null || true
  find "$BUILD_DIR" -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true
  find "$BUILD_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

  echo "Building $ZIP_MAIN..."
  cp -r agents config config.py data emailer graph scoring thesis archive "$BUILD_DIR/"
  cp lambda/handler.py "$BUILD_DIR/handler.py"
  (cd "$BUILD_DIR" && zip -r "../../$ZIP_MAIN" . -x "*.pyc" -x "*/__pycache__/*") > /dev/null
  rm -f "$BUILD_DIR/handler.py"

  echo "Building $ZIP_ALERTS..."
  cp lambda/alerts_handler.py "$BUILD_DIR/alerts_handler.py"
  (cd "$BUILD_DIR" && zip -r "../../$ZIP_ALERTS" . -x "*.pyc" -x "*/__pycache__/*") > /dev/null
  rm -f "$BUILD_DIR/alerts_handler.py"

  echo "Packages built: $ZIP_MAIN ($( du -sh "$ZIP_MAIN" | cut -f1 )), $ZIP_ALERTS ($( du -sh "$ZIP_ALERTS" | cut -f1 ))"
}

deploy_main() {
  echo "Deploying $FUNCTION_MAIN..."
  S3_KEY="$S3_PREFIX/$ZIP_MAIN"
  aws s3 cp "$ZIP_MAIN" "s3://$BUCKET/$S3_KEY" --region "$REGION" --quiet
  if aws lambda get-function --function-name "$FUNCTION_MAIN" --region "$REGION" &>/dev/null; then
    aws lambda update-function-code \
      --function-name "$FUNCTION_MAIN" \
      --s3-bucket "$BUCKET" \
      --s3-key "$S3_KEY" \
      --region "$REGION" > /dev/null
  else
    aws lambda create-function \
      --function-name "$FUNCTION_MAIN" \
      --runtime "$RUNTIME" \
      --role "$ROLE_ARN" \
      --handler "handler.handler" \
      --code "S3Bucket=$BUCKET,S3Key=$S3_KEY" \
      --timeout 600 \
      --memory-size 1024 \
      --environment "Variables={S3_BUCKET=$BUCKET}" \
      --region "$REGION" > /dev/null
  fi
  echo "  $FUNCTION_MAIN deployed."
}

deploy_alerts() {
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
  echo "  $FUNCTION_ALERTS deployed."
}

build_package

case "$TARGET" in
  main)    deploy_main ;;
  alerts)  deploy_alerts ;;
  both)    deploy_main; deploy_alerts ;;
  *)       echo "Usage: $0 [main|alerts|both]"; exit 1 ;;
esac

echo ""
echo "Deployment complete."
echo ""
