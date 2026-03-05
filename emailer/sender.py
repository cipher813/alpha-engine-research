"""
Email sender — AWS SES delivery.

OPEN ITEM: SES sender identity and recipient addresses must be configured
in universe.yaml and verified in AWS SES before deployment.
"""

from __future__ import annotations

import boto3
from botocore.exceptions import ClientError

from config import AWS_REGION


def send_email(
    subject: str,
    html_body: str,
    plain_body: str,
    recipients: list[str],
    sender: str,
    region: str = AWS_REGION,
) -> bool:
    """
    Send a multipart (HTML + plain text) email via AWS SES.

    Returns True on success, False on failure.
    """
    ses = boto3.client("ses", region_name=region)

    try:
        ses.send_email(
            Source=sender,
            Destination={"ToAddresses": recipients},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {
                    "Text": {"Data": plain_body, "Charset": "UTF-8"},
                    "Html": {"Data": html_body, "Charset": "UTF-8"},
                },
            },
        )
        print(f"Email sent: '{subject}' to {recipients}")
        return True
    except ClientError as e:
        print(f"SES send failed: {e.response['Error']['Message']}")
        return False
    except Exception as e:
        print(f"Email send error: {e}")
        return False
