import boto3
import json
import base64
from pathlib import Path
from botocore.config import Config

# Configuration
ENDPOINT_NAME = "smoldocling-endpoint"

# Initialize predictor with longer timeout
config = Config(
    read_timeout=600,  # 10 minutes for first request (model loading)
    connect_timeout=60
)
runtime = boto3.client("sagemaker-runtime", config=config)

# Example 1: Using a local image file (base64 encoded)
print("Testing with local image file...")
image_path = Path("document.png")
if not image_path.exists():
    raise FileNotFoundError(f"Image file not found: {image_path}")

with open(image_path, "rb") as f:
    image_bytes = f.read()
image_base64 = base64.b64encode(image_bytes).decode("utf-8")

payload = {
    "image": image_base64,
    "prompt": "Convert this page to docling.",
    "output_format": "markdown",  # Options: markdown, html, doctags
    "max_new_tokens": 8192
}

response = runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Body=json.dumps(payload)
)

result = json.loads(response["Body"].read().decode())
print("\n=== Markdown Output ===")
print(result.get("markdown", ""))
print("\n=== DocTags ===")
print(result.get("doctags", "")[:500] + "...")

# Example 2: Using a URL (if needed)
# Uncomment to test with URL
"""
print("\n\nTesting with image URL...")
payload = {
    "image": "https://example.com/your-document.jpg",
    "prompt": "Convert this page to docling.",
    "output_format": "html",
    "max_new_tokens": 8192
}

response = runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Body=json.dumps(payload)
)

result = json.loads(response["Body"].read().decode())
print("\n=== HTML Output ===")
print(result.get("html", "")[:500] + "...")
"""

print("\n✓ Test completed successfully!")
