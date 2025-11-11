import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
import tarfile
import os

# Configuration
try:
    role = sagemaker.get_execution_role()
except ValueError:
    role = "arn:aws:iam::706163521949:role/service-role/AmazonSageMaker-ExecutionRole-20251029T061649"
    print(f"Using SageMaker execution role: {role}")

sess = sagemaker.Session()
region = sess.boto_region_name

print(f"SageMaker role: {role}")
print(f"Region: {region}")

# Create model artifact with custom inference code
code_dir = "smoldocling_code"
model_artifact = "model.tar.gz"

print(f"\nCreating model artifact from {code_dir}...")
with tarfile.open(model_artifact, "w:gz") as tar:
    tar.add(code_dir, arcname="code")

# Upload to S3
s3_client = boto3.client("s3")
bucket = sess.default_bucket()
prefix = "smoldocling-model"
s3_path = f"s3://{bucket}/{prefix}/{model_artifact}"

print(f"Uploading to {s3_path}...")
s3_client.upload_file(model_artifact, bucket, f"{prefix}/{model_artifact}")

# Clean up local artifact
os.remove(model_artifact)

# Model configuration
# Note: You need a Hugging Face token to access this model
# Set it as an environment variable: export HF_TOKEN=your_token_here
import os
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("Warning: HF_TOKEN not found in environment variables.")
    print("The model may require authentication. Set it with:")
    print("  export HF_TOKEN=your_huggingface_token")

hub = {
    "HF_MODEL_ID": "docling-project/SmolDocling-256M-preview",
    "HF_TASK": "image-to-text",
}

if hf_token:
    hub["HF_TOKEN"] = hf_token

# Create Hugging Face Model
print("\nCreating SageMaker model...")
huggingface_model = HuggingFaceModel(
    model_data=s3_path,
    role=role,
    transformers_version="4.51.3",
    pytorch_version="2.6.0",
    py_version="py312",
    env=hub,
)

# Check and cleanup existing endpoint
sm_client = boto3.client("sagemaker", region_name=region)
endpoint_name = "smoldocling-endpoint"

# Check if endpoint exists and delete it
try:
    endpoint_desc = sm_client.describe_endpoint(EndpointName=endpoint_name)
    print(f"Found existing endpoint: {endpoint_name}")
    print(f"Status: {endpoint_desc['EndpointStatus']}")
    print(f"Deleting endpoint...")
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    print("✓ Endpoint deleted")
except sm_client.exceptions.ClientError as e:
    if "Could not find endpoint" in str(e):
        print(f"No existing endpoint found")
    else:
        print(f"Error checking endpoint: {e}")

# Check if endpoint config exists and delete it
try:
    sm_client.describe_endpoint_config(EndpointConfigName=endpoint_name)
    print(f"Found existing endpoint config: {endpoint_name}")
    print(f"Deleting endpoint config...")
    sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
    print("✓ Endpoint config deleted")
except sm_client.exceptions.ClientError as e:
    if "Could not find" in str(e):
        print(f"No existing endpoint config found")
    else:
        print(f"Error checking endpoint config: {e}")

# Deploy the model
print("\nDeploying endpoint (this will take 5-10 minutes)...")
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",
    endpoint_name=endpoint_name,
    container_startup_health_check_timeout=600,
)

print(f"\n✓ Endpoint deployed successfully: {predictor.endpoint_name}")
print(f"You can now test it with: python test_smoldocling.py")
