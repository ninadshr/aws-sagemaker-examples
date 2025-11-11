import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.s3 import S3Uploader
import tarfile
import os
import argparse

# Configuration - Model specific parameters
MODEL_ID = "codellama/CodeLlama-7b-Instruct-hf"
ENDPOINT_NAME = "codellama-7b-instruct-4bit"
INFERENCE_CODE_DIR = "codellama_code"
INSTANCE_TYPE = "ml.g5.xlarge"
HF_TASK = "text-generation"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Deploy HuggingFace model to SageMaker')
parser.add_argument('--model-id', default=MODEL_ID, help='HuggingFace model ID')
parser.add_argument('--endpoint-name', default=ENDPOINT_NAME, help='SageMaker endpoint name')
parser.add_argument('--code-dir', default=INFERENCE_CODE_DIR, help='Directory containing inference code')
parser.add_argument('--instance-type', default=INSTANCE_TYPE, help='SageMaker instance type')
parser.add_argument('--task', default=HF_TASK, help='HuggingFace task type')
args = parser.parse_args()

# Get execution role
try:
    role = sagemaker.get_execution_role()
except ValueError:
    role = "arn:aws:iam::706163521949:role/service-role/AmazonSageMaker-ExecutionRole-20251029T061649"
    print(f"Using SageMaker execution role: {role}")

# Get region and session
region = boto3.Session().region_name
sess = sagemaker.Session()

# Verify inference code exists
inference_py = os.path.join(args.code_dir, "inference.py")
requirements_txt = os.path.join(args.code_dir, "requirements.txt")

if not os.path.exists(inference_py):
    raise FileNotFoundError(f"{inference_py} not found. Please ensure the inference code exists.")

if not os.path.exists(requirements_txt):
    raise FileNotFoundError(f"{requirements_txt} not found. Please ensure the requirements file exists.")

print(f"Model: {args.model_id}")
print(f"Endpoint: {args.endpoint_name}")
print(f"Instance: {args.instance_type}")
print("Creating model artifact...")

# Create tar.gz file
model_artifact = f"{args.endpoint_name}-model.tar.gz"
with tarfile.open(model_artifact, "w:gz") as tar:
    tar.add(args.code_dir, arcname="code")

# Upload to S3
s3_prefix = args.endpoint_name.replace("-", "_")
s3_path = S3Uploader.upload(model_artifact, f"s3://{sess.default_bucket()}/{s3_prefix}")
print(f"Model artifact uploaded to: {s3_path}")

# Get Hugging Face DLC image URI
image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"

# Create HuggingFace Model
huggingface_model = HuggingFaceModel(
    model_data=s3_path,
    role=role,
    image_uri=image_uri,
    env={
        'HF_MODEL_ID': args.model_id,
        'HF_TASK': args.task,
    }
)

print(f"\nDeploying {args.model_id} with 4-bit quantization and lazy loading...")
print(f"Instance type: {args.instance_type}")

# Deploy
sm_client = boto3.client("sagemaker", region_name=region)

# Check if endpoint exists and delete it
try:
    endpoint_desc = sm_client.describe_endpoint(EndpointName=args.endpoint_name)
    print(f"Found existing endpoint: {args.endpoint_name}")
    print(f"Status: {endpoint_desc['EndpointStatus']}")
    print(f"Deleting endpoint...")
    sm_client.delete_endpoint(EndpointName=args.endpoint_name)
    print("✓ Endpoint deleted")
except sm_client.exceptions.ClientError as e:
    if "Could not find endpoint" in str(e):
        print(f"No existing endpoint found")
    else:
        print(f"Error checking endpoint: {e}")

# Check if endpoint config exists and delete it
try:
    sm_client.describe_endpoint_config(EndpointConfigName=args.endpoint_name)
    print(f"Found existing endpoint config: {args.endpoint_name}")
    print(f"Deleting endpoint config...")
    sm_client.delete_endpoint_config(EndpointConfigName=args.endpoint_name)
    print("✓ Endpoint config deleted")
except sm_client.exceptions.ClientError as e:
    if "Could not find" in str(e):
        print(f"No existing endpoint config found")
    else:
        print(f"Error checking endpoint config: {e}")

print("This will take 5-10 minutes...")

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type=args.instance_type,
    endpoint_name=args.endpoint_name,
)

print(f"\n✓ Endpoint deployed: {args.endpoint_name}")
print(f"✓ Using 4-bit quantization (NF4)")
print(f"✓ Lazy loading enabled - model loads on first request")
print("\nTest with:")
print(f'  python test_codellama_hf.py')

# Cleanup
os.remove(model_artifact)
print("\n✓ Cleaned up local artifacts")
