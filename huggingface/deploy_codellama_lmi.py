import boto3
import sagemaker
from sagemaker import Model, image_uris
from sagemaker.predictor import Predictor

# Get execution role
try:
    role = sagemaker.get_execution_role()
except ValueError:
    # Replace with your SageMaker execution role ARN
    role = "arn:aws:iam::<YOUR_ACCOUNT_ID>:role/service-role/AmazonSageMaker-ExecutionRole"
    print(f"Using SageMaker execution role: {role}")

# Get the LMI container image
region = boto3.Session().region_name
# AWS Deep Learning Container image URI
image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:0.27.0-deepspeed0.12.6-cu121"

# LMI configuration for CodeLlama with quantization
env = {
    'HF_MODEL_ID': 'codellama/CodeLlama-7b-Instruct-hf',
    'OPTION_DTYPE': 'fp16',  # Use float16 to save memory
    'OPTION_MAX_MODEL_LEN': '4096',  # Limit context length
    'OPTION_TENSOR_PARALLEL_DEGREE': '1',  # Single GPU
    'OPTION_ROLLING_BATCH': 'auto',  # Enable continuous batching
}

# Create model
model = Model(
    image_uri=image_uri,
    env=env,
    role=role,
)

print("Deploying CodeLlama-7b-Instruct-hf with LMI container...")
print("Using fp16 precision for memory optimization")
print("This will take 5-10 minutes...")

# Deploy
endpoint_name = "codellama-7b-instruct"
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",
    endpoint_name=endpoint_name,
    update_endpoint=True,
)

print(f"\nEndpoint deployed: {endpoint_name}")
print("Test with: python test_codellama.py")
