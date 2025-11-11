import boto3
import json

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime')

endpoint_name = "codellama-7b-instruct"

# Test prompt for code generation
payload = {
    "inputs": "def fibonacci(n):",
    "parameters": {
        "max_new_tokens": 150,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True
    }
}

print(f"Testing endpoint: {endpoint_name}")
print(f"Prompt: {payload['inputs']}\n")

# Invoke endpoint
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=json.dumps(payload)
)

# Parse response
result = json.loads(response['Body'].read().decode())
print("Generated code:")
print("=" * 50)

# LMI container returns different format - handle both
if isinstance(result, dict):
    # LMI format: {"generated_text": "..."}
    print(result.get('generated_text', result))
elif isinstance(result, list) and len(result) > 0:
    # HuggingFace format: [{"generated_text": "..."}]
    print(result[0]['generated_text'])
else:
    print(result)
    
print("=" * 50)
