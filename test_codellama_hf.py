import boto3
import json

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime')

endpoint_name = "codellama-7b-instruct-4bit"

def parse_response(response):
    """Parse the response from the endpoint"""
    result = json.loads(response['Body'].read().decode())
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], str):
        # Response is a list with JSON string
        output = json.loads(result[0])
        return output['generated_text']
    elif isinstance(result, dict) and 'generated_text' in result:
        return result['generated_text']
    else:
        return result

# Test 1: Simple code completion
print("Test 1: Code completion")
print("-" * 50)
payload = {
    "inputs": "def fibonacci(n):",
    "parameters": {
        "max_new_tokens": 256,
        "temperature": 0.1,
        "do_sample": True
    }
}

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=json.dumps(payload)
)

print(parse_response(response))
print()

# Test 2: Code explanation
print("Test 2: Code explanation")
print("-" * 50)
payload = {
    "inputs": "[INST] Explain what this Python code does: def factorial(n): return 1 if n <= 1 else n * factorial(n-1) [/INST]",
    "parameters": {
        "max_new_tokens": 200,
        "temperature": 0.2,
    }
}

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=json.dumps(payload)
)

print(parse_response(response))
print()

# Test 3: Bug fixing
print("Test 3: Bug fixing")
print("-" * 50)
payload = {
    "inputs": "[INST] Fix the bug in this code: def divide(a, b): return a / b [/INST]",
    "parameters": {
        "max_new_tokens": 300,
        "temperature": 0.1,
    }
}

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=json.dumps(payload)
)

print(parse_response(response))
