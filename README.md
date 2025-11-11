# Deploying CodeLlama-7b-Instruct-hf on AWS SageMaker

## Overview
Complete guide for deploying Meta's CodeLlama-7b-Instruct-hf model on AWS SageMaker for manufacturing code generation use cases. This repository provides production-ready deployment scripts, inference code, and CloudFormation templates for seamless integration into manufacturing workflows.

## Why This Matters

### Manufacturing Applications
- **Automated PLC Code Generation**: Generate ladder logic and structured text for industrial controllers
- **Process Documentation**: Convert manufacturing processes into executable code and documentation
- **Quality Control Automation**: Create inspection scripts and validation routines
- **Legacy System Modernization**: Translate legacy manufacturing code to modern standards

### Performance Benefits
- **Low Latency**: Optimized inference with GPU acceleration for real-time code generation
- **High Throughput**: Batch processing support for large-scale code generation tasks
- **Scalability**: Auto-scaling capabilities to handle variable workloads
- **Cost Efficiency**: Pay-per-use model with automatic scale-down during idle periods

### Cost Optimization
- **Right-sized Instances**: ml.g5.2xlarge provides optimal price/performance ratio
- **Serverless Options**: Deploy with SageMaker Serverless Inference for sporadic workloads
- **Spot Instances**: Reduce costs by up to 70% for non-critical workloads

## Quick Start

### Prerequisites
- AWS Account with SageMaker access
- Python 3.8+
- AWS CLI configured with appropriate credentials
- Hugging Face account (for model access)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/codellama-sagemaker-deployment.git
cd codellama-sagemaker-deployment

# Install dependencies
pip install -r requirements.txt
```

### Deployment Options

#### Option 1: Hugging Face Container (Recommended)
```bash
python deploy_hf_model.py
```

#### Option 2: Large Model Inference (LMI) Container
```bash
python deploy_codellama_lmi.py
```

### Testing the Endpoint
```bash
python test_codellama_hf.py
```

## Manufacturing Use Cases

### 1. Automated PLC Code Generation
Generate ladder logic for industrial automation:
```python
prompt = """Write a PLC ladder logic program to control a conveyor belt system with:
- Start/Stop buttons
- Emergency stop
- Speed control (0-100%)
- Position sensors"""
```

### 2. Manufacturing Process Documentation
Convert process descriptions to executable code:
```python
prompt = """Convert this manufacturing process to Python code:
1. Heat material to 250°C
2. Hold for 30 minutes
3. Cool to 100°C at 5°C/min
4. Log all temperature readings"""
```

### 3. Quality Control Script Automation
Generate inspection and validation routines:
```python
prompt = """Create a quality control script that:
- Measures part dimensions
- Compares against tolerances
- Generates pass/fail report
- Logs defects to database"""
```

## Performance Benchmarks

### Inference Latency
- **Cold Start**: ~30-45 seconds (first request)
- **Warm Inference**: 2-5 seconds per request
- **Batch Processing**: 50-100 requests/minute

### Model Performance
- **Code Quality**: High accuracy for Python, JavaScript, and structured text
- **Context Length**: Up to 16K tokens
- **Output Length**: Configurable (default: 512 tokens)

### Resource Utilization
- **Instance Type**: ml.g5.2xlarge
- **GPU Memory**: ~8GB
- **CPU Memory**: ~16GB

## Cost Analysis

### Real-World Cost Data

#### Hugging Face Deployment (ml.g5.2xlarge)
- **Hourly Rate**: ~$1.50/hour
- **Monthly Cost (24/7)**: ~$1,080/month
- **Cost per 1000 requests**: ~$0.50 (assuming 3 sec/request)

#### Serverless Inference
- **Compute**: $0.000133/second
- **Memory**: $0.0000111/GB-second
- **Ideal for**: <100 requests/day

#### Cost Optimization Tips
1. Use SageMaker Savings Plans for 30-50% discount
2. Implement auto-scaling to scale down during off-hours
3. Use Serverless Inference for development/testing
4. Batch requests when possible to maximize throughput

## Repository Structure

```
codellama-sagemaker-deployment/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── deploy_hf_model.py            # Hugging Face deployment script
├── deploy_codellama_lmi.py       # LMI deployment script
├── test_codellama_hf.py          # Test script for HF endpoint
├── test_codellama.py             # Test script for LMI endpoint
├── notebooks/                     # Jupyter notebooks (coming soon)
│   └── deploy-codellama-7b.ipynb
├── scripts/                       # Utility scripts
│   └── inference.py              # Custom inference handler
├── cloudformation/                # IaC templates (coming soon)
│   └── sagemaker-endpoint.yaml
├── code/                          # LMI inference code
│   ├── inference.py
│   ├── requirements.txt
│   └── serving.properties
└── codellama_code/                # HF inference code
    ├── inference.py
    └── requirements.txt
```

## Advanced Configuration

### Custom Inference Parameters
```python
payload = {
    "inputs": "Your prompt here",
    "parameters": {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True
    }
}
```

### Model Quantization
For cost optimization, consider using quantized models:
- INT8 quantization: 50% memory reduction
- 4-bit quantization: 75% memory reduction

## Troubleshooting

### Common Issues
1. **Model Loading Timeout**: Increase health check timeout in deployment script
2. **Out of Memory**: Use smaller batch sizes or upgrade instance type
3. **Slow Inference**: Enable flash attention or use quantized models

### Monitoring
- CloudWatch Metrics: Monitor invocations, latency, and errors
- SageMaker Model Monitor: Track data quality and model drift
- Cost Explorer: Analyze spending patterns

## Security Best Practices
- Use VPC endpoints for private connectivity
- Enable encryption at rest and in transit
- Implement IAM roles with least privilege
- Use SageMaker Model Registry for version control

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
MIT License - See LICENSE file for details

## Author
**Ninad Shringarpure**  
Senior Solutions Architect, AWS  
Specializing in AI/ML for Manufacturing and Industrial IoT

[LinkedIn Profile](https://www.linkedin.com/in/ninadshringarpure)

## Acknowledgments
- Meta AI for CodeLlama model
- AWS SageMaker team for deployment infrastructure
- Hugging Face for model hosting and inference toolkit

## References
- [CodeLlama Paper](https://arxiv.org/abs/2308.12950)
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Hugging Face Inference Toolkit](https://github.com/aws/sagemaker-huggingface-inference-toolkit)
