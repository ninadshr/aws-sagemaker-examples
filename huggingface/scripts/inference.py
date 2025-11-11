import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for lazy loading
model = None
tokenizer = None

def model_fn(model_dir):
    """Lazy load model - returns None, actual loading happens on first request"""
    logger.info("model_fn called - model will be loaded on first inference request")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    return None

def load_model_lazy():
    """Load model with 4-bit quantization on first inference request"""
    global model, tokenizer
    
    if model is not None:
        return model, tokenizer
    
    logger.info("Loading CodeLlama-7b-Instruct-hf with 4-bit quantization...")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    model_id = "codellama/CodeLlama-7b-Instruct-hf"
    
    # Ensure CUDA is visible
    if not torch.cuda.is_available():
        logger.error("CUDA not available! Checking environment...")
        logger.error(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        raise RuntimeError("CUDA is required for 4-bit quantization")
    
    # Configure 4-bit quantization with explicit device
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=False,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    logger.info(f"Model loaded successfully with 4-bit quantization on device: {model.device}")
    return model, tokenizer

def predict_fn(data, model_components):
    """Run inference with lazy-loaded model"""
    try:
        # Lazy load model on first request
        model, tokenizer = load_model_lazy()
        
        # Extract parameters
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", {})
        
        # Default generation parameters
        max_new_tokens = parameters.get("max_new_tokens", 512)
        temperature = parameters.get("temperature", 0.1)
        top_p = parameters.get("top_p", 0.95)
        do_sample = parameters.get("do_sample", True)
        
        # Tokenize input
        input_ids = tokenizer(inputs, return_tensors="pt").input_ids.to(model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode output
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return {"generated_text": generated_text}
        
    except Exception as e:
        logger.error(f"Error in predict_fn: {str(e)}", exc_info=True)
        raise

def input_fn(request_body, content_type):
    """Parse input data"""
    if content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {content_type}")

def output_fn(prediction, accept):
    """Format output"""
    if accept == "application/json":
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
