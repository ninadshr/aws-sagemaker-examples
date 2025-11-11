import json
import base64
from io import BytesIO
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """Load the model and processor"""
    try:
        logger.info(f"Loading model from {model_dir}")
        
        # SmolDocling uses AutoModelForVision2Seq
        processor = AutoProcessor.from_pretrained(model_dir)
        model = AutoModelForVision2Seq.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,  # Use bfloat16 as per official example
            _attn_implementation="eager",  # Use eager for compatibility
        )
        
        logger.info(f"Model type: {type(model)}")
        
        # Fix generation_config if it's a string (common serialization issue)
        if hasattr(model, 'generation_config'):
            if isinstance(model.generation_config, str):
                logger.warning("generation_config is a string, recreating it")
                from transformers import GenerationConfig
                model.generation_config = GenerationConfig.from_pretrained(model_dir)
            
            # Ensure pad_token_id is set
            if model.generation_config and model.generation_config.pad_token_id is None:
                logger.info("Setting pad_token_id in generation_config")
                model.generation_config.pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        
        # Also set in model config
        if model.config.pad_token_id is None:
            logger.info("Setting pad_token_id in model config")
            model.config.pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        
        logger.info(f"Model loaded successfully. pad_token_id: {model.config.pad_token_id}")
        return {"model": model, "processor": processor}
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        raise

def predict_fn(data, model_components):
    """Run inference"""
    try:
        model = model_components["model"]
        processor = model_components["processor"]
        
        logger.info("Starting prediction")
        
        # Decode base64 image
        image_bytes = base64.b64decode(data["image"])
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        logger.info(f"Image loaded: {image.size}")
        
        # Get custom prompt or use default for document conversion
        user_prompt = data.get("prompt", "Convert this page to docling.")
        
        # Create input messages in chat format (required by SmolDocling)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt}
                ]
            },
        ]
        
        # Apply chat template
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        logger.info(f"Chat template applied")
        
        # Process inputs - pass image directly, not as list
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        logger.info(f"Inputs processed, keys: {inputs.keys()}")
        
        # Generate with proper config
        max_new_tokens = data.get("max_new_tokens", 8192)  # Default from official example
        
        # Explicitly set generation parameters to avoid config issues
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
            "do_sample": False,
        }
        
        logger.info(f"Starting generation with kwargs: {generation_kwargs}")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **generation_kwargs)
        
        logger.info("Generation complete, decoding")
        
        # Trim the prompt from output
        prompt_length = inputs.input_ids.shape[1]
        trimmed_generated_ids = generated_ids[:, prompt_length:]
        
        # Decode output (keep special tokens for DocTags format)
        result = processor.batch_decode(
            trimmed_generated_ids,
            skip_special_tokens=False,
        )[0].lstrip()
        
        logger.info(f"Result length: {len(result)} chars")
        return {"generated_text": result, "format": "doctags"}
        
    except Exception as e:
        logger.error(f"Error in predict_fn: {str(e)}", exc_info=True)
        raise

def input_fn(request_body, content_type):
    """Parse input data"""
    if content_type == "application/json":
        data = json.loads(request_body)
        # Handle both direct format and HuggingFace format with "inputs" key
        if "inputs" in data:
            return data["inputs"]
        return data
    raise ValueError(f"Unsupported content type: {content_type}")

def output_fn(prediction, accept):
    """Format output"""
    if accept == "application/json":
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
