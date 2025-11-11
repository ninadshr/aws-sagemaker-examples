import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
import base64
from io import BytesIO
from PIL import Image
import requests

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def model_fn(model_dir):
    """Load the model and processor"""
    processor = AutoProcessor.from_pretrained(model_dir)
    
    # Try flash attention, fall back to eager if not available
    try:
        import flash_attn
        attn_implementation = "flash_attention_2" if DEVICE == "cuda" else "eager"
    except ImportError:
        attn_implementation = "eager"
    
    model = AutoModelForVision2Seq.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        _attn_implementation=attn_implementation,
    ).to(DEVICE)
    
    return {"model": model, "processor": processor}


def input_fn(request_body, content_type):
    """Parse input data"""
    if content_type == "application/json":
        import json
        data = json.loads(request_body)
        return data
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(data, model_dict):
    """Run inference"""
    model = model_dict["model"]
    processor = model_dict["processor"]
    
    # Get image from input
    image_data = data.get("image")
    prompt_text = data.get("prompt", "Convert this page to docling.")
    max_new_tokens = data.get("max_new_tokens", 8192)
    output_format = data.get("output_format", "markdown")  # markdown, html, or doctags
    
    # Load image from URL or base64
    if image_data.startswith("http"):
        response = requests.get(image_data, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else:
        # Assume base64 encoded image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
    
    # Create input messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ],
        },
    ]
    
    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(DEVICE)
    
    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    prompt_length = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, prompt_length:]
    doctags = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=False,
    )[0].lstrip()
    
    # Return based on requested format
    if output_format == "doctags":
        return {"doctags": doctags}
    
    # Populate document
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")
    
    if output_format == "markdown":
        return {"markdown": doc.export_to_markdown(), "doctags": doctags}
    elif output_format == "html":
        return {"html": doc.export_to_html(), "doctags": doctags}
    else:
        return {"doctags": doctags}


def output_fn(prediction, accept):
    """Format output"""
    if accept == "application/json":
        import json
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
