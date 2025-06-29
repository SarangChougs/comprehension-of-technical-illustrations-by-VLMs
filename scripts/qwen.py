## Script to containing functions to load and run Qwen VLM models

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from datasets import load_dataset

DEFAULT_MIN_PIXELS= 256*28*28
DEFAULT_MAX_PIXELS = 1280*28*28

## Function to load Qwen/Qwen2-VL-7B-Instruct model
def load_qwen2_vl_7b_instruct():
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    print(f"*** Loading model {model_name} into memory ***")
    
    # Load model weights to available gpu
    model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
    )

    # Load model processor with default min and max input image pixels
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS)

    return model, processor

## Function to load Qwen/Qwen2.5-VL-7B-Instruct model
def load_qwen2_5_vl_7b_instruct():
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"*** Loading model {model_name} into memory ***")
    
    # Load model weights to available gpu
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
    )

    # Load model processor with default min and max input image pixels
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS)

    return model, processor

## Function to inference Qwen/Qwen2-VL-7B-Instruct model
def inference_qwen2_vl_7b_instruct(model, processor, prompt, image_path):
    print("*** Running Inference ***")
    
    # Set max new tokens
    max_new_tokens_from_config = 768
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path
                    #"image": f"file://{image_path}"
                },
                {"type": "text", "text": f"{prompt}"},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens_from_config)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

## Function to load different model versions 
def load_qwen(model_name):
    loading_function, inference_function = None, None
    
    if model_name == "Qwen/Qwen2-VL-7B-Instruct":
        loading_function = load_qwen2_vl_7b_instruct
        inference_function = inference_qwen2_vl_7b_instruct
    elif model_name == "Qwen/Qwen2.5-VL-7B-Instruct":
        loading_function = load_qwen2_5_vl_7b_instruct
        inference_function = inference_qwen2_vl_7b_instruct
    else:
        print("Invalid Model name")

    return loading_function, inference_function

if __name__ == "__main__":
    prompt = "Describe this image."
    hf_repo = "SarangChouguley/master_thesis_v1"
    # Get the test image from dataset
    dataset = load_dataset(hf_repo, data_dir='tools_and_accessary_identification', split='train')
    image = dataset[0]['image']
    load, run = load_qwen("Qwen/Qwen2.5-VL-7B-Instruct")
    model, processor = load()
    output = run(model, processor, prompt, image)
    print(output)