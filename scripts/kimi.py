import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from datasets import load_dataset

# model global variables
max_partition = 9
temperature = None
max_new_tokens = 512
system_prompt = 'Do not any provide extra text, only Provide a valid json output.'

def run_kimi(model, processor, text, image):
    '''
    Function to run Kimi VLM
    Input: model, processor, text, image
    Return: output text
    '''
    print("*** Running Inference ***")
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
        
    return output

def load_kimi_16b_unquantised():
    '''
    Function to load unquantized version of kimi-16b
    Input: None
    Return: model, processor
    '''
    model_name = "moonshotai/Kimi-VL-A3B-Instruct"
    print(f"*** Loading model {model_name} into memory ***")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    return model, processor

def load_kimi(model_name):
    '''
    Function to return load and run functions for different kimi models defined in this file.
    Input: model_name
    Output: loading_function, inference_function
    '''
    loading_function, inference_function = None, None
    
    if model_name == "moonshotai/Kimi-VL-A3B-Instruct":
        loading_function = load_kimi_16b_unquantised
        inference_function = run_kimi
    else:
        print("Invalid Model name")

    return loading_function, inference_function

if __name__ == "__main__":
    # test input data
    prompt = "Describe this image."
    hf_repo = "SarangChouguley/master_thesis_v1"
    # Get the test image from dataset
    dataset = load_dataset(hf_repo, data_dir='tools_and_accessary_identification', split='train')
    image = dataset[0]['image']
    # load functions
    load, run = load_kimi("moonshotai/Kimi-VL-A3B-Instruct")
    # load model
    model, processor = load()
    # run model
    output = run(model, processor, prompt, image)
    print(output)