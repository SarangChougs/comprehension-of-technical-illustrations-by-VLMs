import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from datasets import load_dataset

# model global variables
max_partition = 9
temperature = None
max_new_tokens = 1024
system_prompt = 'Do not provide extra text, only Provide a valid json output.'

def run_model(model, text_tokenizer, visual_tokenizer, text, image):
    '''
    Function to run Ovis2 VLM
    Input: model, text_tokenizer, visual_tokenizer, text, image
    Return: output text
    '''
    print("*** Running Inference ***")
    # process input
    query = f'<image>\n{text}\n {system_prompt}'
    if type(image) is str:
        image = Image.open(image)
    images = [image]
    prompt, input_ids, pixel_values = model.preprocess_inputs(query, images, max_partition=max_partition)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
    pixel_values = [pixel_values]
    
    # generate output
    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=temperature,
            repetition_penalty=1.05,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
        )
        # Only suppress warnings in a specific section
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
        output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        
    return output

def load_model(model_name):
    '''
    Function to load Ovis vlm
    Input: None
    Return: model, text tokenizer, visual tokenizer
    '''
    # load model
    print(f"*** Loading model {model_name} into memory ***")
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.bfloat16,
                                                 multimodal_max_length=32768,
                                                 trust_remote_code=True).cuda()
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()

    return model, text_tokenizer, visual_tokenizer

def load_ovis():
    '''
    Function to return load and run functions for different ovis models defined in this file.
    Input: model_name
    Output: loading_function, inference_function
    '''
    
    loading_function = load_model
    inference_function = run_model

    return loading_function, inference_function

if __name__ == "__main__":
    # test input data
    prompt = "Describe this image."
    hf_repo = "SarangChouguley/master_thesis_v1"
    # Get the test image from dataset
    dataset = load_dataset(hf_repo, data_dir='tools_and_accessary_identification', split='train')
    image = dataset[0]['image']
    # define models
    models = {
        'v2-8b': 'AIDC-AI/Ovis2-4B',
        'v2-16b' : 'AIDC-AI/Ovis2-16B',
        'v2-4b': 'AIDC-AI/Ovis2-4B',
        'v1.6-9b': 'AIDC-AI/Ovis1.6-Gemma2-9B'
    }
    # load functions
    load, run = load_ovis()
    # load model
    model, text_tokenizer, visual_tokenizer = load(models['v1.6-9b'])
    # run model
    output = run(model, text_tokenizer, visual_tokenizer, prompt, image)
    print(output)