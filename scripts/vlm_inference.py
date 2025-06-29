import os, json, re
from app.vlm_class import VisionLanguageModel
from app.utils import append_date_time
from tqdm.auto import tqdm
import argparse
from datasets import load_dataset

def run_vlm(model, index=None, test=True, output_dir='output/vlm_output'):
    abilities = ['visual_assembly_recognition', 'safety_symbol_interpretation', 'tools_and_accessary_identification'] # Add new abilities here

    # Load the class
    vlm = VisionLanguageModel(model_map(model))

    # Load the model
    vlm.load()

    results = []
    if index is not None:
        ability = abilities[index]
        print(f"--- Running model {model} on ability {ability} ---")
        results = run_single_ability(vlm, ability, test)
        # Save the result
        save_results(results, model, ability, output_dir = output_dir)
    else:
        for ability in abilities:
            print(f"--- Running model {model} on ability {ability} ---")
            results = run_single_ability(vlm, ability, test)
            # Save the result
            save_results(results, model, ability, output_dir = output_dir)
    

def run_single_ability(model, ability, test):
    '''
    Function to run vlm on the subdataset/ability
    '''
    hf_repo = "SarangChouguley/master_thesis_v2"
    # Get the dataset/ subdataset
    dataset = load_dataset(hf_repo, data_dir=ability, split='train')
      
    # Test on one datapoint
    if test:
        results=[]
        # get the first datapoint
        item = dataset[0]

        # Get the question/prompt
        ability_prompt = prompt_map(ability)
        
        # take item prompt if present
        #if item['prompt'] and item['prompt'] != 'None':
        #    ability_prompt = item['prompt']
        
        if ability == 'safety_symbol_interpretation':
            #prompt = ability_prompt + ' \n Extra Information: '  + str(item['extra_info'])
            prompt = ability_prompt
        if ability == 'visual_assembly_recognition':
            prompt = item['question'] + ' \n Options: '  + str(item['options']) + ability_prompt
        if ability == 'tools_and_accessary_identification':
            #prompt = item['question'] + ' \n Context: '  + item['context'] + ability_prompt
            prompt = item['question'] + ability_prompt
        # get the image
        image = item['image']
        # get the response
        response = model.run(prompt, image)
        # print raw response
        print(f"Model raw response: {response}")
        # clean the response
        response = clean_response(response)
        result = {"id": item['Filename'], "prompt": prompt, "response": response, "ground_truth": item['ground_truth']}
        print(result)
        results.append(result)
        
        return results
        
    # Run through the dataset
    results = []
    for idx, item in tqdm(enumerate(dataset)):

        # Get the question/prompt
        ability_prompt = prompt_map(ability)
        
        # take item prompt if present
        #if item['prompt'] and item['prompt'] != 'None':
        #    ability_prompt = item['prompt']
        
        if ability == 'safety_symbol_interpretation':
            #prompt = ability_prompt + ' \n Extra Information: '  + str(item['extra_info'])
             prompt = ability_prompt
        if ability == 'visual_assembly_recognition':
            prompt = item['question'] + ' \n Options: '  + str(item['options']) + ability_prompt
        if ability == 'tools_and_accessary_identification':
            #prompt = item['question'] + ' \n Context: '  + item['context'] + ability_prompt
            prompt = item['question'] + ability_prompt
        # get the image
        image = item['image']
        # get the response
        response = model.run(prompt, image)
        # clean the response
        response = clean_response(response)
        result = {"id": f"{item['Filename']}_{idx}", "prompt": prompt, "response": response, "ground_truth": item['ground_truth']}
        print(result)
        # append the result
        results.append(result)
        
    return results

def model_map(model):
    '''
    Function to map model load name to simple model name
    '''
    map = {
        'qwen2' : 'Qwen/Qwen2-VL-7B-Instruct',
        'ovis16b' : 'AIDC-AI/Ovis2-16B',
        'ovis8b' : 'AIDC-AI/Ovis2-8B',
        'ovis4b': 'AIDC-AI/Ovis2-4B',
        'ovis1-6_9b': 'AIDC-AI/Ovis1.6-Gemma2-9B',
        'qwen2_5': "Qwen/Qwen2.5-VL-7B-Instruct",
        'internvl2-5_26b': 'OpenGVLab/InternVL2_5-26B-AWQ',
        'internvl2-5_8b': 'OpenGVLab/InternVL2_5-8B-MPO',
        'kimi': 'moonshotai/Kimi-VL-A3B-Instruct',
        'internvl3_14b' : 'OpenGVLab/InternVL3-14B-AWQ',
        'internvl3_8b': 'OpenGVLab/InternVL3-8B',
        'internvl3_9b': 'OpenGVLab/InternVL3-9B'
    }
    return map[model]

def ability_map(ability):
    '''
    Function to map abilities to their acronyms
    '''
    map = {
        'visual_assembly_recognition': 'var',
        'safety_symbol_interpretation': 'ssi',
        'tools_and_accessary_identification': 'tai'
    }
    return map[ability]
    
def prompt_map(ability):
    '''
    Function to map ability with prompt
    '''
    # Prompt for visual assembly recognition
    var_prompt = ''' Choose from the given options. Explain your reasoning. Note: - Respond only in valid json. - Example Response: {'answer': 'A', 'reasoning': 'Explain your reasoning here.' }'''

    # Prompt for safety symbol interpretation
    ssi_prompt = '''What safety warning or hazard is shown in this image? Explain your reasoning. Note: - Respond only in valid json. - Example Response: {'answer': 'State the warning here', 'reasoning': 'Explain your reasoning here'}'''

    # Prompt for tools and accessary identification 
    tai_prompt = ''' Explain your reasoning. Note: - Respond only in valid json. - Example Response: {'answer': 'State Tool/Accessary name here', 'reasoning': 'Explain your reasoning here'}'''

    map = {
        'visual_assembly_recognition': var_prompt,
        'safety_symbol_interpretation': ssi_prompt,
        'tools_and_accessary_identification': tai_prompt
    }
    return map[ability]

# new clean reponse function which handles ovis response also
def clean_response(response):
    '''
    Function to extract cleaned json from vlm response
    '''
    try:
        # Handle markdown code blocks
        if '```' in response:
            # Extract content between triple backticks
            json_match = re.search(r'```(?:json)?\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                response = json_match.group(1).strip()
                print(f"Extracted JSON from markdown code block: {response[:50]}...")
        
        # Try to parse the entire response as JSON first
        try:
            parsed_json = json.loads(response)
            # Ensure we have the required fields
            if "answer" in parsed_json and "reasoning" in parsed_json:
                return parsed_json  # Return the parsed object, not a string
        except json.JSONDecodeError:
            # If not valid JSON, look for JSON objects embedded in text
            json_pattern = r'\{[\s\S]*?\}'
            json_matches = re.findall(json_pattern, response)
            
            for json_str in json_matches:
                try:
                    parsed_json = json.loads(json_str)
                    # Check if this JSON has our required fields
                    if "answer" in parsed_json and "reasoning" in parsed_json:
                        print(f"Extracted embedded JSON object: {json_str[:50]}...")
                        return parsed_json
                except json.JSONDecodeError:
                    # Try multiple approaches to fix the JSON
                    
                    # First try: Handle single quotes in property values
                    try:
                        # Replace single quotes with double quotes, but need to be careful
                        # with nested quotes inside strings
                        fixed_json = fix_single_quotes(json_str)
                        parsed_json = json.loads(fixed_json)
                        if "answer" in parsed_json and "reasoning" in parsed_json:
                            print(f"Fixed single quotes in JSON: {fixed_json[:50]}...")
                            return parsed_json
                    except (json.JSONDecodeError, Exception) as e:
                        print(f"Single quote fix failed: {str(e)[:50]}...")
                    
                    # Second try: Fix unquoted keys and replace single quotes
                    try:
                        fixed_json = json_str.replace("'", '"')
                        fixed_json = re.sub(r'(\w+):', r'"\1":', fixed_json)
                        
                        parsed_json = json.loads(fixed_json)
                        if "answer" in parsed_json and "reasoning" in parsed_json:
                            print(f"Fixed and extracted embedded JSON: {fixed_json[:50]}...")
                            return parsed_json
                    except json.JSONDecodeError:
                        # Continue to the next match
                        continue
            
            # If we got here, we couldn't find valid JSON in the matches
            # Try to fix the entire response as a last resort
            try:
                fixed_response = fix_single_quotes(response)
                parsed_json = json.loads(fixed_response)
                if "answer" in parsed_json and "reasoning" in parsed_json:
                    print(f"Fixed invalid JSON with advanced fix: {fixed_response[:50]}...")
                    return parsed_json
            except (json.JSONDecodeError, Exception):
                # If that fails, try the simpler approach
                fixed_response = response.replace("'", '"')
                fixed_response = re.sub(r'(\w+):', r'"\1":', fixed_response)
                
                try:
                    parsed_json = json.loads(fixed_response)
                    if "answer" in parsed_json and "reasoning" in parsed_json:
                        print(f"Fixed invalid JSON with simple fix: {fixed_response[:50]}...")
                        return parsed_json
                    else:
                        print("Fixed JSON missing required fields (answer and reasoning)")
                        return fixed_response
                except json.JSONDecodeError:
                    print(f"Couldn't find or fix valid JSON: {response[:50]}...")
                    return response
                
    except Exception as e:
        print(f"Error in clean_response: {e}")
        return response

def fix_single_quotes(text):
    """
    Advanced function to properly handle single-quoted JSON.
    Handles nested quotes and properly formats the JSON.
    """
    # Step 1: Replace property names in single quotes with double quotes
    result = re.sub(r"'(\w+)':", r'"\1":', text)
    
    # Step 2: Handle property names without quotes
    result = re.sub(r"(\w+):", r'"\1":', result)
    
    # Step 3: Replace single quotes for string values with double quotes
    # This is trickier because we need to handle apostrophes within the strings
    
    # Find all string values (content between single quotes)
    string_value_pattern = r"'([^']*(?:''[^']*)*)'"
    
    # Process each match
    def replace_string(match):
        content = match.group(1)
        # Replace any escaped single quotes within the string
        content = content.replace("''", "'")
        # Escape any double quotes in the content
        content = content.replace('"', '\\"')
        # Return with double quotes
        return f'"{content}"'
    
    result = re.sub(string_value_pattern, replace_string, result, flags=re.DOTALL)
    
    return result

def save_results(results, model, ability, output_dir='output/vlm_output'):
    file_name = f'result_{ability_map(ability)}.jsonl'
    save_path = os.path.join(output_dir, model, file_name)
    print(f"Saving evaluation results to {save_path}")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, 'w') as output:
        for result in results:
            output.write(json.dumps(result) + '\n')