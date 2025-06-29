## Script containing utility funcitons

import yaml
import datetime
import pandas as pd
import json, re, os
from typing import Dict, List, Any, Optional, Union
from openai import OpenAI
import ast

## Function to load yaml file
def load_config():
    yaml_file_path = "config.yaml"
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


## Function to append date_time to file
def append_date_time(file):
    # Get current date and time
    current_time = datetime.datetime.now()
    
    # Format it as a string (e.g., '22-10_14-30')
    timestamp = current_time.strftime("%d-%m_%H-%M")
    
    
    # Create the full filename by appending the timestamp
    filename = f"{file}_{timestamp}.jsonl"
    
    return filename

def load_output(file_name='17-03_08-43_test_mar_14.jsonl'):
    try:
        base_dir = '~/Master_thesis/final_evaluation/'
        file = os.path.join(base_dir, file_name)
        df = pd.read_json(file, lines=True)
        return df
    except ValueError as e:
        with open(file, "r") as f:
            for i, line in enumerate(f, 1):
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error on line {i}: {e}")

# Function to read the model response from dataframe, and extract answer and reasoning
# INPUT: vlm response text
# OUTPUT: vlm answer and reasoning
def extract_answer_reasoning(response_text, answer_key_name="answer", reason_key_name="reasoning", answer_type="open_ended"):
    # Initialize default values
    answer = None
    reasoning = None
    
    try:
        # Try to parse as JSON (handles both proper JSON and JSON-like strings)
        try:
            # Case 1: Proper JSON with double quotes
            data = json.loads(response_text)
            
            # Extract answer - handle both "Option (X)" and single letter formats
            if answer_key_name in data:
                answer_raw = data[answer_key_name]
                if answer_type == "mcq":
                    # If answer is just a letter like "A" or "Option (A)"
                    if isinstance(answer_raw, str):
                        # Extract just the letter if it's in "Option (X)" format
                        match = re.search(r'\(([A-D])\)', answer_raw)
                        if match:
                            answer = match.group(1)
                        else:
                            # If it's just the letter
                            answer = answer_raw.strip()
                else:
                    answer = answer_raw
            
            # Extract reasoning
            if reason_key_name in data:
                reasoning = data[reason_key_name]
                
        except json.JSONDecodeError:
            # Case 2: JSON-like with single quotes or other formatting issues
            # Try to extract using regex patterns
            answer_match = re.search(fr'"{answer_key_name}":\s*"?([^,}}"]+)"?', response_text)
            if not answer_match:
                answer_match = re.search(fr'"{answer_key_name}":\s*([^,}}]+)', response_text)
            
            if answer_match:
                answer_raw = answer_match.group(1).strip
                if answer_type == "mcq":
                    # Extract just the letter if it's in "Option (X)" format
                    option_match = re.search(r'\(([A-D])\)', answer_raw)
                    if option_match:
                        answer = option_match.group(1)
                    else:
                        # If it's just the letter
                        answer = answer_raw
                else:
                    answer = answer_raw
            
            reasoning_match = re.search(fr'"{reason_key_name}":\s*"([^"]+)"', response_text)
            if not reasoning_match:
                reasoning_match = re.search(fr'"{reason_key_name}":\s*"?(.*?)"?\s*\}}', response_text, re.DOTALL)
            
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
    
    except Exception as e:
        # Fall back to manual regex for severely malformed responses
        try:
            # Look for Option (X) pattern
            option_match = re.search(fr'{answer_key_name}\s*\(([A-D])\)', response_text)
            if option_match:
                answer = option_match.group(1)
            
            # Look for any reasoning following "reasoning:" or similar pattern
            reasoning_match = re.search(fr'{reason_key_name}"?:\s*"?(.*?)"?\s*(\}}|$)', response_text, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
        except:
            pass
    
    # Final cleanup
    if isinstance(answer, str):
        answer = answer.strip('"\'').strip()
    
    # Return extracted values
    return answer, reasoning

## Function to take vlm response and return cleaned version
def clean_vlm_response(df):
    # Apply the extract function to create new columns from vlm response
    df[['vlm_answer', 'vlm_reasoning']] = df['response'].apply(
        lambda x: pd.Series(extract_answer_reasoning(x))
    )
    return df

def clean_jsonl_file(input_file, output_file):
    """
    Clean a JSONL file by properly parsing the response field that may contain
    markdown code blocks or other formatting issues.
    
    Args:
        input_file (str): Path to the input JSONL file
        output_file (str): Path to the output cleaned JSONL file
    
    Returns:
        int: Number of records processed
    """
    cleaned_records = []
    records_processed = 0
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                # Parse the JSONL record
                record = json.loads(line.strip())
                records_processed += 1
                
                # Clean the response field if it contains markdown code blocks
                response = record['response']
                
                # Extract JSON from markdown code blocks if present
                if '```json' in response or '```' in response:
                    # Extract content between triple backticks
                    json_match = re.search(r'```(?:json)?\n(.*?)\n```', response, re.DOTALL)
                    if json_match:
                        extracted_json = json_match.group(1).strip()
                        record['response'] = extracted_json
                
                # If response is a JSON string (escaped JSON), parse it to ensure it's valid
                try:
                    # Try to parse the response as JSON to validate it
                    json.loads(record['response'])
                except json.JSONDecodeError:
                    # If it fails, attempt to fix common issues
                    # Convert single quotes to double quotes
                    fixed_response = record['response'].replace("'", '"')
                    try:
                        # Check if the fixed version is valid JSON
                        json.loads(fixed_response)
                        record['response'] = fixed_response
                    except json.JSONDecodeError:
                        # If still invalid, keep original but log it
                        print(f"Warning: Couldn't fix response in record {record['id']}")
                
                cleaned_records.append(record)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                print(f"Problematic line: {line[:100]}...")
    
    # Write the cleaned records to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in cleaned_records:
            f.write(json.dumps(record) + '\n')
    
    print(f"Processed {records_processed} records, saved {len(cleaned_records)} to {output_file}")
    return records_processed

def clean_json(input_txt):
    """
    Clean a JSON text by properly parsing the response field that may contain
    markdown code blocks or other formatting issues.
    
    Args:
        input_txt (str): input json text
    
    Returns:
        str: cleaned json text
    """
    try:
        # Parse the JSON record
        response = input_txt.strip()
        extracted_json = {}
        
        # Extract JSON from markdown code blocks if present
        if '```json' in response or '```' in response:
            # Extract content between triple backticks
            json_match = re.search(r'```(?:json)?\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                extracted_json = json_match.group(1).strip()
        
        # If response is a JSON string (escaped JSON), parse it to ensure it's valid
        try:
            # Try to parse the response as JSON to validate it
            json.loads(response)
        except json.JSONDecodeError:
            # If it fails, attempt to fix common issues
            # Convert single quotes to double quotes
            fixed_response = response.replace("'", '"')
            try:
                # Check if the fixed version is valid JSON
                json.loads(fixed_response)
                extracted_json = fixed_response
            except json.JSONDecodeError:
                # If still invalid, keep original but log it
                print("Warning: Couldn't fix response in input")
        
        # Return the cleaned record as JSON string
        return json.dumps(extracted_json)
    
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Problematic text: {input_txt[:100]}...")
        # Return the original text if parsing fails
        return input_txt

def max_pooling(judges_list: List[str], ability: str, data_dir: str, output_file: Optional[str] = None) -> str:
    """
    Calculate final evaluation result from panel of judges using max pooling.
    Args:
        judges_list (List[str]): List of judges name
        output_file (Optional[str]): Path to save the pooled results, defaults to auto-generated name
            
    Returns:
        str: Path to the saved pooled evaluation file
    """
    # Set base dir of output files
    #base_dir = 'output/llm_judge_output'
    base_dir = data_dir
    input_file_name = f'eval_result_{ability}.jsonl'
    
    if output_file is None:
        output_file_name = f"pooled_results_{ability}.jsonl"
    else:
        output_file_name = output_file
    
    output_file_path = os.path.join(base_dir, output_file_name)
    all_results = [] # list to store all the judges result
    id_to_results = {} # Dict to group results by example id
    
    # Load all the files
    for judge in judges_list:
        file_path = os.path.join(base_dir, judge, input_file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                results = [json.loads(line) for line in f]
                all_results.append(results)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if not all_results:
        raise ValueError("No valid judge files could be loaded")
    
    # Group results by example id
    for judge_results in all_results:
        for result in judge_results:
            example_id = result.get('id')
            if example_id not in id_to_results:
                id_to_results[example_id] = []
            id_to_results[example_id].append(result)
    
    # Perform max pooling and create final results
    pooled_results = []
    for example_id, results in id_to_results.items():
        pooled_result = {}
        # Extract all 'eval_result' values for this example
        eval_results = [r.get('llm_eval', '').get('answer', '').lower() for r in results]
        
        # Count 'true' vs 'false' votes
        true_count = eval_results.count('true')
        false_count = eval_results.count('false')
        
        # Apply max pooling (majority vote)
        if true_count >= false_count:
            pooled_result['pooled_eval_result'] = 'True'
        else:
            pooled_result['pooled_eval_result'] = 'False'
        
        # Add voting stats and id
        pooled_result['id'] = example_id
        pooled_result['true_votes'] = true_count
        pooled_result['false_votes'] = false_count
        pooled_result['total_votes'] = len(results)
        
        # Add judges that were used
        pooled_result['judges_used'] = judges_list
        pooled_results.append(pooled_result)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file_path)), exist_ok=True)
    
    # Save as JSONL (one JSON object per line)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for result in pooled_results:
            f.write(json.dumps(result) + '\n')
    
    # Calculate statistics
    correct_count = sum(1 for r in pooled_results if r['pooled_eval_result'] == 'True')
    total_count = len(pooled_results)
    incorrect_count = total_count - correct_count
    accuracy = (correct_count / total_count)*100 if total_count > 0 else 0
    
    print("\n--- Pooled Evaluation Results ---")
    print(f"Ability Evaluated: {ability}")
    print(f"Total examples: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy} %")
    print(f"Results saved to: {output_file_path}")
    
    return accuracy, correct_count, incorrect_count, total_count

####### Function using chatgpt #########

client = OpenAI(api_key = "")

input_path = "result_var_2.jsonl"
output_path = "formatted_vlm_response.jsonl"

def is_valid_json_response(text):
    if isinstance(text, dict):
        return "answer" in text and "reasoning" in text
    elif isinstance(text, str):
        try:
            parsed = ast.literal_eval(text)  # Safe parsing for single-quote dicts
            return isinstance(parsed, dict) and "answer" in parsed and "reasoning" in parsed
        except (ValueError, SyntaxError):
            return False
    return False

def extract_answer_reasoning(raw_response):
    system_msg = (
        "You will be given a one to two word name of a tool or list of tools or component. Your task is to extract the tool name "
        "and reasoning in valid JSON format like: {'answer': 'State the tool name or names here', 'reasoning': 'Explain your reasoning here.'}. "
        "Note: keep the 'reasoning' as close to the original input as possible while extracting."
    )
    user_msg = f"Raw response: {raw_response}\n\nRespond only with valid JSON."

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0,
    )

    reply = response.choices[0].message.content.strip()
    try:
        parsed = ast.literal_eval(reply)
        return parsed
    except json.JSONDecodeError:
        print("❌ Failed to parse GPT output as JSON.")
        print("📝 Raw GPT output:", reply)
        return None

def correct_response_format(input_path):
  temp_path = input_path + ".tmp"
  with open(input_path, "r", encoding="utf-8") as infile, open(temp_path, "w", encoding="utf-8") as outfile:
      for i, line in enumerate(infile, 1):
          data = json.loads(line)
          item_id = data.get("id", f"line_{i}")

          if is_valid_json_response(data["response"]):
              print(f"✅ Skipping already formatted response for ID: {item_id}")
              outfile.write(json.dumps(data) + "\n")
              continue

          print(f"🔄 Processing ID: {item_id}...")
          formatted = extract_answer_reasoning(data["response"])

          if formatted:
              print(f"✅ Successfully extracted for ID: {item_id}")
              data["formatted_response"] = formatted
          else:
              print(f"⚠️ Extraction failed for ID: {item_id}")
              data["formatted_response"] = {"answer": None, "reasoning": "Could not extract reasoning."}

          outfile.write(json.dumps(data) + "\n")
  # Step 2: Replace original file with temp
  os.replace(temp_path, input_path)
  return input_path

def merge_response_fields(data):
    """
    Merge 'formatted_response' into 'response' if it's valid.
    Overwrites 'response' with the cleaned JSON format if available.
    Removes 'formatted_response' after merging.
    """
    if isinstance(data.get("formatted_response"), dict):
        formatted = data["formatted_response"]
        if "answer" in formatted and "reasoning" in formatted:
            data["response"] = formatted  # overwrite
            data.pop("formatted_response", None)  # remove old field
    return data

def merge_and_save(input_path):
  temp_path = input_path + ".tmp"

  # Step 1: Read, merge, write to temp file
  with open(input_path, "r", encoding="utf-8") as infile, \
      open(temp_path, "w", encoding="utf-8") as outfile:
      for line in infile:
          data = json.loads(line)
          merged = merge_response_fields(data)
          outfile.write(json.dumps(merged) + "\n")

  # Step 2: Replace original file with temp
  os.replace(temp_path, input_path)
  print(f"✅ Merged responses written back to: {input_path}")

def correct_vlm_response(vlm_name, ability, data_dir='output'):
    base_dir = f"{data_dir}/vlm_output"
    input_file_name = f'result_{ability}.jsonl'
    file_path = os.path.join(base_dir, vlm_name, input_file_name)
    formatted_file_path = correct_response_format(file_path)
    merge_and_save(formatted_file_path)
