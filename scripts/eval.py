from app.llm_judge import LLMJudge
from app.utils import max_pooling, load_output
from tqdm.auto import tqdm
import os, json

# Define PoLL Judges
poll_judges = ['phi', 'mistral', 'qwen']
abilities = ['visual_assembly_recognition', 'safety_symbol_interpretation', 'tools_and_accessary_identification']

def run_poll(vlm_name, ability_name, data_dir = "", **kwargs):
    """
    Run evaluation using multiple LLM judges sequentially.
    
    Args:
        vlm_name (str): Name of vlm whose response needs to be evaluated
        ability_name (str): Name of ability that needs to be evaluated
            
    Returns:
       Str: Path to saved PoLL output
    """
    print(f"=== Starting PoLL evaluation for VLM: {vlm_name} and Ability: {ability_name}===")

    # Get the vlm response file as input file
    input_file = f"{data_dir}/vlm_output/{vlm_name}/result_{ability_map(ability_name)}.jsonl"

    # define output dir
    output_dir = os.path.join(data_dir, 'llm_judge_output', vlm_name)
    
    result_files = []
    
    for model_name in poll_judges:
        print(f"\n=== Starting evaluation with {model_name} ===")
        
        # Initialize judge with current model
        judge = LLMJudge(judge_name=model_name, verbose=True, ability=ability_map(ability_name), output_dir= output_dir, **kwargs)
        
        # Load evaluation data
        judge.load_eval_data(input_file)
        
        # Load judge model
        judge.load_judge()
        
        # Run evaluation and save results
        results = judge.run_eval()
        
        
        # Clean up resources
        del judge
        import gc
        gc.collect()
        
        print(f"=== Completed evaluation with {model_name} ===\n")

    ## Run max pooling on the judges results
    accuracy, score, incorrect, total = max_pooling(poll_judges, ability_map(ability_name), data_dir = output_dir)
    print(f"VLM model evaluated: {vlm_name}")
    print("-----------------------------\n")
    return accuracy, score, incorrect, total

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

def run_exact_match(vlm_name, ability, data_dir, **kwargs):
    """
    Run exact match evaluation
    
    Args:
        vlm_name (str): Name of vlm whose response needs to be evaluated
        ability_name (str): Name of ability that needs to be evaluated
            
    Returns:
       Str: Path to saved evaluation output
    """
    print(f"=== Starting exact match evaluation for VLM: {vlm_name} and Ability: {ability}===")

    # Get the vlm response file as input file
    input_file = f"{data_dir}/vlm_output/{vlm_name}/result_{ability_map(ability)}.jsonl"
    
    # load the data
    data = load_output(input_file)

    # eval score
    score = 0
    # new result array
    results = []
    
    # iterate over rows
    for idx, row in tqdm(data.iterrows()):
        result = False
        # get the vlm answer
        GEN_ANSWER = row['response']['answer']
        #get the gold answer
        GOLD_ANSWER = row['ground_truth']
        # compare them
        if GEN_ANSWER == GOLD_ANSWER:
            result = True
            score = score + 1

        result_row = row.to_dict()
        result_row['match'] = result
        results.append(result_row)

    # save eval to new file
    file_name = f"eval_result_{ability_map(ability)}.jsonl" #append_date_time("eval")
    base_dir = f'{data_dir}/exact_match'
    save_path = os.path.join(base_dir, vlm_name, file_name)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # calculate accuracy
    accuracy = (score / len(data))*100 if len(data) > 0 else 0
    incorrect = len(data) - score
    total = len(data)

    # Print final result
    print("\n--- Exact Match Evaluation Results ---")
    print(f"Total examples: {len(data)}")
    print(f"Correct: {score}")
    print(f"Accuracy: {accuracy} %")
    print(f"Results saved to: {save_path}")
    print("-----------------------------\n")
    return accuracy, score, incorrect, total
    

def run_eval(vlm_name, index=None, data_dir = "output", **kwargs):
    """
    Run evaluation on vlm output
    
    Args:
        vlm_name (str): vlm name, whose response needs to be evaluated
    """
    #for ability in abilities:
    ability = None
    abilities = ['visual_assembly_recognition', 'safety_symbol_interpretation', 'tools_and_accessary_identification'] # Add new abilities here
    accuracy, correct, incorrect, total = 0, 0, 0, 0
    if index is not None:
        ability = abilities[index]
        
        if ability == 'visual_assembly_recognition':
            accuracy, correct, incorrect, total = run_exact_match(vlm_name, ability, data_dir, **kwargs)
        elif ability == 'safety_symbol_interpretation':
           accuracy, correct, incorrect, total = run_poll(vlm_name, ability, data_dir, **kwargs)
        elif ability == 'tools_and_accessary_identification':
            accuracy, correct, incorrect, total = run_poll(vlm_name, ability, data_dir, **kwargs)
        else:
            print(f"Invalid ability: {ability}")
    else:
        for ability in abilities:
            if ability == 'visual_assembly_recognition':
                accuracy, correct, incorrect, total = run_exact_match(vlm_name, ability, data_dir, **kwargs)
            elif ability == 'safety_symbol_interpretation':
                accuracy, correct, incorrect, total = run_poll(vlm_name, ability, data_dir, **kwargs)
            elif ability == 'tools_and_accessary_identification':
                accuracy, correct, incorrect, total = run_poll(vlm_name, ability, data_dir, **kwargs)
            else:
                print(f"Invalid ability: {ability}")       
    return accuracy, correct, incorrect, total