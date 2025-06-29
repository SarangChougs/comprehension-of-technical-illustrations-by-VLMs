import json
import os, sys, re
import argparse
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from transformers import pipeline
from tqdm.auto import tqdm

from app.utils import load_output, clean_vlm_response, clean_json, append_date_time

class LLMJudge:
    """A utility for evaluating VLM responses using an LLM judge."""
    
    def __init__(self, ability: str, output_dir: str, judge_name: str = 'phi', verbose: bool = False, test: bool = False):
        """
        Initialize the LLM judge.
        
        Args:
            judge_name (str): Name of the judge model to use
            verbose (bool): Whether to print detailed logs
        """
        self.judge_name = judge_name
        self.verbose = verbose
        self.judge = {
            'phi': 'stelterlab/phi-4-AWQ',
            'mistral': 'stelterlab/Mistral-Small-24B-Instruct-2501-AWQ',
            'qwen' : 'Qwen/Qwen2.5-32B-Instruct-AWQ'
        }
        # Judge Prompt for tools and accessary identificaiton
        #self.judge_prompt = '''You will be given a Reference answer and a Provided Answer. Judge whether the Provided Answer is correct by comparing it to the Reference Answer. Differently formatted answer, similar names, and alternative spellings should all be considered the same. Extra information is ok. If the Provided Answer is correct say exactly "True", otherwise say "False". Provide response in a valid following json structure {'answer': 'your answer', 'reason': 'your reason'}'''
        # Judge prompt for safety symbol recognition
        self.judge_prompt = '''You will be given a Reference answer and a Provided Answer. Judge whether the Provided Answer is correct by comparing it to the Reference Answer. The answers question related to safety information, hence while evaluating judge on the basis of semantic meaning and not exact match. Extra information is ok. If the Provided Answer is correct say exactly "True", otherwise say "False". Provide response in a valid following json structure {'answer': 'your answer', 'reason': 'your reason'}'''
        self.pipe = None
        self.data = None
        self.prompt = None
        self.results = []
        self.test = test
        self.ability = ability # added ability to save evaluation files with ability name
        self.output_dir = output_dir
    
    def log(self, message: str) -> None:
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            print(f'**** Verbose **** \n {message} \n*********')
    
    def load_judge(self) -> 'LLMJudge':
        """
        Load the LLM judge model.
        
        Returns:
            LLMJudge: The current instance for method chaining
        """
        self.log(f"Loading judge model: {self.judge.get(self.judge_name)}")
        self.pipe = pipeline(
            "text-generation", 
            model=self.judge.get(self.judge_name), 
            trust_remote_code=True, 
            max_new_tokens=728
        )
        return self
    
    def load_eval_data(self, eval_file: str = 'cleaned.jsonl') -> 'LLMJudge':
        """
        Load and prepare evaluation data.
        
        Args:
            eval_file (str): Path to the evaluation data file
            use_cleaner (bool): Whether to use JSONLCleaner for processing
            
        Returns:
            LLMJudge: The current instance for method chaining
        """
        self.log(f"Loading evaluation data from: {eval_file}")
        
        # Use simple loading function
        data = load_output(eval_file)
        #data = clean_vlm_response(raw_data)
        
        self.data = data
        return self
    
    def generate_prompt(self, data_row: Union[Dict[str, Any], pd.Series]) -> 'LLMJudge':
        """
        Generate a prompt for the judge model.
        
        Args:
            data_row (dict or pd.Series): Data row containing VLM answer and ground truth
            
        Returns:
            LLMJudge: The current instance for method chaining
        """
        #QUESTION = data_row['question']
        
        GEN_ANSWER = data_row['response']['answer']
        
        GOLD_ANSWER = data_row['ground_truth']
        
        data_prompt = f'''Provided Answer: "{GEN_ANSWER}"
Reference Answer: "{GOLD_ANSWER}"'''
        
        self.prompt = self.judge_prompt + data_prompt
        return self

    def save_eval(self, results: List[Dict[str, Any]]) -> str:
        """
        Save the evaluation results in jsonl file
        
        Args:
            results (list): List of evaluation result dictionaries
                
        Returns:
            str: Path to saved evaluation file
        """
        file_name = f"eval_result_{self.ability}.jsonl" #append_date_time("eval")
        #base_dir = 'output/llm_judge_output'
        base_dir = self.output_dir
        save_path = os.path.join(base_dir, self.judge_name, file_name)
        self.log(f"Saving evaluation results to {save_path}")
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        return save_path
    
    def evaluate_single(self, data_row: Union[Dict[str, Any], pd.Series]) -> str:
        """
        Evaluate a single data row.
        
        Args:
            data_row (dict or pd.Series): Data row to evaluate
            
        Returns:
            str: Evaluation result
        """
        self.generate_prompt(data_row)
        message = [{"role": "user", "content": self.prompt}]
        if self.test:
            print(f"Judge Input: \n {message[0]['content']}")
        output = self.pipe(message, temperature = 0.1)[0]['generated_text'][1]['content'].strip()
        
        return output
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse and clean JSON response from the judge model.
        
        Args:
            response (str): Raw response from the judge model
            
        Returns:
            dict: Parsed JSON object
        """
        # Try to parse the string directly
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to handle single quotes instead of double quotes
        try:
            # Replace single quotes with double quotes but be careful with nested quotes
            fixed_text = re.sub(r"'([^']*)':", r'"\1":', response)  # Fix keys
            fixed_text = re.sub(r": *'([^']*)'", r': "\1"', fixed_text)  # Fix string values
            fixed_text = fixed_text.replace("'", '"')  # Replace any remaining single quotes
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from code blocks
        code_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1).strip())
            except json.JSONDecodeError:
                # Try the single quote fix on the extracted content
                code_content = code_match.group(1).strip()
                try:
                    fixed_code = re.sub(r"'([^']*)':", r'"\1":', code_content)
                    fixed_code = re.sub(r": *'([^']*)'", r': "\1"', fixed_code)
                    fixed_code = fixed_code.replace("'", '"')
                    return json.loads(fixed_code)
                except json.JSONDecodeError:
                    pass
        
        # Try to extract based on curly braces
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            try:
                # Try with single quote fix
                braces_content = json_match.group(1)
                fixed_braces = re.sub(r"'([^']*)':", r'"\1":', braces_content)
                fixed_braces = re.sub(r": *'([^']*)'", r': "\1"', fixed_braces)
                fixed_braces = fixed_braces.replace("'", '"')
                return json.loads(fixed_braces)
            except json.JSONDecodeError:
                pass
        
        # Manual parsing as last resort
        try:
            answer_match = re.search(r"'answer':\s*'([^']*)'", response)
            reason_match = re.search(r"'reason':\s*'([^']*)'", response)
            
            if answer_match and reason_match:
                return {
                    "answer": answer_match.group(1),
                    "reason": reason_match.group(1)
                }
        except:
            pass
        
        # If all parsing attempts fail, create a fallback object
        self.log(f"Warning: Could not parse JSON response: {response[:100]}...")
        return {
            "answer": "Error",
            "reason": "Failed to parse response",
            "raw_response": response
        }
    
    def run_eval(self) -> Union[pd.DataFrame, Dict]:
        """
        Run evaluation on all data or test on a single example.
        
        Args:
            None
            
        Returns:
            pd.DataFrame or dict: DataFrame with evaluation results or single result dict
        """
        results = []
        if self.pipe is None:
            self.load_judge()
        
        if self.data is None:
            raise ValueError("No evaluation data loaded. Call load_eval_data first.")
        
        if self.test:
            # Test mode - evaluate first example only
            self.log("Running in test mode on first example")
            raw_result = self.evaluate_single(self.data.iloc[0])
            
            # Parse the result properly
            cleaned_result = self.parse_json_response(raw_result)
            print(f"Test result: {cleaned_result}")
            result_row = self.data.iloc[0].to_dict()
            result_row['llm_eval'] = cleaned_result
            results.append(result_row)
            # Save results
            self.save_eval(results)
            return result_row
        
        # Full evaluation
        self.log(f"Evaluating {len(self.data)} examples")
        
        
        for idx, row in tqdm(self.data.iterrows()):
            raw_result = self.evaluate_single(row)
            #self.log(f"Example {idx+1}/{len(self.data)}")
            
            # Parse the result properly
            cleaned_result = self.parse_json_response(raw_result)
            
            # Store result
            result_row = row.to_dict()
            result_row['llm_eval'] = cleaned_result
            results.append(result_row)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        self.save_eval(results)
        
        return results_df


def main():
    """Command line interface for the LLM judge."""
    parser = argparse.ArgumentParser(description='Evaluate VLM responses using an LLM judge')
    
    # Required arguments
    parser.add_argument('input_file', help='Path to the input JSONL file')
    
    # Optional arguments
    parser.add_argument('-j', '--judge', default='phi-4-awq', help='Name of the judge model to use')
    parser.add_argument('-r', '--results', help='Path to save evaluation results')
    parser.add_argument('-t', '--test', action='store_true', help='Run in test mode on a single example')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    judge = LLMJudge(judge_name=args.judge, verbose=args.verbose, test=args.test)
    judge.load_eval_data(args.input_file)
    judge.load_judge()
    judge.run_eval()


if __name__ == "__main__":
    main()