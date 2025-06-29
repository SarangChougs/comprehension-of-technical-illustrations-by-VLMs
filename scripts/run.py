## Main file, to run the CLI

import argparse
from app.vlm_inference import run_vlm
from app.eval import run_eval
from app.utils import correct_vlm_response
from datetime import datetime
import pandas as pd
import os
from tqdm import tqdm

#result_folder = 'without_context'
output_path = 'output/without_context'

def main():
    """Command line interface for the VLM Eval Runner."""
    parser = argparse.ArgumentParser(description='Run VLM and evaluate')
    parser.add_argument('-m', '--model', type=str, dest='model', help='model name', default=None)
    parser.add_argument('-t', '--test', action='store_true', help='Enable test mode')
    parser.add_argument('-v', '--vlm', action='store_true', help='Run VLM')
    parser.add_argument('-e', '--eval', action='store_true', help='Run Eval')
    parser.add_argument('-f', '--fix', action='store_true', help='Fix output')
    parser.add_argument('-ea', '--eval_all', action='store_true', help='Evaluate all')
    parser.add_argument('-rf', '--run_full', action='store_true', help='Run full')
    parser.add_argument('-a', '--ability', type=str, dest='ability', help='ability short form', default=None)
    
    args = parser.parse_args()

    models = [
        #'ovis16b' ,
        #'ovis8b' ,
        #'ovis4b',
        #'qwen2_5',
        #'internvl2-5_8b',
        'kimi',
        #'internvl3_14b',
        #'internvl3_8b',
        #'internvl3_9b'
    ]
    
    ability_index = None

    ## Get the ability index
    if args.ability == 'var':
        ability_index = 0
    if args.ability == 'ssi':
        ability_index = 1
    if args.ability == 'tai':
        ability_index = 2

    # Run evaluation, inference, fix or all
    if args.vlm:
        print("******* Running VLM *******")
        run_vlm(args.model, test=args.test, index=ability_index)
    if args.eval:
        print("******* Running evaluation ********")
        run_eval(args.model, index=ability_index, data_dir = output_path)
    if args.fix:
        print("******* Fixing VLM response ********")
        correct_vlm_response(args.model, args.ability, data_dir = output_path)
    if args.eval_all:
        print("******* Running Evaluation on all models ********")
        try:
            summaries = []
            for model in tqdm(models):
                summary = {
                    "Model": model,
                    "Accuracy": 0,
                    "Correct": 0,
                    "Incorrect": 0,
                    "Total": 0,
                    "Ability": args.ability
                } 
                # correct vlm response if needed
                print("******* Fixing VLM response ********")
                correct_vlm_response(model, args.ability, data_dir = output_path)
                
                # run evaluation on corrected vlm response
                print("******* Running evaluation ********")
                # create per model summary
                summary["Accuracy"], summary["Correct"], summary["Incorrect"], summary["Total"] = run_eval(model, index=ability_index, data_dir = output_path)
    
                # append summary to summaries
                summaries.append(summary)
        except Exception as e:
            print(f"**** Error Occured: {e}")
            print(f"******** {args.ability} Evaluation summary **********")
            # Create summary DataFrame
            summary_df = pd.DataFrame(summaries)
            summary_df = summary_df[['Model', 'Accuracy', 'Correct', 'Incorrect', 'Total', 'Ability']]
            file_name = "eval_summary.csv"
            file_path = os.path.join(output_path, file_name)
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            summary_df.to_csv(file_path, index=False)
            print("\nSummary Table:")
            print(summary_df)
            print(f"/n******** {args.ability} Evaluation summary end **********")
        
    if args.run_full:
        try:
            print(f"******* Running Inference and Evaluation on all models for {args.ability} ********")
            summaries = []
            for model in tqdm(models):
                summary = {
                    "Model": model,
                    "Accuracy": 0,
                    "Correct": 0,
                    "Incorrect": 0,
                    "Total": 0,
                    "Ability": args.ability
                }
                # get vlm response
                print("******* Running VLM *******")
                run_vlm(model, test=args.test, index=ability_index, output_dir = f"{output_path}/vlm_output")
                
                # correct vlm response if needed
                print("******* Fixing VLM response ********")
                correct_vlm_response(model, args.ability, data_dir = output_path)
                
                # run evaluation on corrected vlm response
                print("******* Running evaluation ********")
                # create per model summary
                summary["Accuracy"], summary["Correct"], summary["Incorrect"], summary["Total"] = run_eval(model, index=ability_index, data_dir = output_path)
    
                # append summary to summaries
                summaries.append(summary)
        except Exception as e:
            print(f"**** Error Occured: {e}")
            print(f"******** {args.ability} Evaluation summary **********")
            # Create summary DataFrame
            summary_df = pd.DataFrame(summaries)
            summary_df = summary_df[['Model', 'Accuracy', 'Correct', 'Incorrect', 'Total', 'Ability']]
            file_name = "eval_summary.csv"
            file_path = os.path.join(output_path, file_name)
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            summary_df.to_csv(file_path, index=False)
            print("\nSummary Table:")
            print(summary_df)
            print(f"/n******** {args.ability} Evaluation summary end **********")

if __name__ == "__main__":
    main()