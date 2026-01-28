# import aisuite as ai
# from dotenv import load_dotenv
import json
from tqdm import tqdm
import pandas as pd

from tasks.task2.prompting import StatementPrompting
from tasks.task2.utils import parse_json
# load_dotenv()  
import pdb
import random
import numpy as np

from tasks.hf_backend import HFChatModel
import argparse
hf_model = None

# MODEL = "openai:gpt-4o"
# MODEL = "openai:gpt-4o-mini"
# MODEL = "openai:gpt-3.5-turbo"
# MODEL = "anthropic:claude-sonnet-4-20250514"
# MODEL = "groq:DeepSeek-R1-Distill-Llama-70b"
# MODEL = "groq:llama-3.3-70b-versatile"
# MODEL = "groq:gemma2-9b-it"
# MODEL = "groq:qwen-qwq-32b"



# client = ai.Client()


def eval_value_action(country, topic, value, option1, option2):
    global hf_model
    prompting_method = StatementPrompting()

    outputs = {
        "country": country,
        "topic": topic,
        "value": value,
        "option1": option1,
        "option2": option2,
        "evaluation_0": None,
        "evaluation_1": None,
        "evaluation_2": None,
        "evaluation_3": None,
        "evaluation_4": None,
        "evaluation_5": None,
        "evaluation_6": None,
        "evaluation_7": None,
    }

    for prompt_index in tqdm(range(8)):
        action_prompt = prompting_method.generate_prompt(country=country, topic=topic, value=value, option1=option1, option2=option2, index=prompt_index)
        # response = client.chat.completions.create(
        #     model=MODEL,
        #     messages=[{"role": "user", "content": action_prompt[0]}],
        #     temperature=0.2
        # )
        text = hf_model.chat(action_prompt[0], temperature=0.2, max_new_tokens=256)
        try:
            r = parse_json(text)
        except:
            raise ValueError(f"Failed to parse response: {text}")
        outputs[f"evaluation_{prompt_index}"] = r
        print(f"{prompt_index} r={r}", )
    return outputs


def main():
    global hf_model
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id (remote) or local path")
    parser.add_argument("--out_csv", default="evaluation_results.csv")
    parser.add_argument("--sub_sample", action="store_true")
    args = parser.parse_args()
    hf_model = HFChatModel(args.model)

    sub_sample = False

    if sub_sample:
        sample_size = 50
        df = pd.read_csv("/home/zs473554/projects/value_action_gap/src/outputs/full_data/value_action_gap_full_data_gpt_4o_generation.csv")
        # Generate all even numbers from 2 to 10000
        even_numbers = list(range(2, len(df), 2))
        # Randomly sample 100 even numbers (without replacement)
        sampled_evens = random.sample(even_numbers, sample_size)
        sampled_odds = list(map(lambda x: x + 1, sampled_evens))
        index_list = np.sort(sampled_evens + sampled_odds)
        filtered_df = df.loc[index_list]
        filtered_df.to_csv(args.out_csv, index=False)
        
    else:
        df = pd.read_csv("/home/zs473554/projects/value_action_gap/src/outputs/filtered_sample_value_action_evaluation_gpt_4o_mini.csv")

        results = []
        # Group by country, topic, and absolute value to pair opposite polarities
        grouped = df.groupby(['country', 'topic', 'value'])

        for (country, topic, value), group in grouped:
            # Skip if we don't have exactly 2 rows (positive and negative polarity)

            if len(group) != 2:
                print("== skip len(group)!=1 ==")
                continue
                
            # Sort by value to ensure consistent ordering (negative first, positive second)
            group = group.sort_values('polarity')

            assert group.iloc[0]['polarity'] == "negative" and group.iloc[1]['polarity'] == "positive"
            
            # option1 = parse_json(group.iloc[0]['generation_prompt_id_5'])["Human Action"]  # negative polarity
            # option2 = parse_json(group.iloc[1]['generation_prompt_id_5'])["Human Action"]  # positive polarity\n")
            try:
                option1 = parse_json(group.iloc[0]['generation_prompt'])["Human Action"]  # negative polarity
                option2 = parse_json(group.iloc[1]['generation_prompt'])["Human Action"]  # positive polarity\n")
            except:
                continue
            outputs = eval_value_action(country=country, topic=topic, value=value, option1=option1, option2=option2)
            results.append(outputs)
            
            # print(f"========{country}-{topic}-{value} done \n")
            cases = [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]

            # prompts that led to option 1
            print("===========")
            print("OPTION 1\n")
            for i in range(8):
                if outputs[f"evaluation_{i}"]["action"] == "Option 1":
                    print(f"Prompt {cases[i]} \n")
            
            print("===========")
            print("OPTION 2\n")
            for i in range(8):
                if outputs[f"evaluation_{i}"]["action"] == "Option 2":
                    print(f"Prompt {cases[i]} \n")
            print("===========")


        result_df = pd.DataFrame(results)
        result_df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
