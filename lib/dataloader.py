import json
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message=".*promote_options.*")
warnings.filterwarnings("ignore", message=".*precompiled_charsmap.*")

TSV_FILE_PATH = "./data/Train_GCC-training_with_header.tsv"
DEFAULT_PROMPT_FILE_PATH = "./data/flux2_prompts_100.json"

def get_gcc3m(num_samples):
    try:
        df = pd.read_csv(TSV_FILE_PATH, sep='\t', encoding='utf-8')
        print(f"Successfully read TSV file: {TSV_FILE_PATH}")
        print(f"Total number of rows: {len(df)}")
        print(f"Column names: {list(df.columns)}")
        
        prompts = df['caption'].head(num_samples).tolist()
        print(f"Extracted {len(prompts)} prompts from GCC3M")
        
        return prompts
    
    except FileNotFoundError:
        print(f"Error: File {TSV_FILE_PATH} does not exist.")
        return []
    except Exception as e:
        print(f"Error occurred while reading the file: {str(e)}")
        return []

def get_prompt_file(prompt_file, num_samples=50):
    prompt_file = prompt_file or DEFAULT_PROMPT_FILE_PATH
    try:
        with open(prompt_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict):
            prompts = payload.get("prompts", [])
        elif isinstance(payload, list):
            prompts = payload
        else:
            raise ValueError("Prompt file must contain either a JSON list or an object with a 'prompts' field.")

        prompts = [prompt.strip() for prompt in prompts if isinstance(prompt, str) and prompt.strip()]
        prompts = prompts[:num_samples]
        print(f"Successfully read prompt file: {prompt_file}")
        print(f"Extracted {len(prompts)} prompts from prompt file")
        return prompts

    except FileNotFoundError:
        print(f"Error: Prompt file {prompt_file} does not exist.")
        return []
    except Exception as e:
        print(f"Error occurred while reading the prompt file: {str(e)}")
        return []


def get_loaders(name, num_samples=50, prompt_file=None):
    if name == 'gcc3m':
        return get_gcc3m(num_samples)
    if name == 'prompt_file':
        return get_prompt_file(prompt_file, num_samples)
    raise ValueError(f"Unknown dataset: {name}")

