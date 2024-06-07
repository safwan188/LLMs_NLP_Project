import os
import json
import time
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import textwrap
from IPython.display import display, Markdown


def count_tokens(string, encoding_name="cl100k_base"):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


def query_gemini_api(system_prompt, user_prompt,model_name):
    genai.configure(api_key='api_key_here')
    model = genai.GenerativeModel(model_name)

    
    # Gemini does not support system prompts directly, so we include the system prompt in the user prompt
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    start_time = time.time()  # Start time before the API call

    response = model.generate_content(combined_prompt)
    
    end_time = time.time()  # End time after the API call
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(elapsed_time)
    return response.text, elapsed_time


def create_and_evaluate_llm(file_path, output_path,model_name):
    with open(file_path, 'r') as file:
        test_cases = json.load(file)
    
    results = []
    system_prompt = """
             Here is a sentence with pronouns tagged:

<tagged_sentence>
{{TAGGED_SENTENCE}}
</tagged_sentence>

Your task is to resolve the pronouns in this sentence, replacing each pronoun with the specific noun it refers to based on the context of the sentence.

To do this, follow these steps:

1. Read the sentence carefully, paying attention to the overall meaning and the relationships between different entities mentioned.

2. Look at each pronoun enclosed in <PR> tags:
   a. Determine which noun that pronoun is referring to, based on the context.
   b. Replace the pronoun and its surrounding <PR> tags with that noun.

3. Do not make any other changes to the sentence. Your only task is to replace the tagged pronouns with their referent nouns. All other words should be left exactly as they are.

4. After resolving all pronouns, output the complete sentence inside <resolved_sentence> tags.

For example, if the tagged sentence was "John took the ball and <PR>he</PR> kicked <PR>it</PR>", the resolved sentence would be:
<resolved_sentence>John took the ball and John kicked the ball</resolved_sentence>

Now please resolve the pronouns in the sentence provided.
                    """
    for test_case in test_cases:
        tagged_sentence = test_case['original_sentence']
        
        prompt = f"Here is the  tagged sentence you should resolve pronouns for: \n\n<tagged_sentence>{tagged_sentence}</tagged_sentence> Provide your output with no extra text, commentary or chat. Simply output the sentence with pronouns resolved. "
        
        llm_input = {
            "prompt": prompt,
            "completion": test_case['rewritten_sentence'],
            "complexity_level": test_case['complexity_level'],
        }

        user_prompt = llm_input["prompt"]
        
        model_output, execution_time = query_gemini_api(system_prompt, user_prompt,model_name)
        
        result = {
            "complexity_level": llm_input["complexity_level"],
            "prompt": llm_input["prompt"],
            "expected_output": llm_input["completion"],
            "model_output": model_output,
            "tagged_sentence": test_case['original_sentence'],
            "tokens_count": count_tokens(llm_input["prompt"]),
            "execution_time": execution_time
        }
        
        results.append(result)
        
        # Wait for 20 seconds before sending the next request to avoid rate limits for pro model 1.5
        time.sleep(20)
    
    with open(output_path, 'w') as outfile:
        json.dump(results, outfile, indent=4)
    
    return results





genai.configure(api_key='api_key_here')

for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)
# Example usage
models_names=['gemini-1.5-pro-latest']

models_names=['gemini-1.0-pro-latest', 'gemini-1.0-pro-001','gemini-1.0-pro','gemini-1.5-flash-latest',]
models_names=['gemini-1.5-pro-latest']


for model_name in models_names:
    print(f"Using model: {model_name}")
    results = create_and_evaluate_llm("x_new_generated.json", f"prompt5_anaphora_resolution_results_{model_name}.json", model_name)
    print(f"Results saved to anaphora_resolution_results_{model_name}.json")

