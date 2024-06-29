import os
import re
import pickle
import time
import requests
from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

def rate_limited(max_per_minute):
    min_interval = 60.0 / float(max_per_minute)
    def decorate(func):
        last_called = [0.0]
        def rate_limited_function(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            last_called[0] = time.time()
            return func(*args, **kwargs)
        return rate_limited_function
    return decorate

def get_tools_embeddings(items, args):
    tools_embedding = {}
    os.makedirs(f"./tools_emb/{args.embedding_model}", exist_ok=True)
    for sub_task in list(items.keys()):
        sub_task_pkl_path = f"./tools_emb/{args.embedding_model}/{sub_task}_task_tools_emb.pkl"
        if os.path.exists(sub_task_pkl_path):
            with open(sub_task_pkl_path, "rb") as f:
                tools_embedding[sub_task] = pickle.load(f)
        else:
            task_name, task_desc = [], []
            for item in items[sub_task]:
                tools = item["tools"]
                pattern = re.compile(r'\d+\.\s*([^\:：]+)[:：]\s*(.*)')
                matches = pattern.findall(tools)
                for match in matches:
                    tool_name, tool_desc = match
                    task_name.append(tool_name)
                    task_desc.append(tool_desc)
            if args.embedding_model == "minilm":
                tool_embeddings = args.emb_model.encode(task_desc)
            elif args.embedding_type == "gemini":
                tool_embeddings = []
                for desc in task_desc:
                    embedding = args.emb_model(desc)["embedding"]
                    tool_embeddings.append(embedding)
            else:
                raise Exception("Wrong embedding type")
            
            tools_embedding[sub_task] = {
                    "name": task_name,
                    "desc": task_desc,
                    "embeddings": tool_embeddings
                }
            with open(sub_task_pkl_path, "wb") as fw:
                pickle.dump(tools_embedding[sub_task], fw)
    
    return tools_embedding


class GeminiGeneration:
    def __init__(self, api_key, model_name):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
    @rate_limited(59)
    def generation_gemini(self, prompt):
        try:
            gemini_response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=int(self.model.count_tokens(prompt).total_tokens) + 1024,
                    temperature=0.0),
                safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    },
                )
            return gemini_response.text
        except Exception as e:
            if "429" in str(e):
                print("Need to switched key due to 429 Error.")
            else:
                print(f'Unknown error occurred: {e}')
            return ""

class GeminiEmbedding:
    def __init__(self, api_key, model_name="models/text-embedding-004"):
        genai.configure(api_key=api_key)
        self.model = model_name
        
    def get_embedding_gemini(self, prompt):
        result = genai.embed_content(
            model=self.model,
            content=prompt,
            task_type="semantic_similarity"
        )
        return result


class OpenAIGeneration:
    def __init__(self, api_key, model):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def generation_openai(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature = 0.0,
                max_tokens = 2048
            ) 
            return completion.choices[0].message.content
        except Exception as e:
            print(f'Unknown error occurred: {e}')
            return ""


class VllmGeneration:
    def __init__(self, api_url):
        self.api_url = api_url
    def generation_vllm(self, prompt):
        headers = {'Content-Type': 'application/json'}
        post_data = {
                'model': 'model',
                'messages': [{"role": "user", "content": prompt}],
                'max_tokens': 1024,
                'temperature': 0.0,
                'top_p': 0.95,
            }
        try:
            response = requests.post(self.api_url, headers=headers, json=post_data).json()
            result = response['choices'][0]['message']['content']
            return result
        except Exception as e:
            print(f'Error occurred: {e}')
            return ""
