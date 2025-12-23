from typing  import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class Router:
    def __init__(self, model_registry, model_name, available_agents:Optional[dict]=None, remember_turn:int=2, max_generate_token:int=256):
        loaded = model_registry(model_name)
        self.tokenizer = loaded.tokenizer
        self.model = loaded.model
        self.available_agents = {} if not available_agents else available_agents
        self.remember_turn = remember_turn
        self.max_generate_token=max_generate_token

    def build_system_prompt(self, language = 'Korean'):
        agent_descript_str = "\n".join([f"- {name}: {agent.description}" for name, agent in self.available_agents.items()])
#         return f"""
# You are the Router.
# Do NOT answer the user. Output JSON only.

# Available Agents:
# {agent_descript_str}

# Task:
# - Split the user request into tasks (one objective per task). 
# - For each task: choose one Agent from the Available Agents.
# - If the request is ambiguous, set needs_clarification=true and tasks=[], and ask the user via clarifying_question in {language}.
# - If a task explicitly requested by the user falls within the scope of a specific Agent, you MUST use that Agent.
# - If changeable or time-sensitive information (e.g., role, price, ranking, recent events) is required, you MUST use the Search Agent.
# - For requests about time-independent concepts (e.g., definitions, principles, theories, etc.), set direct_answer_possible=true and use Final Answer Agent.
# - Write objective in English and {language} only.

# Return exactly this JSON structure:
# {{
#   "direct_answer_possible": true|false,
#   "needs_clarification": true|false,
#   "clarifying_question": "",
#   "tasks": [
#     {{
#       "objective": "",
#       "agent": ""
#     }}
#   ]
# }}
# """.strip()
        return f"""
You are the Router.
Do NOT answer the user. Output JSON only.

Available Agents:
{agent_descript_str}

Task:
- Select ALL agent types that are required from the Available Agents based on the user request.
- If the request is ambiguous, set route=clarification and ask the user via clarifying_question in {language}.
- If a task explicitly requested by the user falls within the scope of a specific Agent, you MUST use that Agent.
- If changeable or time-sensitive information (e.g., role, price, ranking, recent events) is required, you MUST use the Search Agent.
- For requests about time-independent concepts (e.g., definitions, principles, theories, etc.), set route=direct_answer and use only Final Answer Agent.
- If using one agent, set route="single_agent".
- If using multi-agent, set route="planner".
- Write high_level_intent in English only.

Return exactly:
{{
  "route": "",
  "using_agents": [],
  "high_level_intent": "",
  "clarifying_question": ""
}}
""".strip()

    def generate(self, user_input:str, history:list, language="Korean"):
        messages = history[max(0, len(history) - self.remember_turn*2):]+[{'role': "user", "content": user_input}]
        if messages[0]['role']=="system":
            messages = messages[1:]
        messages = [{'role': "system", 'content': self.build_system_prompt(language)}] + messages
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(f"\n---------------------------- [INPUT] ----------------------------\n{input_text}\n")

        inputs = self.tokenizer(
          input_text,
          return_tensors='pt'
        )

        output = self.model.generate(
            **inputs.to(DEVICE),
            eos_token_id = self.tokenizer.eos_token_id,
            pad_token_id = self.tokenizer.eos_token_id,
            max_new_tokens = self.max_generate_token,
            # temperature=0.1
        )
        
        generated_output = output[0][len(inputs.input_ids[0]):].tolist()
        output_text = self.tokenizer.decode(generated_output, skip_special_tokens=True)
        print("\n---------------------------- [OUTPUT] ----------------------------\n")
        print(output_text)
        print(f"{len(generated_output)=}\n")

        return output_text

class Planner:
    def __init__(self, model_registry, model_name, remember_turn:int=2, max_generate_token:int=512):
        loaded = model_registry(model_name)
        self.tokenizer = loaded.tokenizer
        self.model = loaded.model
        self.remember_turn = remember_turn
        self.max_generate_token = max_generate_token

    # def build_system_prompt(self, language='Korean'):


    # def generate(user_input, router_output, history, language):

    



class LoadLLM:
    def __init__(self, model:AutoModelForCausalLM, tokenizer:AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

class ModelRegistry:
    def __init__(self):
        self.model_dict={}

    def __call__(self, model_name):
        if model_name not in self.model_dict:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        dtype=torch.float16,
                        device_map=DEVICE,
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    )
            self.model_dict.update({model_name:LoadLLM(model=model, tokenizer=tokenizer)})
            
        return self.model_dict[model_name]

