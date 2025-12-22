from typing  import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import file_utils, functions
import Agents

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class Router:
    def __init__(self, model_name, available_agents:Optional[list]=None, remember_turn:int=2, max_generate_token:int=256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=DEVICE,
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            )
        self.available_agents = [] if not available_agents else available_agents
        self.remember_turn = remember_turn
        self.max_generate_token=max_generate_token

    def build_system_prompt(self, language = 'Korean'):
        agent_descript_str = "\n".join([f"- {agent.name}: {agent.description}" for agent in self.available_agents])
        return f"""
You are the Router.
Decide which Agent should handle the user's request.
Do NOT answer the user. Output JSON only.

Available Agents:
{agent_descript_str}

Task:
- Split the user request into tasks (one objective per task).
- For each task: set task_id as "t1","t2",... and choose exactly ONE agent from the Available Agents.
- If ambiguous, set needs_clarification=true and add into clarifying_question in {language}.
- If the information is changeable or time-sensitive (e.g., organization, role, price, ranking, status, recent events), or when the user explicitly requests a search, MUST use Search Agent.
- For requests about time-independent concepts (e.g., definitions, principles, theories, etc.), set direct_answer_possible=true and use only Final Answer Agent.
- Write objective in English only.

Return exactly this JSON structure:
{{
  "direct_answer_possible": true|false,
  "needs_clarification": true|false,
  "clarifying_question": "",
  "tasks": [
    {{
      "task_id": "",
      "objective": "",
      "agent": "",
    }}
  ]
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
        
        print("\n---------------------------- [OUTPUT] ----------------------------\n")
        print(self.tokenizer.decode(output[0], skip_special_tokens=False))
        print("\n----------------------------------------------------------------------------------------------------------\n")
        print(self.tokenizer.decode(generated_output, skip_special_tokens=False))
        print(print(f"{len(generated_output)=}\n"))

# class Planner:




class Orchestrator:
    """
    전체 흐름 관리하는 모듈
    """
    



