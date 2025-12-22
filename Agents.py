from abc import ABC

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from tools import search_tool

class Agent(ABC):
    name: str
    description: str

    def __init__(self, model_name:str, **kwargs):
        pass

class DummyAgent(Agent):
    def __init__(self, model_name, name, description):
        self.model_name = model_name
        self.name = name
        self.description = description

class SearchAgent(Agent):
    name = "Search Agent"
    description = "Search the web for the information."
    # (
    #     "Use this agent when answering the question requires checking current or "
    #     "changing facts, numeric values, or official sources, even if the question "
    #     "does not explicitly mention recency."
    # )

    def __init__(self, model_name, remember_turn=2):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            # device_map="auto",
        ).to("cpu")     # 사용할 때마다 GPU에 올려서 쓰기
        self.remember_turn = remember_turn
    
    def build_system_prompt(self, language='Korean'):
        return f"""
You are a search-decision agent.

You MUST NOT answer the user's question.
You ONLY decide whether to 'search' or 'finish'.

Output rules:
- Output EXACTLY one valid JSON object and nothing else.

Valid output format:
{{
  "thought": "Reason for the decision"
  "action": 'search' or 'finish'
  "action_input": {{}}  // parameters
}}

Language rules:
- "thought" must be answered in {language}.
- Never use any language other than {language} or English.

Entity handling:
- Treat each requested entity(e.g., people, object) independently.
- Each search query MUST target exactly ONE entity.
- Queries must be short, keyword-based, and focused.



"""

    def generate(self, user_input, history, max_repeat=3):
        history = history[max(0, len(history)-2*self.remember_turn):]   # 사용할 history만 뽑기
        




class FinalAnswerAgent(Agent):
    name = "Final Answer Agent"
    description = "Generates a final answer."

    def __init__(self, model_name, remember_turn=2):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            # device_map="auto",
        ).to("cpu")
        self.remember_turn = remember_turn



        
class EmailAgent(Agent):
    name = "Email Agent"
    description = (
        # "Use this agent If you need to read emails"
    )

    def __init__(self,model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            # device_map="auto",
        ).to("cpu")
