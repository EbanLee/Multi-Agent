from abc import ABC

import torch

from tools.search_tool import WebSearchTool

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

    def __init__(self, model_registry, model_name, remember_turn=2, max_generate_token=128):
        loaded = model_registry(model_name)
        self.tokenizer = loaded.tokenizer
        self.model = loaded.model
        self.remember_turn = remember_turn
        self.max_generate_token = max_generate_token
        self.search_tools = WebSearchTool()
    
    def build_system_prompt(self, language='Korean'):
        return f"""
You are a search-decision agent.

You MUST NOT answer the user's question.
You ONLY decide whether to 'search' or 'finish'.

Search Tool description:
{self.search_tools.description}

Output rules:
- Output EXACTLY one valid JSON object and nothing else.

Valid output format:
{{
  "thought": "Reason for the decision",
  "action": 'search' or 'finish',
  "entity": "ONE entity only",
  "action_input": {{}}  // parameters
}}

Language rules:
- "thought" must be answered in {language}.
- Never use any language other than {language} or English.

Entity handling:
- Treat each requested entity separately. (e.g., each person, each object)
- Query must be short, keyword-based, and focused.

Finish rules:
- Choose "finish" when all requested entities can be answered with known information.
- If information is insufficient after a search, refine the query before retrying.
""".strip()

    def generate(self, user_input, history, max_repeat=3, language='Korean'):
        history = history[max(0, len(history)-2*self.remember_turn):] + [{'role': 'user', 'content': user_input}]   # 사용할 history만 뽑기
        sys_prompt = {'role': 'system', 'content': self.build_system_prompt(language)}
        if history[0]['role']=='system':
            history = history[1:]
        messages = [sys_prompt] + history

        for _ in range(max_repeat):
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            print(f"\n---------------------------- [INPUT] ----------------------------\n{input_text}\n")

            inputs = self.tokenizer(
                input_text,
                return_tensors='pt',
            )

            output = self.model.generate(
                **inputs.to(DEVICE),
                max_new_tokens = self.max_generate_token,
                eos_token_id = self.tokenizer.eos_token_id,
                pad_token_id = self.tokenizer.pad_token_id
            )

            generated_output = output[0][len(inputs.input_ids[0]):].tolist()
            output_text = self.tokenizer.decode(generated_output, skip_special_tokens=True)
            print("\n---------------------------- [OUTPUT] ----------------------------\n")
            print(output_text)
            print(f"{len(generated_output)=}\n")

            return output_text



class FinalAnswerAgent(Agent):
    name = "Final Answer Agent"
    description = "Generates a final answer."

    def __init__(self, model_registry, model_name, remember_turn=2):
        loaded = model_registry(model_name)       
        self.tokenizer = loaded.tokenizer
        self.model = loaded.model
        self.remember_turn = remember_turn




        
class EmailAgent(Agent):
    name = "Email Agent"
    description = (
        # "Use this agent If you need to read emails"
    )

    def __init__(self, model_registry, model_name):
        loaded = model_registry(model_name)       
        self.tokenizer = loaded.tokenizer
        self.model = loaded.model
