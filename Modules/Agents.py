from abc import ABC
import json

import torch

from utils import functions
from tools.search_tool import WebSearchTool
from tools.email_tool import EmailReadTool, EmailSendTool

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
        tool_str = functions.dumps_json({'name': self.search_tools.name, 'description': self.search_tools.description, 'args_schema': self.search_tools.args_schema})
        return f"""
You are a search-decision agent.

You NEVER generate answers to the user's request.
You ONLY decide whether to call a tool or 'finish'.

You can use the following tool:
{tool_str}

Output rules:
- Output EXACTLY one valid JSON object and nothing else.

Valid output format:
{{
  "action": 'search' or 'finish',
  "entity": "ONE entity only",
  "action_input": {{}}  // parameters go here
  "thought": "Reason for the 'action'",
}}

Language rules:
- "thought" must be answered in {language}.
- Never use any language other than {language} or English.

""".strip()

    def generate(self, user_input, history, max_repeat=3, language='Korean'):
        history = history[max(0, len(history)-2*self.remember_turn):] + [{'role': 'user', 'content': user_input}]   # 사용할 history만 뽑기
        sys_prompt = {'role': 'system', 'content': self.build_system_prompt(language)}
        if history[0]['role']=='system':
            history = history[1:]
        messages = [sys_prompt] + history

        observation = None
        result = []
        for _ in range(max_repeat):
            if observation is not None:
                messages.append({'role':'tool', 'content':observation})

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

            messages.append({'role':'assistant', 'content':output_text})

            try:
                output_dict = functions.loads_json(output_text)
            except json.JSONDecodeError:
                observation = None
                messages += [
                    {
                        "role": "user",
                        "content": f"Not JSON. Respond again with ONLY one JSON object.",
                    }
                ]
                continue

            action = output_dict.get("action")
            if action.strip().lower() == "finish":
                break

            action_input: dict[str, str] = output_dict.get("action_input")
            
            # 도구 사용 성공했을 때 만 result에 저장.
            try:
                observation = self.search_tools(**action_input)
                print("---------- OBSERVATION ---------- \n", observation, "\n")
            except Exception as e:
                observation = functions.dumps_json(
                    {
                        "ok": False,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "retryable": True,
                    }
                )
                # messages+=[{'role': 'user', 'content': f"action={action} \naction_input={action_input} \n[tool call error] {e}\n\n Please answer again."}]
                continue

            result.append(observation)
            observation = functions.dumps_json({"ok": True, "results": observation})

        if not result:
            return None

        return result[-1]



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
    description = "Handle email-related tasks such as reading and writing emails."

    def __init__(self, model_registry, model_name):
        loaded = model_registry(model_name)       
        self.tokenizer = loaded.tokenizer
        self.model = loaded.model
