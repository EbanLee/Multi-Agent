from abc import ABC
import json
import copy

import torch

from utils import functions
from tools.search_tool import WebSearchTool

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Agent(ABC):
    name: str
    description: str

    def __init__(self, model_name:str, **kwargs):
        pass

    def generate(self, user_input, history, language='Korean'):
        pass

class DummyAgent(Agent):
    def __init__(self, model_name, name, description):
        self.model_name = model_name
        self.name = name
        self.description = description

class SearchAgent(Agent):
    name = "Search Agent"
    description = "fetch information from the web."
    # (
    #     "Use this agent when answering the question requires checking current or "
    #     "changing facts, numeric values, or official sources, even if the question "
    #     "does not explicitly mention recency."
    # )

    def __init__(self, model_registry, model_name, remember_turn=2, max_generate_token=256, max_repeat=3):
        loaded = model_registry(model_name)
        self.tokenizer = loaded.tokenizer
        self.model = loaded.model
        self.remember_turn = remember_turn
        self.max_generate_token = max_generate_token
        self.search_tools = WebSearchTool()
        self.max_repeat = max_repeat
    
    def build_system_prompt(self, language='Korean'):
        tool_str = functions.dumps_json({'name': self.search_tools.name, 'description': self.search_tools.description, 'args_schema': self.search_tools.args_schema})
        return f"""
You are a search-decision agent.

You NEVER generate answers to the user's request.
You ONLY decide whether to call a tool or "finish".

You can use the following tool:
{tool_str}

Rule:
- when action is "finish", write the synthesized fact only in "result". Do not explain, summarize, or interpret.
- Output EXACTLY one valid JSON object and nothing else.

Valid output format:
{{
  "action": "search" | "finish",
  "action_input": {{}},  // parameters go here
  "thought": "Reason for the action",
  "result": string
}}

Language rules:
- "thought" is answered in English.
- Never use any language other than {language} and English.

""".strip()

    def generate(self, user_input, history, language='Korean'):
        history = history[max(0, len(history)-2*self.remember_turn):] + [{'role': 'user', 'content': user_input}]   # 사용할 history만 뽑기
        sys_prompt = {'role': 'system', 'content': self.build_system_prompt(language)}
        if history[0]['role']=='system':
            history = history[1:]
        messages = [sys_prompt] + history

        observation = None
        result = []
        for _ in range(self.max_repeat):
            if observation is not None:
                messages.append({'role':'tool', 'content':observation})

            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # print(f"\n---------------------------- [INPUT] ----------------------------\n{input_text}\n")

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
                result.append(output_dict["result"])
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



class TextAgent(Agent):
    name = "Text Agent"
    description = "Handles language-level tasks such as summarization, translation, and rewriting."

    def __init__(self, model_registry, model_name, remember_turn=2, max_generate_token=1024, max_repeat=3):
        loaded = model_registry(model_name)       
        self.tokenizer = loaded.tokenizer
        self.model = loaded.model
        self.remember_turn = remember_turn
        self.max_generate_token = max_generate_token
        self.max_repeat = max_repeat

    def build_system_prompt(self, language='Korean'):
        return f"""
You are the Text Agent.
Output EXACTLY one JSON object.

Task:
- "note": brief explanation.
- "result": Respond only to requests without explanation.

Language Rule:
- Use only English in "note"
- If the user explicitly specifies an output language, "result" uses that language.
- Otherwise, respond in {language} to "result".

Valid output format:
{{
  "note": string,
  "result": string
}}
""".strip()

    def generate(self, user_input, history, language='Korean'):
        history = history[max(0, len(history)-2*self.remember_turn):] + [{'role': 'user', 'content': user_input}]   # 사용할 history만 뽑기
        sys_prompt = {'role': 'system', 'content': self.build_system_prompt(language)}
        if history[0]['role']=='system':
            history = history[1:]
        messages = [sys_prompt] + history
        
        for i in range(self.max_repeat):
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(input_text, return_tensors="pt")
            outputs = self.model.generate(
                **inputs.to(DEVICE),
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_generate_token
            )

            generated_output = outputs[0][len(inputs.input_ids[0]):].tolist()
            output_text = self.tokenizer.decode(generated_output, skip_special_tokens=True)

            try:
                output_dict = functions.loads_json(output_text)
                break
            except Exception:
                messages += [{"role":"assistant", "content": output_text}, {"role":"user", "content": "Not JSON. Respond again with ONLY one JSON object."}]

        return output_dict["result"]

        
class EmailAgent(Agent):
    name = "Email Agent"
    description = "Handle email-related tasks such as reading and writing emails."

    def __init__(self, model_registry, model_name, tool_registry:dict, remember_turn=2, max_generate_token=512, max_repeat=6):
        loaded = model_registry(model_name)       
        self.tokenizer = loaded.tokenizer
        self.model = loaded.model
        self.remember_turn = remember_turn
        self.max_generate_token = max_generate_token
        self.tool_registry = tool_registry
        self.max_repeat = max_repeat

    def build_system_prompt(self, language='Korean'):
        tool_str = "\n".join([f"- {name}:\n{functions.dumps_json({'description': tool.description, 'args_schema': tool.args_schema})}\n" for name, tool in self.tool_registry.items()])
        return f"""
You are an Email Decision Agent.
You ONLY decide whether to call one email tool or finish.

Available Tools:
{tool_str}

Decision rules:
1. Reading/listing: if message_id is unknown, call search_emails first.
2. Getting body: use get_emails.
3. Sending email:
   - Only call a send tool when ALL required fields (to, subject, body_text) are explicitly provided.
   - Never guess missing email addresses, subjects, or message content.
   - If any required field is missing, choose action="finish" and explain what is missing.
4. tool_name MUST be exactly one of the tool names listed in Available Tools.
5. tool_args MUST strictly follow the selected tool's args_schema.

Language rules:
- "thought" is answered in English.
- Never use any language other than {language} and English.

Output MUST be exactly one JSON and nothing else:
{{
  "action": "tool_call" | "finish",
  "tool_name": string | null,
  "tool_args": object | null,
  "finish_reason": "done" | "need_user_input" | null,
  "missing_fields": string[],
  "thought": string
}}
""".strip()

    def generate(self, user_input, history, language='Korean'):
        history = history[max(0, len(history)-2*self.remember_turn):] + [{'role': 'user', 'content': user_input}]   # 사용할 history만 뽑기
        sys_prompt = {'role': 'system', 'content': self.build_system_prompt(language)}
        if history[0]['role']=='system':
            history = history[1:]
        messages = [sys_prompt] + history

        observation = None
        result = []
        for step in range(self.max_repeat):
            print(f"\nSTEP {step+1}: ")
            if observation is not None:
                messages.append({'role':'tool', 'content':observation})

            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # print(f"\n---------------------------- [INPUT] ----------------------------\n{input_text}\n")

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
            
            tool_name = output_dict.get("tool_name")
            tool_args: dict[str, str] = output_dict.get("tool_args")

            # 도구 사용 성공했을 때 만 result에 저장.
            try:
                observation = self.tool_registry[tool_name](**tool_args)
                print("---------- OBSERVATION ---------- \n", observation, "\n")
            except Exception as e:
                observation = functions.dumps_json(
                    {
                        "ok": False,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }
                )
                # messages+=[{'role': 'user', 'content': f"action={action} \naction_input={action_input} \n[tool call error] {e}\n\n Please answer again."}]
                continue

            result.append(copy.deepcopy(observation))
            if tool_name=="get_emails":
                for idx, curr_email in enumerate(observation):
                    curr_email.pop("body")
                    curr_email["body_fetched"] = True

                    observation[idx]=curr_email

            observation = functions.dumps_json({"ok": True, "results": observation})

        if not result:
            return None

        return result[-1]