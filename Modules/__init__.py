from typing  import Optional
from time import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import functions
from . import Agents

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        return f"""
You are the Router.
Do NOT answer the user. Output JSON only.

Available Agents:
{agent_descript_str}

route decision:
- route MUST be exactly one of: "single_agent", "planner", "clarification".
- If required information for an action is missing or the request has unresolved references,
  route="clarification" and ask in {language}.
- Otherwise, determine using_agents first:
  - If using_agents contains 2+ agents, route="planner".
  - Else if only one agent is in using_agents, route="single_agent".

Rules:
- If the user requests an EMAIL OPERATION (read/search/list/send/reply/forward) on emails,
  using_agents MUST include "Email Agent".
- If the user requests text generation/transformation (summarize/translate/rewrite/format) of content,
  using_agents MUST include "Answer Agent".
- If the request requires changeable or time-sensitive information,
  using_agents MUST include "Search Agent".

preserve_spans:
- ALWAYS include user-provided proper nouns and literal identifiers verbatim
  (e.g., person/org/service/app names, emails, IDs, dates, file names).
- ALSO include any user-provided string explicitly used to select, filter, or target items.
- Do not translate, modify, or normalize preserved strings.
- preserve_spans MUST follow the order of appearance in the user input.
- If the same string appears multiple times, include it only once at its first occurrence.
- preserve_spans is referenced as {{P0}},{{P1}},... in high_level_intent by index.

Write high_level_intent in English.

high_level_intent rules:
- high_level_intent MUST be abstract and MUST NOT contain any substring from preserve_spans (use {{Pn}} placeholders instead).
- If preserve_spans is non-empty, high_level_intent MUST reference preserved values ONLY via indexed placeholders {{P0}}, {{P1}}, ...
- Each placeholder {{Pn}} refers to preserve_spans[n] by index.
  (e.g., "Search for {{P0}} and send the results to {{P1}} via email.").

Output JSON only:
{{
  "route": "single_agent" | "planner" | "clarification",
  "clarifying_question": "",
  "using_agents": [],
  "preserve_spans": [],
  "high_level_intent": ""
}}
""".strip()

    def generate(self, user_input:str, history:list, language="Korean", max_repeat=3)->dict:
        messages = history[max(0, len(history) - self.remember_turn*2):]+[{'role': "user", "content": user_input.strip()}]
        if messages[0]['role']=="system":
            messages = messages[1:]
        messages = [{'role': "system", 'content': self.build_system_prompt(language)}] + messages
        for _ in range(max_repeat):
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # print(f"\n---------------------------- [INPUT] ----------------------------\n{input_text}\n")

            inputs = self.tokenizer(
            input_text,
            return_tensors='pt'
            )

            output = self.model.generate(
                **inputs.to(DEVICE),
                eos_token_id = self.tokenizer.eos_token_id,
                pad_token_id = self.tokenizer.eos_token_id,
                max_new_tokens = self.max_generate_token,
                temperature=0.1
            )
            
            generated_output = output[0][len(inputs.input_ids[0]):].tolist()
            output_text = self.tokenizer.decode(generated_output, skip_special_tokens=True)
            messages += [
                    {
                        "role": "assistant",
                        "content": output_text,
                    }
                ]
            # print("\n---------------------------- [OUTPUT] ----------------------------\n")
            # print(output_text)
            # print(f"{len(generated_output)=}\n")
            try:
                output_dict = functions.loads_json(output_text)
                break
            except Exception:
                messages += [
                    {
                        "role": "user",
                        "content": f"Not JSON. Respond again with ONLY one JSON object.",
                    }
                ]

        return output_dict

class Planner:
    def __init__(self, model_registry, model_name, available_agents:Optional[dict]=None, remember_turn:int=2, max_generate_token:int=512):
        loaded = model_registry(model_name)
        self.tokenizer = loaded.tokenizer
        self.model = loaded.model
        self.remember_turn = remember_turn
        self.max_generate_token = max_generate_token
        self.available_agents = available_agents

    def build_system_prompt(self, language='Korean'):
        agent_descript_str = "\n".join([f"- {name}: {agent.description}" for name, agent in self.available_agents.items()])
        return f"""
You are the Planner.
Your role is to create a global execution plan based on the given tasks.
Output JSON only.

Available Agents:
{agent_descript_str}

Input:
- User input
- Router output: using_agents, high_level_intent, preserve_spans.

Rules (planning):
- Each task MUST be exactly ONE agent-step, where one agent performs exactly one purpose, and be assigned to exactly ONE agent.
- Any task that requires specific input data may only proceed if that data is explicitly available in the current context (e.g., user input, prior task outputs, or provided history). If the required data is not available, you MUST add a prior task to obtain it.
- If a task depends on a previous task output, include that task_id in depends_on.

Rules (agent responsibility):
- Email Agent be used for any email-related actions, including searching, reading, and sending emails.
- Answer Agent be used ONLY for language-level tasks such as general answer, summarize, translate, rewrite, or format.
- Search Agent be used when changeable or time-sensitive information is required.

Rules (entities):
- Router output may contain placeholders {{P0}}, {{P1}}, ... in high_level_intent.
- You MAY replace these placeholders using the corresponding values from preserve_spans.
- You MUST NOT introduce new concrete entities beyond those present in the user input or preserve_spans.

Rules (language):
- Write objective and acceptance_criteria in English.

Output JSON only:
{{
  "tasks": [
    {{
      "task_id": "t1",
      "agent": "",
      "objective": "",
      "depends_on": [],
      "acceptance_criteria": ""
    }}
  ]
}}
""".strip()
# - {{PRESERVE_i}} is an explicit exception and must be replaced with preserve_spans[i] as-is.

    def generate(self, user_input:str, router_output:dict, history, language='Korean',  max_repeat=3)->dict:
        # using_router_output = {"using_agents": router_output["using_agents"], "high_level_intent": router_output["high_level_intent"], "preserve_spans": router_output["preserve_spans"]}
        using_router_output = {key: val for key, val in router_output.items() if key.strip() not in ["route", "clarifying_question"]}
        total_input = f"[User Input]:\n{user_input.strip()}\n\n[Router Output]:\n{functions.dumps_json(using_router_output)}"
        messages = history[max(0, len(history) - self.remember_turn*2):]+[{'role': "user", "content": total_input}]
        if messages[0]['role']=="system":
            messages = messages[1:]
        messages = [{'role': "system", 'content': self.build_system_prompt(language)}] + messages

        for _ in range(max_repeat):
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # print(f"\n---------------------------- [INPUT] ----------------------------\n{input_text}\n")

            inputs = self.tokenizer(input_text, return_tensors='pt')
            outputs = self.model.generate(
                **inputs.to(DEVICE),
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_generate_token
            )

            generated_output = outputs[0][len(inputs.input_ids[0]):].tolist()
            output_text = self.tokenizer.decode(generated_output, skip_special_tokens=True)
            messages += [
                        {
                            "role": "assistant",
                            "content": output_text,
                        }
                    ]
            # print("\n---------------------------- [OUTPUT] ----------------------------\n")
            # print(output_text)
            # print(f"{len(generated_output)=}\n")
            try:
                output_dict = functions.loads_json(output_text)
                break
            except Exception:
                messages += [
                    {
                        "role": "user",
                        "content": f"Not JSON. Respond again with ONLY one JSON object.",
                    }
                ]

        return output_dict


class Orchestrator:
    def __init__(self, router:Router, planner:Planner, agents:list[Agents.Agent]):
        self.agents = agents
        self.router = router
        self.planner = planner
        self.history = []
        
    def run_router(self, user_input:str, history:list, language="Korean"):
        route = self.router.generate(user_input=user_input, history=history, language=language)
        return route
    
    def run_planner(self, user_input:str, router_output:dict, history:list, language='Korean'):
        plan = self.planner.generate(user_input=user_input, router_output=router_output, history=history, language=language)
        return plan
        
    def get_dependent_result(self, depends_on, task_results) -> dict:
        result = {}
        for task_id in depends_on:
            task_result = task_results[task_id]
            result[task_id] = task_result
        
        return result


    def run_agent(self, agent_name, input_string, language="Korean"):
        agent:Agents.Agent = self.agents[agent_name]
        tool_result = agent.generate(input_string, self.history, language)
        
        return tool_result

    def run(self, user_input):
        language = functions.detect_language(user_input)
        
        torch.cuda.empty_cache()
        self.router.model.eval()
        self.planner.model.eval()

        with torch.no_grad():
            route = self.run_router(user_input, self.history, language)
            print(f"Route:\n{functions.dumps_json(route)}\n")

            plan = self.run_planner(user_input, route, self.history, language)
            print(f"Plan:\n{functions.dumps_json(plan)}\n")
            
            tasks = plan["tasks"]
            task_results = {}
            for i, task in enumerate(tasks):
                # task_id = task["task_id"]
                # agent_name = task["agent"]
                # objective = task["objective"]
                # depends_on = task["depends_on"]
                # acceptance_criteria = task["acceptance_criteria"]
                input_string = ""

                dependent_results = self.get_dependent_result(task["depends_on"], task_results)
                if dependent_results:
                    dependent_results_string = functions.dumps_json(dependent_results)
                    input_string+=f"[Context]:\n{dependent_results_string}\n\n"

                task_string = functions.dumps_json({key:val for key, val in task.items() if key in ["objective", "depends_on", "acceptance_criteria"]})
                input_string += f"[TASK]:\n{task_string}"
                
                print(f"\n------------------------------------ TASK {i+1} ------------------------------------\n{task['agent']}")
                start_time = time()
                agent_tool_result = self.run_agent(task["agent"], input_string.strip(), language)
                end_time = time()
                print(f"{agent_tool_result=}\n")
                print(f"Durations: {(end_time-start_time):.2f}")

                task_results[task["task_id"]] = agent_tool_result
                

