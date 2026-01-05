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
    def __init__(self, model_registry, model_name, available_agents:Optional[dict]=None, remember_turn:int=1, max_generate_token:int=128):
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

Step 1) Select using_agents:
- Email operations (read/search/list/send/reply/forward) -> include "Email Agent"
- Time-sensitive/changeable info (current/latest/recent, people/org roles, prices, rankings, events) -> include "Search Agent"
- Text transformation (summarize/translate/rewrite/format/draft) -> include "Text Agent"

Step 2) preserve_spans:
- Preserve user-provided proper nouns or identifiers verbatim
  (person/org/service/app names, emails, URLs, IDs, explicit dates, file names).
- Add once, in first-appearance order.
- Copy exactly as-is.
- Refer only via {{P0}}, {{P1}}, ... in high_level_intent.

Step 3) route ∈ {"direct","planner","clarification"}:
- If a requested ACTION lacks required fields
  (e.g., email send/reply/forward: to/subject/body or reply target)
  -> route="clarification" and write clarifying_question in {language}.
- Else if using_agents contains "Email Agent" or "Search Agent"
  -> route="planner".
- Else
  -> route="direct".

high_level_intent:
- Write in English.
- If the same attribute is requested for multiple entities, MUST use "each" to indicate independent results.
- MUST NOT contain any substring from preserve_spans (use {{Pn}} placeholders instead).
- If preserve_spans is non-empty, high_level_intent MUST reference preserved values ONLY via indexed placeholders {{P0}}, {{P1}}, ...

Output JSON only:
{{
  "route": "direct" | "planner" | "clarification",
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
                temperature=0.1,
                do_sample=False,
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
Create an execution plan as a JSON object only.

Available Agents:
{agent_descript_str}

Input:
- User input
- Router output: using_agents, high_level_intent, preserve_spans.

Rules:
- ONE task = ONE agent = ONE action (ONE verb) on ONE target.
- If a task requires more than one action or more than one target, it MUST be split into separate tasks; do NOT use "and" to combine targets in a single objective.
- If a task needs data not present in the current context (e.g., user input, prior task outputs, or provided history) yet, MUST add a prior task to obtain that data.
- If a task depends on a previous task output, include that task_id in depends_on.

Agent selection:
- Email Agent: any email-related action (search, read, send).
- Search Agent: time-sensitive/changeable info is required (e.g., People, Organizations, price, ranking, Events).
- Text Agent: ONLY language-level tasks (answer, summarize, translate, rewrite, format).

Constraints:
- You may replace placeholders {{P0}},{{P1}},... in high_level_intent using preserve_spans values.
- When multiple entities are present, create separate tasks per entity unless a joint relationship is explicitly requested.
- You MUST NOT introduce new concrete entities beyond those present in the user input or preserve_spans.

Output rules:
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
                max_new_tokens=self.max_generate_token,
                do_sample=False,
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


class FinalAnswerGenerator():
    def __init__(self, model_registry, model_name, remember_turn=2, max_generate_token=1024):
        loaded = model_registry(model_name)       
        self.tokenizer = loaded.tokenizer
        self.model = loaded.model
        self.remember_turn = remember_turn
        self.max_generate_token = max_generate_token

    def build_system_prompt(self, language='Korean'):
        return f"""
You are the Final Answer Generator.
Produce the final user-facing response.

Rules:
- Use the provided context to answer the user's request.
- If execution results are provided, incorporate them accurately.
- If no execution results are provided, answer directly from the user input.
- Do not mention internal steps, agents, or tools.
- Follow the language policy.

Language policy:
- If the user explicitly specifies an output language, use that language.
- Otherwise, respond in {language}.
""".strip()

    def generate(self, user_input, history, language='Korean'):
        history = history[max(0, len(history)-2*self.remember_turn):] + [{'role': 'user', 'content': user_input}]   # 사용할 history만 뽑기
        sys_prompt = {'role': 'system', 'content': self.build_system_prompt(language)}
        if history[0]['role']=='system':
            history = history[1:]
        messages = [sys_prompt] + history
        
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

        return output_text


class Orchestrator:
    def __init__(self, router:Router, planner:Planner, agents:list[Agents.Agent], finalizer:FinalAnswerGenerator):
        self.agents = agents
        self.router = router
        self.planner = planner
        self.finalizer = finalizer

        self.router.model.eval()
        self.planner.model.eval()
        
    def run_router(self, user_input:str, history:list, language="Korean"):
        route = self.router.generate(user_input=user_input, history=history, language=language)
        return route
    
    def run_planner(self, user_input:str, router_output:dict, history:list, language='Korean'):
        plan = self.planner.generate(user_input=user_input, router_output=router_output, history=history, language=language)
        return plan
        
    def get_dependent_results(self, depends_on, task_results) -> dict:
        result = {}
        for task_id in depends_on:
            task_result = task_results[task_id]
            result[task_id] = task_result
        
        return result


    def run_agent(self, agent_name, input_string, history, language="Korean"):
        agent:Agents.Agent = self.agents[agent_name]
        tool_result = agent.generate(input_string, history, language)
        
        return tool_result

    def execute_plans(self, plan, history, language)->dict:
        """
        Plan의 각 Task에 해당하는 결과 반환 - {task_name: result}형태
        """
        tasks = plan["tasks"]
        task_results = {}
        for i, task in enumerate(tasks):
            # task_id = task["task_id"]
            # agent_name = task["agent"]
            # objective = task["objective"]
            # depends_on = task["depends_on"]
            # acceptance_criteria = task["acceptance_criteria"]
            input_string = ""

            dependent_results = self.get_dependent_results(task["depends_on"], task_results)
            if dependent_results:
                dependent_results_string = functions.dumps_json(dependent_results)
                input_string+=f"[Context]:\n{dependent_results_string}\n\n"
            task_string = functions.dumps_json({key:val for key, val in task.items() if key in ["objective", "depends_on", "acceptance_criteria"]})
            input_string += f"[TASK]:\n{task_string}"
            
            print(f"\n------------------------------------ TASK {i+1} ------------------------------------\n{task['agent']}")
            start_time = time()
            agent_tool_result = self.run_agent(task["agent"], input_string.strip(), history, language)
            end_time = time()
            print(f"{agent_tool_result=}\n")
            print(f"Task Durations: {(end_time-start_time):.2f}\n")

            task_results[task["task_id"]] = agent_tool_result
        
        return task_results


    def run(self, user_input, history):
        language = functions.detect_language(user_input)
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            start_time = time()
            router_output = self.run_router(user_input, history, language)
            end_time = time()
            print(f"Route:\n{functions.dumps_json(router_output)}\n")
            print(f"Durations: {(end_time-start_time):.2f}\n\n")

            if router_output["route"]=="clarification":
                return router_output["clarifying_question"]
            
            elif router_output["route"]=="direct":
                return self.finalizer.generate(user_input=user_input, history=history)

            elif router_output["route"]=="planner":
                start_time = time()
                plan = self.run_planner(user_input, router_output, history, language)
                end_time = time()
                print(f"Plan:\n{functions.dumps_json(plan)}\n")
                print(f"Planning Durations: {(end_time-start_time):.2f}\n\n")
                
                # 마지막 작업이 Text작업이면 하지않고 Finalizer로 보내기
                if plan["tasks"][-1]["agent"].strip()=="Text Agent":
                    plan["tasks"].pop()

                start_time = time()
                task_results = self.execute_plans(plan, history, language)
                end_time = time()
                
                print(f"Task Result:\n{task_results}\n")
                print(f"Execute Total Plan Durations: {(end_time-start_time):.2f}\n\n")

                total_input = f"[User Input]:\n{user_input}\n\n[Plan]:\n{functions.dumps_json(plan)}\n\n[Execution Result]:\n{functions.dumps_json(task_results)}".strip()
                final_answer = self.finalizer.generate(user_input=total_input, history=history)
                
                return final_answer
