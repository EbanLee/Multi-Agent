from typing  import Optional
from time import time
import re

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
    def __init__(self, model_registry:ModelRegistry, model_name, available_agents:Optional[dict[str, Agents.Agent]]=None, remember_turn:int=1, max_generate_token:int=128):
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
- Email operations (read/search/send) -> include "Email Agent"
- Time-sensitive/changeable info (current/latest, people/org roles, rankings) -> include "Search Agent"
- Text tasks (summarize/rewrite/format) -> include "Text Agent"

Step 2) preserve_spans:
- Extract each proper nouns or identifiers (e.g., people name, organization/service name, URL, ID, file name) from user input.
- Copy each string exactly as-is.
- Add once, in first-appearance order.

Step 3) route ∈ {"direct","planner","clarification"}:
- If required information is missing or a reference is ambiguous:
  -> route="clarification" and write clarifying_question in {language}.
- Else if using_agents contains "Email Agent" or "Search Agent":
  -> route="planner".
- Else:
  -> route="direct".

Step 4) high_level_intent:
- Write in English.
- If preserve_spans is non-empty:
  - Placeholders are written as {{P0}}, {{P1}}, ... in the same order as preserve_spans.
  - high_level_intent MUST contain each placeholder.
  - Except for placeholders, preserve_spans MUST NOT appear in any other forms (translations, romanizations, or paraphrases).
  - Except for placeholders, the entire high_level_intent MUST be in English.
- NEVER mention the agent.

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

        # Place holder code로 변환
        for i, preserve_span in sorted(enumerate(output_dict["preserve_spans"]), key=lambda x: -len(x[1])):
            pattern = re.escape(preserve_span)
            output_dict["high_level_intent"] = re.sub(pattern, f"{{P{i}}}", output_dict["high_level_intent"], flags=re.IGNORECASE)

        return output_dict

class Planner:
    def __init__(self, model_registry:ModelRegistry, model_name, available_agents:Optional[dict[str, Agents.Agent]]=None, remember_turn:int=2, max_generate_token:int=512):
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
Create an execution plan in execution order. Output JSON only.

Available Agents:
{agent_descript_str}

Input:
- User input
- Router output: using_agents, high_level_intent, preserve_spans.

Core rules:
- Each task MUST have one atomic objective: the objective and acceptance_criteria MUST reference EXACTLY ONE entity; any attribute, if present, MUST reference that entity only.
- If required data is not available, create a prior task to obtain it.
- If a task uses the output of prior tasks, list that task_id in depends_on.
- Requested user-facing tasks (show/display) MUST be scheduled last unless the user explicitly specifies order (e.g., "first/before/after").

Agent selection:
- Text Agent: language-only tasks (summarize/translate/format) and user-facing tasks (show/display/render).
- Email Agent: email-related action (search, read, send).
- Search Agent: requires time-sensitive or changeable information.

Entity & placeholder constraints:
- Use preserve_spans only as-is (not {{P0}}, {{P1}} ...) in objective and acceptance_criteria.
- Do NOT introduce concrete entities not explicitly mentioned in the user input or preserve_spans.

Output constraints:
- Except for items in preserve_spans, objective and acceptance_criteria MUST be in English.

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
    def __init__(self, model_registry:ModelRegistry, model_name, remember_turn=2, max_generate_token=1024):
        loaded = model_registry(model_name)       
        self.tokenizer = loaded.tokenizer
        self.model = loaded.model
        self.remember_turn = remember_turn
        self.max_generate_token = max_generate_token

    def build_system_prompt(self, language='Korean'):
        return f"""
You are the Final Answer Generator.

Rules:
- If the user requests text (e.g., answer, information, explanation), return the requested response.
- If the user request involves performing external actions or modifying system state (e.g., sending, saving, updating):
  - Default behavior: return only the status of the actions performed.
  - Do NOT return generated content.
  - EXCEPTION: If the user explicitly requests the content to be shown/displayed/presented, return that content.
- If you return an action status and one or more requested contents, separate them using clear headings.
- Never mention internal names or identifiers.

Language policy:
- If the user explicitly specifies an output language, use that language.
- Otherwise, All responses must be in {language}.
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
    def __init__(self, router:Router, planner:Planner, agents:dict[str, Agents.Agent], finalizer:FinalAnswerGenerator):
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
                input_string+=f"[DEPENDENCIES]:\n{dependent_results_string}\n\n"
            task_string = functions.dumps_json({key:val for key, val in task.items() if key in ["objective", "depends_on", "acceptance_criteria"]})
            input_string += f"[TASK]:\n{task_string}"
            
            print(f"\n------------------------------------ TASK {i+1} ------------------------------------\n{task['agent']}")
            start_time = time()
            agent_result = self.run_agent(task["agent"], input_string.strip(), history, language)
            end_time = time()
            print(f"{agent_result=}\n")
            print(f"Task Durations: {(end_time-start_time):.2f}\n")

            task_results[task["task_id"]] = agent_result

            # 실패하면 종료
            if isinstance(agent_result, dict):
                agent_success = agent_result.get("ok", True)
                if not agent_success:
                    break
        
        return task_results


    def run(self, user_input, history):
        language = functions.detect_language(user_input)
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            start_time = time()
            router_output = self.run_router(f"[User Input]:\n{user_input}", history, language)
            end_time = time()
            print(f"Route:\n{functions.dumps_json(router_output)}\n")
            print(f"Routing Durations: {(end_time-start_time):.2f}\n\n")

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
                while plan["tasks"][-1]["agent"].strip()=="Text Agent":
                    plan["tasks"].pop()

                start_time = time()
                task_results = self.execute_plans(plan, history, language)
                end_time = time()
                
                print(f"Task Result:\n{functions.dumps_json(task_results)}\n")
                print(f"Execute Total Plan Durations: {(end_time-start_time):.2f}\n\n")

                plan["tasks"] = [{key:val for key, val in t.items() if key.strip()!="acceptance_criteria"} for t in plan["tasks"]]
                total_input = f"[User Input]:\n{user_input}\n\n[Execution Plan]:\n{functions.dumps_json(plan['tasks'])}\n\n[Execution Result]:\n{functions.dumps_json(task_results)}".strip()
                print(f"!!!!!!!!!!!!!!!!!!! Final Input !!!!!!!!!!!!!!!!!!!\n{total_input}\n")
                final_answer = self.finalizer.generate(user_input=total_input, history=history)
                
                return final_answer
