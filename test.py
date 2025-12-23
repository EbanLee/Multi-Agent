from datetime import datetime

import torch

from tools import *
from utils import functions, file_utils
from Modules import Agents
import Modules

# search_t = search_tool.WebSearchTool()
# print(search_t("손흥민"))
config = file_utils.read_yaml("config.yaml")
# search_agent = SearchAgent(config['search_agent_model_name'])
# final_answer_agent = FinalAnswerAgent(config['final_answer_model_name'])
def router_test(model_registry):
    print("\nRouter Test!\n")
    agents = [
        Agents.DummyAgent("model_name", name=Agents.SearchAgent.name, description=Agents.SearchAgent.description), 
        Agents.DummyAgent("model_name", name=Agents.EmailAgent.name, description=Agents.EmailAgent.description), 
        Agents.DummyAgent("model_name", name=Agents.FinalAnswerAgent.name, description=Agents.FinalAnswerAgent.description)
        ]
    agents = {agent.name:agent for agent in agents}

    torch.cuda.empty_cache()
    router = Modules.Router(model_registry, config['router_model_name'], available_agents=agents)

    while True:
        user_input = input("입력: ")
        language = functions.detect_language(user_input)
        user_input = f"[Current Time: {datetime.now().strftime('%Y-%m-%d')}]\n{user_input}"

        history = []    # {'role': "user", "content": "손흥민의 팀은?"}, {'role': "assistant", "content": "손흥민의 팀은 LAFC입니다."}
        router.model.eval()
        with torch.no_grad():
            router_output = router.generate(user_input=user_input, history=history, language=language)

        torch.cuda.empty_cache()

def search_agent_test(model_registry):
    print("\nSearch Agent Test!\n")
    torch.cuda.empty_cache()
    search_agent = Agents.SearchAgent(model_registry, config["search_agent_model_name"])

    while True:
        user_input = input("입력: ")
        language = functions.detect_language(user_input)
        user_input = f"[Current Time: {datetime.now().strftime('%Y-%m-%d')}]\n{user_input}"

        history = []
        search_agent.model.eval()
        with torch.no_grad():
            search_agent.generate(user_input, history, language=language)

        torch.cuda.empty_cache()


def planner_test(model_registry):
    print("\nRouter Test!\n")
    user_input = "김민혁에게서 온 이메일 내용을 읽고 그안에 있는 선수 이름을 검색해서 정보를 알려줘"
    dummy_router_output = f"""
{{
  "direct_answer_possible": false,
  "needs_clarification": false,
  "clarifying_question": "",
  "tasks": [
    {{
      "task_id": "t1",
      "objective": "Read the email from Kim Minhyuk and extract the player name.",
      "agent": "Email Agent"
    }},
    {{
      "task_id": "t2",
      "objective": "Search for the player information based on the extracted name.",
      "agent": "Search Agent"
    }}
  ]
}}
""".strip()
    dummy_router_output = functions.loads_json(dummy_router_output)
    print(functions.dumps_json(dummy_router_output))



if __name__=="__main__":
    model_registry = Modules.ModelRegistry()
    router_test(model_registry)
    # search_agent_test(model_registry)
    # planner_test(model_registry)
