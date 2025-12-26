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
    print("\nPlanner Test!\n")

    agents = [
        Agents.DummyAgent("model_name", name=Agents.SearchAgent.name, description=Agents.SearchAgent.description), 
        Agents.DummyAgent("model_name", name=Agents.EmailAgent.name, description=Agents.EmailAgent.description), 
        Agents.DummyAgent("model_name", name=Agents.FinalAnswerAgent.name, description=Agents.FinalAnswerAgent.description)
        ]
    agents = {agent.name:agent for agent in agents}

    torch.cuda.empty_cache()
    planner = Modules.Planner(model_registry, config["planner_model_name"], available_agents=agents)

    user_input = "김준호에게서 온 이메일 내용을 읽고 거기에 있는 축구팀의 어제 경기 결과에 대해 답장해줘."
    dummy_router_output = f"""
{{
  "route": "planner",
  "using_agents": ["Email Agent", "Search Agent"],
  "high_level_intent": "Read the email from Kim Joonho, find the game result as requested in the email, summarize it, and send it back to Kim Joonho.",
  "clarifying_question": ""
}}
""".strip()
    dummy_router_output = functions.loads_json(dummy_router_output)
    print(functions.dumps_json(dummy_router_output))

    history = []
    with torch.no_grad():
        planner_output = planner.generate(user_input, dummy_router_output, history)

    torch.cuda.empty_cache()

"""
ex)
{
  "tasks": [
    {
      "task_id": "t1",
      "agent": "Email Agent",
      "objective": "Read the email from Kim Joonho",
      "depends_on": [],
      "acceptance": "Successfully read the email from Kim Joonho"
    },
    {
      "task_id": "t2",
      "agent": "Search Agent",
      "objective": "Find the game result as requested in the email",
      "depends_on": ["t1"],
      "acceptance": "Found the game result as requested in the email"
    },
    {
      "task_id": "t3",
      "agent": "Email Agent",
      "objective": "Summarize the game result and send it back to Kim Joonho",
      "depends_on": ["t2"],
      "acceptance": "Summarized the game result and sent it back to Kim Joonho"
    }
  ]
}
"""


if __name__=="__main__":
    model_registry = Modules.ModelRegistry()
    # router_test(model_registry)
    # planner_test(model_registry)
    search_agent_test(model_registry)
