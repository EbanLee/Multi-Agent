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
    agents = [
        Agents.DummyAgent("model_name", name=Agents.SearchAgent.name, description=Agents.SearchAgent.description), 
        Agents.DummyAgent("model_name", name=Agents.FinalAnswerAgent.name, description=Agents.FinalAnswerAgent.description)
        ]
    agents = {agent.name:agent for agent in agents}

    torch.cuda.empty_cache()
    router = Modules.Router(model_registry, config['router_model_name'], available_agents=agents)

    user_input = "비트코인과 이더리움의 이슈를 알려줘"
    language = functions.detect_language(user_input)
    user_input = f"[Current Time: {datetime.now().strftime('%Y-%m-%d')}]\n{user_input}"

    history = []    # {'role': "user", "content": "손흥민의 팀은?"}, {'role': "assistant", "content": "손흥민의 팀은 LAFC입니다."}
    router.model.eval()
    with torch.no_grad():
        router.generate(user_input=user_input, history=history, language=language)

    torch.cuda.empty_cache()

def search_agent_test(model_registry):
    torch.cuda.empty_cache()
    search_agent = Agents.SearchAgent(model_registry, config["search_agent_model_name"])

    user_input = "손흥민과 김민재의 팀"
    language = functions.detect_language(user_input)
    user_input = f"[Current Time: {datetime.now().strftime('%Y-%m-%d')}]\n{user_input}"

    history = []
    search_agent.model.eval()
    with torch.no_grad():
        search_agent.generate(user_input, history, language=language)

    torch.cuda.empty_cache()


if __name__=="__main__":
    model_registry = Modules.ModelRegistry()
    # router_test(model_registry)
    search_agent_test(model_registry)
