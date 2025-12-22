import torch

from tools import *
from utils import functions, file_utils
import Agents
import Modules

# search_t = search_tool.WebSearchTool()
# print(search_t("손흥민"))
config = file_utils.read_yaml("config.yaml")
# search_agent = SearchAgent(config['search_agent_model_name'])
# final_answer_agent = FinalAnswerAgent(config['final_answer_model_name'])
agents = [
    Agents.DummyAgent("model_name", name=Agents.SearchAgent.name, description=Agents.SearchAgent.description), 
    Agents.DummyAgent("model_name", name=Agents.FinalAnswerAgent.name, description=Agents.FinalAnswerAgent.description)
    ]
torch.cuda.empty_cache()
router = Modules.Router(config['router_model_name'], available_agents=agents)
user_input = "다익스트라 알고리즘이란?"
language = functions.detect_language(user_input)
history = []    # {'role': "user", "content": "손흥민의 팀은?"}, {'role': "assistant", "content": "손흥민의 팀은 LAFC입니다."}


router.model.eval()
with torch.no_grad():
    router.generate(user_input=user_input, history=history, language=language)

torch.cuda.empty_cache()
