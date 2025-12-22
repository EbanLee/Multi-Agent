
import torch

from utils import file_utils, functions
import Agents

config = file_utils.read_yaml("config.yaml")
print(config)

search_agent = Agents.SearchAgent(model_name=config['search_agent_model_name'])



