import os
import torch
from time import time

from dotenv import load_dotenv

from utils import file_utils
import Modules
import Modules.Agents as Agents
from tools import email_tool

config = file_utils.read_yaml("config.yaml")
# print(config)

search_agent = Agents.SearchAgent(model_name=config['search_agent_model_name'])

def run_orchestrator():
    load_dotenv()
    
    EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
    APP_PASSWARD = os.getenv("APP_PASSWARD")
    SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
    IMAP_HOST = os.getenv("IMAP_HOST", "imap.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

    torch.cuda.empty_cache()

    model_registry = Modules.ModelRegistry()
    email_tool_registry = [
        email_tool.EmailSearchTool(email_addr=EMAIL_ADDRESS, app_password=APP_PASSWARD, imap_host=IMAP_HOST),
        email_tool.EmailGetTool(EMAIL_ADDRESS, APP_PASSWARD, IMAP_HOST),
        email_tool.EmailSendTool(email_addr=EMAIL_ADDRESS, app_password=APP_PASSWARD, smtp_host=SMTP_HOST, smtp_port=SMTP_PORT)
        ]
    email_tool_registry = {t.name:t for t in email_tool_registry}

    agents = [
        Agents.TextAgent(model_registry=model_registry, model_name=config["text_agent_model_name"]),
        Agents.SearchAgent(model_registry=model_registry, model_name=config["search_agent_model_name"]),
        Agents.EmailAgent(model_registry=model_registry, model_name=config["email_agent_model_name"], tool_registry=email_tool_registry),
        ]
    agents = {agent.name:agent for agent in agents}

    router = Modules.Router(model_registry=model_registry, model_name=config["router_model_name"], available_agents=agents)
    planner = Modules.Planner(model_registry=model_registry, model_name=config["planner_model_name"], available_agents=agents)
    finalizer = Modules.FinalAnswerGenerator(model_registry=model_registry, model_name=config["finalizer_model_name"])

    orchestrator = Modules.Orchestrator(router=router, planner=planner, agents=agents, finalizer=finalizer)
    
    history=[]

    while True:
        user_input = input("입력: ")
        if user_input.lower().strip()=="exit":
            break
        start_time = time()
        print("\n사용자 입력: ", user_input)
        result = orchestrator.run(user_input=user_input, history=history)
        end_time = time()
        print(f"\nOrchestrator Result:\n{result}\n")
        print(f"Total Durations: {(end_time-start_time):.2f}\n")
        print("\n--------------------------- !!LOOP END!! ---------------------------\n")
        history += [{"role":"user", "content":user_input}, {"role":"assistant", "content":result}]

if __name__=="__main__":
    run_orchestrator()
