import os

from datetime import datetime

import torch
from dotenv import load_dotenv

from tools import email_tool
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
        Agents.DummyAgent("model_name", name=Agents.AnswerAgent.name, description=Agents.AnswerAgent.description)
        ]
    agents = {agent.name:agent for agent in agents}

    torch.cuda.empty_cache()
    router = Modules.Router(model_registry, config['router_model_name'], available_agents=agents)

    while True:
        user_input = input("입력: ")
        language = functions.detect_language(user_input)
        # user_input = f"[Current Time: {datetime.now().strftime('%Y-%m-%d')}]\n[USER REQUEST]\n{user_input}"

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
        Agents.DummyAgent("model_name", name=Agents.AnswerAgent.name, description=Agents.AnswerAgent.description)
        ]
    agents = {agent.name:agent for agent in agents}

    torch.cuda.empty_cache()
    planner = Modules.Planner(model_registry, config["planner_model_name"], available_agents=agents)

    user_input = "무신사에서 온 메일 요약해서 김민혁에게 보내줘"
    dummy_router_output = f"""
{{
  "route": "planner",
  "using_agents": ["Email Agent", "Answer Agent"],
  "high_level_intent": "Summarize the email from Muji and send it to Kim Minhyuk.",
  "clarifying_question": ""
}}
""".strip()
    dummy_router_output = functions.loads_json(dummy_router_output)
    print(functions.dumps_json(dummy_router_output))

    history = []
    with torch.no_grad():
        planner_output = planner.generate(user_input, dummy_router_output, history)

    torch.cuda.empty_cache()


def email_tool_test():
    load_dotenv()
    
    EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
    APP_PASSWARD = os.getenv("APP_PASSWARD")
    SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
    IMAP_HOST = os.getenv("IMAP_HOST", "imap.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

    email_read_tool = email_tool.EmailSearchTool(EMAIL_ADDRESS, APP_PASSWARD, IMAP_HOST)
    email_result = email_read_tool(unseen_only=False, subject_contains="무신사", from_contains=None)
    for email in email_result:
      for key, val in email.items():
          print(f"{key} - {val}")
      print()

    email_send_tool = email_tool.EmailSendTool(EMAIL_ADDRESS, APP_PASSWARD, SMTP_HOST, SMTP_PORT)


def email_agent_test(model_registry):
    print("\nEmail Agent Test!\n")

    load_dotenv()
    
    EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
    APP_PASSWARD = os.getenv("APP_PASSWARD")
    SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
    IMAP_HOST = os.getenv("IMAP_HOST", "imap.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    
    tool_registry = [email_tool.EmailSearchTool(email_addr=EMAIL_ADDRESS, app_password=APP_PASSWARD, imap_host=IMAP_HOST), email_tool.EmailGetTool(EMAIL_ADDRESS, APP_PASSWARD, IMAP_HOST), email_tool.EmailSendTool(email_addr=EMAIL_ADDRESS, app_password=APP_PASSWARD, smtp_host=SMTP_HOST, smtp_port=SMTP_PORT)]
    tool_registry = {t.name:t for t in tool_registry}

    agent = Agents.EmailAgent(model_registry, model_name=config["email_agent_model_name"], tool_registry=tool_registry)
    
    user_input = "무신사에서 온 메일 요약해서 김민혁에게 보내줘"
    tasks = [
      {
        "task_id": "t1",
        "agent": "Email Agent",
        "objective": "Read the email from Muji on the Muji website.",
        "depends_on": [],
        "acceptance": "The email has been successfully read from Muji."
      },
      {
        "task_id": "t2",
        "agent": "Answer Agent",
        "objective": "Summarize the content of the email from Muji.",
        "depends_on": ["t1"],
        "acceptance": "The email content has been summarized."
      },
      {
        "task_id": "t3",
        "agent": "Email Agent",
        "objective": "Send the summarized email to Kim Minhyuk.",
        "depends_on": ["t2"],
        "acceptance": "The summarized email has been successfully sent to Kim Minhyuk."
      }
    ]

    history = []
    for task in tasks:
        if task['agent']=="Email Agent":
            print("\nSTART! START! START! START! START! START! START! START! START! START! START! START! START! START! \n")
            input_task = {'objective': task['objective'], 'acceptance': task['acceptance']}
            result = agent.generate(user_input, input_task, history)
            break
        print(result)

    


if __name__=="__main__":
    model_registry = Modules.ModelRegistry()
    # router_test(model_registry)
    # planner_test(model_registry)
    # search_agent_test(model_registry)
    # email_tool_test()
    email_agent_test(model_registry)
