import os
from time import time

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
# final_answer_agent = TextAgent(config['text_agent_model_name'])
def router_test(model_registry):
    print("\nRouter Test!\n")
    agents = [
        Agents.DummyAgent("model_name", name=Agents.SearchAgent.name, description=Agents.SearchAgent.description), 
        Agents.DummyAgent("model_name", name=Agents.EmailAgent.name, description=Agents.EmailAgent.description), 
        Agents.DummyAgent("model_name", name=Agents.TextAgent.name, description=Agents.TextAgent.description)
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
            print(router_output)

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
        Agents.DummyAgent("model_name", name=Agents.TextAgent.name, description=Agents.TextAgent.description)
        ]
    agents = {agent.name:agent for agent in agents}

    torch.cuda.empty_cache()
    planner = Modules.Planner(model_registry, config["planner_model_name"], available_agents=agents)

    user_input = "쿠팡에서 온 메일 요약해서 김민혁에게 보내줘"
    dummy_router_output = f"""
{{
  "route": "planner",
  "clarifying_question": "",
  "using_agents": [
    "Answer Agent",
    "Email Agent"
  ],
  "preserve_spans": [
    "쿠팡",
    "김민혁"
  ],
  "high_level_intent": "Summarize emails from {{P0}} and send the summary to {{P1}} via email."
}}
""".strip()
    dummy_router_output = functions.loads_json(dummy_router_output)
    # print(functions.dumps_json(dummy_router_output))

    history = []
    with torch.no_grad():
        planner_output:dict = planner.generate(user_input, dummy_router_output, history)

    torch.cuda.empty_cache()


def email_tool_test():
    print("Email Tool Test!\n")
    load_dotenv()
    
    EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
    APP_PASSWARD = os.getenv("APP_PASSWARD")
    SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
    IMAP_HOST = os.getenv("IMAP_HOST", "imap.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

    email_search_tool = email_tool.EmailSearchTool(EMAIL_ADDRESS, APP_PASSWARD, IMAP_HOST)
    email_result = email_search_tool(from_contains="쿠팡", limit=1)
    print("\n[RESULT]:\n\n")
    for email in email_result:
      for key, val in email.items():
          print(f"{key} - {val}")
      print()

    # email_get_tool = email_tool.EmailGetTool(EMAIL_ADDRESS, APP_PASSWARD, IMAP_HOST)
    # email_get_result = email_get_tool(ids=["7393", "7383", "7364", "7349"], include_body=True, max_body_chars=10000)
    
    # for email in email_get_result:
    #   for key, val in email.items():
    #       print(f"{key} - {val}")
    #   print()


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
    
    user_input = "쿠팡에서 온 메일 요약해서 김민혁에게 보내줘"
    tasks = [
        {
        "task_id": "t1",
        "agent": "Email Agent",
        "objective": "Retrieve the email content from 쿠팡",
        "depends_on": [],
        "acceptance_criteria": "The raw email content from 쿠팡 is successfully retrieved."
        },
        {
        "task_id": "t2",
        "agent": "Answer Agent",
        "objective": "Summarize the email content",
        "depends_on": ["t1"],
        "acceptance_criteria": "A summary of the email content is generated."
        },
        {
        "task_id": "t3",
        "agent": "Email Agent",
        "objective": "Send the summary to 김민혁 via email",
        "depends_on": ["t2"],
        "acceptance_criteria": "The summary is successfully sent to 김민혁 via email."
        }
    ]

    history = []
    for task in tasks:
        if task['agent']=="Email Agent":
            print("\nSTART! START! START! START! START! START! START! START! START! START! START! START! START! START! \n")
            input_task = {'objective': task['objective'], 'acceptance_criteria': task['acceptance_criteria']}
            task_str = f"[TASK]:\n{functions.dumps_json(task)}"
            result = agent.generate(task_str, history)
            break
            print(result)
            exit()
    
def answer_agent_test(model_registry):
    answer_agent = Agents.TextAgent(model_registry, config["text_agent_model_name"])
    refer = {"t1":  [{'id': '7393', 'from_addr': '쿠팡페이 <no-reply@coupangpay.com>', 'to': 'rudwo6769@gmail.com', 'subject': '[쿠팡페이] 개인정보 이용∙제공 내역 및 수집 출처 안내', 'date': 'Thu, 25 Dec 2025 08:11:14 +0900 (KST)', 'body': 'Document \n\n 쿠팡페이 개인정보 이용·제공 내역 및 수집 출처 안내\n\n 안녕하세요 쿠팡페이입니다.\n\n 쿠팡페이를 이용해 주시는 고객 여러분께 감사드리며,\n\n 개인정보 보호법 제20조의 2(개인정보 이용•제공 내역의 통지) 및 동법 제 20조(정보주체 이외로부터 수집한 개인정보의 수집 출처 등 통지)에 따라 개인정보 이용•제공 내역 및 수집 출처에 관하여 안내드립니다.\n\n 고객님께서는 아래 버튼으로 개인정보 처리 내용을 확인하실 수 있습니다.\n\n 쿠팡페이 개인정보 처리방침 확인 \n\n 쿠팡페이 개인정보 수집출처 확인 \n\n 당사가 타사로부터 주기적으로 제공받은 고객님의 개인정보 처리를 원하지 않는 경우, 개인정보 처리의 정지를 요구하거나 동의를 철회하실 수 있습니다.\n\n 단, 개인정보 처리의 정지를 요구하거나 동의를 철회하실 경우, 쿠팡페이 서비스 이용에 제한이 있을 수 있습니다.\n\n 쿠팡페이는 고객님의 소중한 개인정보를 안전하게 관리하기 위하여 최선을 다하겠습니다.\n\n 감사합니다.\n\n 본 메일은 쿠팡페이 이용약관에 동의하신 모든 고객님께 연 1회 발송됩니다.\n\n 2025년 12월 23일 기준으로 작성되었으며, 이후 쿠팡페이에서 탈퇴하셨더라도 메일을 받으실 수 있습니다.\n\n 개인정보 이용·제공 내역 안내 메일 수신을 원치않으시는 경우, 수신거부 를 클릭하세요. \n(단, 개인정보 수집 출처 안내 메일은 관련 법령에 따라 수신동의 여부와 관계없이 발송되므로, 개인정보 이용·제공 내역 안내 메일에 수신거부 하시더라도 별도 발송될 수 있습니다.)\n\n 쿠팡페이㈜ |\n 대표이사: IYER VIJESH V, 정찬묵 |\n 사업자등록번호: 105-87-65162 |\n (05510) 서울특별시 송파구 송파대로 570, 9층(신천동) |\n Tel: 1670-9892\n\n ⓒCoupang Pay, Ltd All rights reserved.\n\n ※ 본 메일은 발신 전용 메일이며, 법률에 근거하여 발송됩니다.'}]}
    task = {
      "task_id": "t2",
      "agent": "Answer Agent",
      "objective": "Summarize the email from 쿠팡",
      "depends_on": [
        "t1"
      ],
      "acceptance_criteria": "Email summary is provided"
    }
    user_input = f"""
[REFERENCE]:
{functions.dumps_json(refer)}

[TASK]:
{functions.dumps_json(task)}
""".strip()
    # print(user_input)
    print(answer_agent.generate(user_input=user_input, history=[]))


def orchestrator_test():
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
    
    user_input = "김민재와 손흥민의 팀은 어디야?"
    # user_input = "okx에서 온 메일 요약해서 김진호에게 보내줘"
    # user_input = "이차방정식과 삼차방정식 설명"

    history=[]

    start_time = time()
    print(user_input)
    result = orchestrator.run(user_input=user_input, history=history)
    end_time = time()
    print(f"\nOrchestrator Result:\n{result}\n")
    print(f"Total Durations: {(end_time-start_time):.2f}\n")
    history += [{"role":"user", "content":user_input}, {"role":"assistant", "content":result}]
    

if __name__=="__main__":
    model_registry = Modules.ModelRegistry()
    # router_test(model_registry)
    # planner_test(model_registry)
    # answer_agent_test(model_registry)
    # search_agent_test(model_registry)
    # email_tool_test()
    # email_agent_test(model_registry)
    orchestrator_test()

