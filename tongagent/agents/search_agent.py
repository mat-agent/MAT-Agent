from transformers.agents import ReactJsonAgent
from transformers.agents.prompts import DEFAULT_REACT_JSON_SYSTEM_PROMPT
from tongagent.llm_engine.gpt import TongGPTEngine
from tongagent.tools.web_surfer import (
    SearchInformationTool,
    NavigationalSearchTool,
    VisitTool,
    PageUpTool,
    PageDownTool,
    FinderTool,
    FindNextTool,
    ArchiveSearchTool,
    WebQATool,
    DownloadTool,
)

from transformers.agents import Tool
import json
from tongagent.utils import load_config
from tongagent.llm_engine import QwenEngine
def create_surfer_agent():
    WEB_TOOLS = [
        SearchInformationTool(),
        NavigationalSearchTool(),
        VisitTool(),
        PageUpTool(),
        PageDownTool(),
        FinderTool(),
        FindNextTool(),
        ArchiveSearchTool(),
        DownloadTool(),
    ]
    config = load_config()
    llm_engine = TongGPTEngine(model="search_agent")
    if config.search_agent.model_name.startswith("Qwen"):
        llm_engine = QwenEngine(config.search_agent.model_name)
    surfer_agent = ReactJsonAgent(
        llm_engine = llm_engine,
        tools = WEB_TOOLS,
        max_iterations = 12,
        verbose = 0,
        system_prompt = DEFAULT_REACT_JSON_SYSTEM_PROMPT + "\nAdditionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information.",
        planning_interval=4,
    )
    return surfer_agent

class SearchTool(Tool):
    name = "ask_search_agent"
    description = """This will send a message to a team member that will browse the internet to answer your question. Ask him for all your web-search related questions, but he's unable to do problem-solving. Provide him as much context as possible, in particular if you need to search on a specific timeframe! And don't hesitate to provide them with a complex search task, like finding a difference between two webpages."""

    inputs = {
        "query": {
            "description": "Your question, as a natural language sentence with a verb! You are talking to an human, so provide them with as much context as possible! DO NOT ASK a google-like query like 'paper about fish species 2011': instead ask a real sentence like: 'What appears on the last figure of a paper about fish species published in 2011?'",
            "type": "string",
        }
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.surfer_agent = create_surfer_agent()
        
    def forward(self, query: str) -> str:
        final_answer = self.surfer_agent.run(f"""
You've been submitted this request by your manager: '{query}'

You're helping your manager solve a wider task: so make sure to not provide a one-line answer, but give as much information as possible so that they have a clear understanding of the answer.

Your final_answer WILL HAVE to contain these parts:
### 1. Search outcome (short version):
### 2. Search outcome (extremely detailed version):
### 3. Additional context:

Put all these in your final_answer, everything that you do not pass as an argument to final_answer will be lost.

You can navigate to .txt or .pdf online files using your 'visit_page' tool.
If it's another format, you can return the url of the file, and your manager will handle the download and inspection from there.

And even if your search is unsuccessful, please return as much context as possible, so they can act upon this feedback.
""")
        # HACK ignore a lot of context
        # answer = "Here is the report from your team member's search:\n"
        # for message in self.surfer_agent.write_inner_memory_from_logs():
        #     content = message['content']
        #     if 'tool_arguments' in str(content):
        #         if len(str(content)) < 1000 or "[FACTS]" in str(content):
        #             answer += "" + str(content) + "\n"
        #         else:
        #             try:
        #                 answer += f"{json.loads(content)['tool_name']}\n"
        #             except:
        #                 answer += f"{content[:1000]}(...)\n"
        #     else:
        #         if len(str(content)) > 2000:
        #             answer += ">>> Tool output too long to show, showing only the beginning:\n" + str(content)[:500] + '\n(...)\n\n'
        #         else:
        #             answer += ">>> "+ str(content) + "\n\n"
        
        # answer += "\nNow here is the team member's final answer deducted from the above:\n"
        answer = "Search Result:\n"
        answer += str(final_answer)
        # print("SearchTool output:\n", answer)
        return answer