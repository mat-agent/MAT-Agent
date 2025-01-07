from typing import Callable, List
from transformers.agents.prompts import DEFAULT_REACT_CODE_SYSTEM_PROMPT
from transformers.agents.tools import DEFAULT_TOOL_DESCRIPTION_TEMPLATE, Tool
from tongagent.tools.tool_box import get_general_tool_box, get_tool_box_gaia
from tongagent.llm_engine.gpt import TongGPTEngine, get_tonggpt_open_ai_client
from tongagent.prompt import DEFAULT_REACT_CODE_SYSTEM_PROMPT, FORMAT_ANSWER_PROMPT_GAIA
from transformers.agents import ReactCodeAgent, HfApiEngine
from transformers.agents.tools import DEFAULT_TOOL_DESCRIPTION_TEMPLATE
from transformers.agents.llm_engine import MessageRole
from typing import Any
from langchain.prompts import ChatPromptTemplate
        
def create_agent() -> ReactCodeAgent:
    llm_engine = TongGPTEngine()
    
    react_agent = ReactCodeAgent(
        llm_engine=llm_engine,
        # tools=TASK_SOLVING_TOOLBOX+WEB_TOOLS,
        tools=get_general_tool_box(),
        max_iterations=15,
        verbose=0,
        memory_verbose=True,
        system_prompt=DEFAULT_REACT_CODE_SYSTEM_PROMPT,
        add_base_tools=False,
        additional_authorized_imports=[
            "requests",
            "zipfile",
            "os",
            "pandas",
            "numpy",
            "sympy",
            "json",
            "bs4",
            "pubchempy",
            "xml",
            "yahoo_finance",
            "Bio",
            "sklearn",
            "scipy",
            "pydub",
            "io",
            "PIL",
            "chess",
            "PyPDF2",
            "pptx",
            "torch",
            "datetime",
            "csv",
            "fractions",
            "matplotlib",
            "pickle"
        ],
        planning_interval=None
    )
    return react_agent

