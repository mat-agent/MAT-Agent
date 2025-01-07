from typing import Callable, List
import json

from transformers.agents.prompts import DEFAULT_REACT_CODE_SYSTEM_PROMPT,PROMPTS_FOR_PLAN_UPDATE, SYSTEM_PROMPT_FACTS, PROMPTS_FOR_INITIAL_PLAN, SYSTEM_PROMPT_FACTS_UPDATE, USER_PROMPT_FACTS_UPDATE, PLAN_UPDATE_FINAL_PLAN_REDACTION
from transformers.agents.tools import DEFAULT_TOOL_DESCRIPTION_TEMPLATE, Tool
from tongagent.tools.tool_box import get_tool_box_gaia
from tongagent.llm_engine.gpt import TongGPTEngine, get_tonggpt_open_ai_client
from tongagent.prompt import DEFAULT_REACT_CODE_SYSTEM_PROMPT, FORMAT_ANSWER_PROMPT_GAIA
from transformers.agents import ReactCodeAgent, HfApiEngine
from transformers.agents.tools import DEFAULT_TOOL_DESCRIPTION_TEMPLATE
from transformers.agents.llm_engine import MessageRole
from typing import Any
from langchain.prompts import ChatPromptTemplate
# from transformers.agents.prompts import DEFAULT_REACT_CODE_SYSTEM_PROMPT
from transformers.agents.agents import AgentGenerationError, AgentParsingError, parse_code_blob, BASE_PYTHON_TOOLS, AgentExecutionError

class ReactCodeGAIAAgent(ReactCodeAgent):
    def __init__(
        self, 
        tools: List[Tool], 
        llm_engine: Callable = HfApiEngine(), 
        system_prompt: str = DEFAULT_REACT_CODE_SYSTEM_PROMPT, 
        tool_description_template: str = DEFAULT_TOOL_DESCRIPTION_TEMPLATE, 
        additional_authorized_imports: List[str] | None = None, planning_interval: int | None = None, 
        **kwargs):
        super().__init__(tools, llm_engine, system_prompt, tool_description_template, additional_authorized_imports, planning_interval, **kwargs)
        template = ChatPromptTemplate.from_template(FORMAT_ANSWER_PROMPT_GAIA)
        client, model_name = get_tonggpt_open_ai_client()
        self.template = template
        self.client = client
        self.model_name = model_name
        
        self.plan = None
        self.image_paths = None
        
    def set_plan(self, plan):
        self.plan = plan
        
    def provide_final_answer(self, task) -> str:
        """
        This method provides a final answer to the task, based on the logs of the agent's interactions.
        """
        self.prompt = [
            {
                "role": MessageRole.SYSTEM,
                "content": "An agent tried to answer an user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:",
            }
        ]
        self.prompt += self.write_inner_memory_from_logs()[1:]
        self.prompt += [
            {
                "role": MessageRole.USER,
                "content": f"Based on the above, please provide an answer to the following user request:\n{task}\nFinish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]\nYOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string. If you are unable to solve the question, make a well-informed EDUCATED GUESS based on the information we have provided. Your EDUCATED GUESS should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. DO NOT OUTPUT 'I don't know', 'Unable to determine', etc.",
            }
        ]
        try:
            return self.llm_engine(self.prompt)
        except Exception as e:
            return f"Error in generating final llm output: {e}."
    
    def direct_run(self, task: str):
        final_answer = super().direct_run(task)
        prompt_input = {
            "question": task,
            "answer": final_answer
        }
        
        prompt = self.template.invoke(prompt_input)
        prompt.to_messages()
        messages = [
            {"role": "user", "content": prompt.to_messages()[0].content}
        ]
        
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name
        )
        
        final_answer = response.choices[0].message.content
        if "Educated guess:" in final_answer:
            final_answer = final_answer.replace("Educated guess:", "").strip()
        return final_answer

    def planning_step(self, task, is_first_step: bool = False, iteration: int = None):
        if self.plan is None:
            return super().planning_step(task, is_first_step, iteration)
        
        if is_first_step:
            message_prompt_facts = {"role": MessageRole.SYSTEM, "content": SYSTEM_PROMPT_FACTS}
            message_prompt_task = {
                "role": MessageRole.USER,
                "content": f"""Here is the task:
```
{task}
```
Now begin!""",
            }

            answer_facts = self.llm_engine([message_prompt_facts, message_prompt_task])

            message_system_prompt_plan = {
                "role": MessageRole.SYSTEM,
                "content": PROMPTS_FOR_INITIAL_PLAN[self.plan_type]["system"],
            }
            message_user_prompt_plan = {
                "role": MessageRole.USER,
                "content": PROMPTS_FOR_INITIAL_PLAN[self.plan_type]["user"].format(
                    task=task,
                    tool_descriptions=self._toolbox.show_tool_descriptions(self.tool_description_template),
                    answer_facts=answer_facts,
                ),
            }
            answer_plan = self.plan

            final_plan_redaction = f"""Here is the plan of action that I will follow to solve the task:
```
{answer_plan}
```"""
            final_facts_redaction = f"""Here are the facts that I know so far:
```
{answer_facts}
```""".strip()
            self.logs.append({"plan": final_plan_redaction, "facts": final_facts_redaction})
            self.logger.info("===== Initial plan: =====")
            self.logger.info(final_plan_redaction)
            self.logger.info("===== Initial facts: =====")
            self.logger.info(final_facts_redaction)
        else:  # update plan
            agent_memory = self.write_inner_memory_from_logs(
                summary_mode=False
            )  # This will not log the plan but will log facts

            # Redact updated facts
            facts_update_system_prompt = {
                "role": MessageRole.SYSTEM,
                "content": SYSTEM_PROMPT_FACTS_UPDATE,
            }
            facts_update_message = {
                "role": MessageRole.USER,
                "content": USER_PROMPT_FACTS_UPDATE,
            }
            facts_update = self.llm_engine([facts_update_system_prompt] + agent_memory + [facts_update_message])

            # Redact updated plan
            plan_update_message = {
                "role": MessageRole.SYSTEM,
                "content": PROMPTS_FOR_PLAN_UPDATE[self.plan_type]["system"].format(task=task),
            }
            plan_update_message_user = {
                "role": MessageRole.USER,
                "content": PROMPTS_FOR_PLAN_UPDATE[self.plan_type]["user"].format(
                    task=task,
                    tool_descriptions=self._toolbox.show_tool_descriptions(self.tool_description_template),
                    facts_update=facts_update,
                    remaining_steps=(self.max_iterations - iteration),
                ),
            }
            plan_update = self.llm_engine(
                [plan_update_message] + agent_memory + [plan_update_message_user], stop_sequences=["<end_plan>"]
            )

            # Log final facts and plan
            final_plan_redaction = PLAN_UPDATE_FINAL_PLAN_REDACTION.format(task=task, plan_update=plan_update)
            final_facts_redaction = f"""Here is the updated list of the facts that I know:
```
{facts_update}
```"""
            self.logs.append({"plan": final_plan_redaction, "facts": final_facts_redaction})
            self.logger.info("===== Updated plan: =====")
            self.logger.info(final_plan_redaction) 
    
    def step(self):
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        The errors are raised here, they are caught and logged in the run() method.
        """
        agent_memory = self.write_inner_memory_from_logs()

        self.prompt = agent_memory.copy()

        self.logger.debug("===== New step =====")

        # Add new step in logs
        current_step_logs = {}
        self.logs.append(current_step_logs)
        current_step_logs["agent_memory"] = agent_memory.copy()

        self.logger.info("===== Calling LLM with these last messages: =====")
        self.logger.info(self.prompt[-2:])

        try:
            llm_output = self.llm_engine(self.prompt, stop_sequences=["<end_action>", "Observation:"], image_paths=self.image_paths)
        except Exception as e:
            raise AgentGenerationError(f"Error in generating llm output: {e}.")

        self.logger.debug("===== Output message of the LLM: =====")
        self.logger.debug(llm_output)
        current_step_logs["llm_output"] = llm_output

        # Parse
        self.logger.debug("===== Extracting action =====")
        try:
            rationale, raw_code_action = self.extract_action(llm_output=llm_output, split_token="Code:")
        except Exception as e:
            self.logger.debug(f"Error in extracting action, trying to parse the whole output. Error trace: {e}")
            rationale, raw_code_action = llm_output, llm_output

        try:
            code_action = parse_code_blob(raw_code_action)
        except Exception as e:
            error_msg = f"Error in code parsing: {e}. Make sure to provide correct code"
            raise AgentParsingError(error_msg)

        current_step_logs["rationale"] = rationale
        current_step_logs["tool_call"] = {"tool_name": "code interpreter", "tool_arguments": code_action}

        # Execute
        self.log_code_action(code_action)
        try:
            result = self.python_evaluator(
                code_action,
                static_tools={
                    **BASE_PYTHON_TOOLS.copy(),
                    **self.toolbox.tools,
                },
                custom_tools=self.custom_tools,
                state=self.state,
                authorized_imports=self.authorized_imports,
            )
            information = self.state["print_outputs"]
            self.logger.warning("Print outputs:")
            self.logger.log(32, information)
            current_step_logs["observation"] = information
        except Exception as e:
            error_msg = f"Code execution failed due to the following error:\n{str(e)}"
            if "'dict' object has no attribute 'read'" in str(e):
                error_msg += "\nYou get this error because you passed a dict as input for one of the arguments instead of a string."
            raise AgentExecutionError(error_msg)
        for line in code_action.split("\n"):
            if line[: len("final_answer")] == "final_answer":
                self.logger.warning(">>> Final answer:")
                self.logger.log(32, result)
                current_step_logs["final_answer"] = result
        return current_step_logs

    
from tongagent.agents.search_agent import SearchTool
    
def create_agent_gaia() -> ReactCodeAgent:
    llm_engine = TongGPTEngine()
    toolbox = get_tool_box_gaia()
    toolbox += [SearchTool()]
    
    react_agent = ReactCodeGAIAAgent(
        llm_engine=llm_engine,
        # tools=TASK_SOLVING_TOOLBOX+WEB_TOOLS,
        tools=toolbox,
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
        planning_interval=1
    )
    return react_agent

from tongagent.tools.web_surfer  import WebQATool, VisitTool, SearchInformationTool
from tongagent.llm_engine.mini_cpm import MiniCPMEngine

def create_agent_simple_gaia() -> ReactCodeAgent:
    llm_engine = MiniCPMEngine()
    toolbox = get_tool_box_gaia()
    toolbox += [
        SearchInformationTool(),
        WebQATool(),
        VisitTool(),
    ]
    
    react_agent = ReactCodeGAIAAgent(
        llm_engine=llm_engine,
        tools=toolbox,
        max_iterations=15,
        verbose=1,
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