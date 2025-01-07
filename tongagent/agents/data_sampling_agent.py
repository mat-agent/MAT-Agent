from typing import Callable, List, Optional, Dict, Any
import os
import json
import shutil

from tongagent.tools.tool_box import get_visual_model_tool_box, get_visual_model_tool_box_for_gaia
from tongagent.llm_engine.gpt import TongGPTEngine
from tongagent.prompt import DEFAULT_REACT_CODE_SYSTEM_PROMPT

from tongagent.utils import load_config
config = load_config()
search_config = getattr(config, 'search_agent')
if search_config.type =='api':
    print ('use the gpt4o mini api as the search tool')
    from tongagent.agents.search_agent_api import SearchTool
else:
    print ('use the search agent as the search tool')
    from tongagent.agents.search_agent import SearchTool

from tongagent.utils import gen_random_id, CACHE_FOLDER

from transformers.agents import ReactCodeAgent, HfApiEngine
from transformers.agents.agents import AgentGenerationError, AgentParsingError, AgentError, AgentMaxIterationsError, parse_code_blob, BASE_PYTHON_TOOLS, AgentExecutionError
# from transformers.agents.prompts import DEFAULT_REACT_CODE_SYSTEM_PROMPT
from transformers.agents.tools import DEFAULT_TOOL_DESCRIPTION_TEMPLATE, Tool
from transformers.agents.python_interpreter import evaluate_python_code, LIST_SAFE_MODULES
class AgentToleranceError(AgentError):
    pass

SAFE_MODULES = list(set(LIST_SAFE_MODULES + [
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
    "pickle",
    "cv2"
]))

def evaluate_python_code_modify(
    code: str,
    static_tools: Optional[Dict[str, Callable]] = None,
    custom_tools: Optional[Dict[str, Callable]] = None,
    state: Optional[Dict[str, Any]] = None,
    authorized_imports: List[str] = SAFE_MODULES,
):
    print('authorized_imports', authorized_imports)
    result = evaluate_python_code(
        code,
        static_tools,
        custom_tools,
        state,
        authorized_imports
    )
    if state is not None and "print_outputs" in state and type(state["print_outputs"]) is str:
        state["print_outputs"] = state["print_outputs"] if len(state["print_outputs"]) > 0 else "No observation found from the code execution. You should use `print` function if need some information from the code execution."
    return result

class DataSamplingAgent(ReactCodeAgent):
    def __init__(
        self, 
        tools: List[Tool], 
        llm_engine: Callable = HfApiEngine(), 
        system_prompt: str = DEFAULT_REACT_CODE_SYSTEM_PROMPT, 
        tool_description_template: str = DEFAULT_TOOL_DESCRIPTION_TEMPLATE, 
        additional_authorized_imports: List[str] | None = None, 
        planning_interval: int | None = None,
        error_tolerance_count: int = -1, 
        **kwargs):
        super().__init__(tools=tools, llm_engine=llm_engine, system_prompt=system_prompt, tool_description_template=tool_description_template, additional_authorized_imports=additional_authorized_imports, planning_interval=planning_interval, **kwargs)
        
        self.image_paths = None
        self.python_evaluator = evaluate_python_code_modify
        self.error_tolerance_count = error_tolerance_count
        # self.authorized_imports = SAFE_MODULES
         
    def direct_run(self, task: str):
        """
        Runs the agent in direct mode, returning outputs only at the end: should be launched only in the `run` method.
        """
        final_answer = None
        iteration = 0
        error_count = 0
        # error_tolerance_count <= 0 disable this function
        while final_answer is None and iteration < self.max_iterations:
            if self.error_tolerance_count > 0 and error_count == self.error_tolerance_count:
                break
            try:
                if self.planning_interval is not None and iteration % self.planning_interval == 0:
                    self.planning_step(task, is_first_step=(iteration == 0), iteration=iteration)
                step_logs = self.step()
                if "final_answer" in step_logs:
                    final_answer = step_logs["final_answer"]
            except AgentError as e:
                self.logger.error(e, exc_info=1)
                self.logs[-1]["error"] = e
                error_count += 1
            finally:
                iteration += 1

        if final_answer is None and iteration == self.max_iterations:
            error_message = "Reached max iterations."
            final_step_log = {"error": AgentMaxIterationsError(error_message)}
            self.logs.append(final_step_log)
            self.logger.error(error_message, exc_info=1)
            final_answer = self.provide_final_answer(task)
            final_step_log["final_answer"] = final_answer
        elif final_answer is None and error_count == self.error_tolerance_count:
            error_message = f"Reached max execution exception. Max exception tolerance: {self.error_tolerance_count}."
            final_step_log = {"error": AgentToleranceError(error_message)}
            self.logs.append(final_step_log)
            self.logger.error(error_message, exc_info=1)
            final_answer = self.provide_final_answer(task)
            final_step_log["final_answer"] = final_answer
            
        return final_answer
    
    def set_image_paths(self, image_paths: List[str]):
        self.image_paths = image_paths
        
    def set_file_paths(self, file_paths: List[str]):
        self.file_paths = file_paths

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
        
        self.log_rationale_code_action(rationale, code_action)
        try:
            self.logger.info(f'authorized_imports {self.authorized_imports}')
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
    
    def save_trajectory(self, path=None, ground_truth=None, final_answer=None) -> str:
        if path is None:
            path = os.path.join(CACHE_FOLDER, gen_random_id())
        
        os.makedirs(path, exist_ok=True)
        print('write to', path)
        agent_memory = self.write_inner_memory_from_logs()
        saved_data = dict()
        saved_data["conversations"] = agent_memory
        saved_data["final_answer"] = str(final_answer)
        saved_data["ground_truth"] = ground_truth
        with open(os.path.join(path, "agent_memory.json"), "w") as f:
            json.dump(saved_data, f, indent=4, ensure_ascii=False)
        
        print(self.state)
        for k, v in self.state.items():
            if type(v) is not str:
                continue
            if os.path.exists(v) and not os.path.isdir(v):
                shutil.copy(v, path)
        return path
    
from tongagent.llm_engine.mini_cpm import MiniCPMEngine
from tongagent.llm_engine import get_llm_engine
def create_agent(
        llm_engine = "tonggpt",
        task = "gta",
        error_tolerance = 3,
        lora_path = None,
        disable_vision = False,
    ) -> DataSamplingAgent:
    
    print("create_agent called", llm_engine, task, error_tolerance, lora_path, disable_vision)
    llm_engine = get_llm_engine(
        engine_type=llm_engine, 
        lora_path=lora_path,
        disable_vision=disable_vision
    )
    
    tool_boxes = []
    if task == "gta":
        tool_boxes = get_visual_model_tool_box()
    else:
        tool_boxes = get_visual_model_tool_box_for_gaia()
    tool_boxes.append(SearchTool())
    react_agent = DataSamplingAgent(
        llm_engine=llm_engine,
        # tools=TASK_SOLVING_TOOLBOX+WEB_TOOLS,
        tools=tool_boxes,
        max_iterations=8,
        verbose=0,
        # memory_verbose=True,
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
            "pickle",
            "cv2"
        ],
        planning_interval=None,
        error_tolerance_count=error_tolerance
    )
    return react_agent

