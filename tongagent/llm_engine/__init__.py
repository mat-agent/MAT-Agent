from tongagent.llm_engine.gpt import TongGPTEngine
from tongagent.llm_engine.mini_cpm import MiniCPMEngine
from tongagent.llm_engine.qwen import QwenEngine
from tongagent.llm_engine.internvl2 import InternVL2Engine
from tongagent.utils import load_config
from tongagent.llm_engine.llava import LLaVAEngine
def get_llm_engine(
    engine_type=None,
    lora_path=None,
    disable_vision=False,
):
    config = load_config()
    if engine_type is None:
        engine_type = config.agent_controller.engine_type
        
    if engine_type == "qwen":
        return QwenEngine(
            model_name=config.qwen.model_name,
            lora_path=lora_path
        )
    elif engine_type == "tonggpt":
        return TongGPTEngine(engine_type)
    elif engine_type == "minicpm":
        return MiniCPMEngine(
            model=lora_path,
            disable_vision=disable_vision
        )
    elif engine_type == "internvl2":
        return InternVL2Engine(
            model_name=config.internvl2.model_name,
            lora_path=lora_path
        )
    elif engine_type == "llava":
        return LLaVAEngine(
            model_name=config.llava.model_name,
        )
    else:
        raise ValueError(f"Unknown LLM engine {engine_type}")
    