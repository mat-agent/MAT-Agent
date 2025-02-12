import string
import shortuuid
import os
from typing import Union

from omegaconf import OmegaConf, DictConfig, ListConfig

CACHE_FOLDER = ".cache"
os.makedirs(CACHE_FOLDER, exist_ok=True)

def get_uuid_builder() -> shortuuid.ShortUUID:
    alphabet = string.ascii_lowercase + string.digits
    su = shortuuid.ShortUUID(alphabet=alphabet)
    return su

def load_config() -> Union[DictConfig, ListConfig]:
    if "AGENT_CONFIG" in os.environ and len(os.environ["AGENT_CONFIG"]) > 0:
        return OmegaConf.load(os.environ["AGENT_CONFIG"])
    
    if "RUN_MODE" in os.environ and os.environ["RUN_MODE"] == "eval":
        return OmegaConf.load("configs/agent_config.yaml")
    
    return OmegaConf.load("configs/agent_config.yaml")

import time
uuid_builder = get_uuid_builder()

def gen_random_id():
    return f"{int(time.time()*1000)}_{uuid_builder.random(length=8)}"    

if __name__ == "__main__":
    print(load_config())
    print(load_config().search_engine[0].cx)