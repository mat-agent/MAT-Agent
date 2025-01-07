from tongagent.utils import load_config

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI
from openai import AzureOpenAI

from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import Union, Any, Optional, List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field


def get_tonggpt_open_ai_client():
    config = load_config()
    endpoint = f"https://api.tonggpt.mybigai.ac.cn/proxy/{config.tonggpt.region}"
    return AzureOpenAI(
            api_key=config.tonggpt.api_key,
            api_version="2024-02-01",
            azure_endpoint=endpoint,
    ), config.tonggpt.model_name
    
class TongGPTChatModel(BaseChatModel):
    client: Any = Field(default=None, exclude=True)
    model_name: str
    # client, model_name = get_tonggpt_open_ai_client()
    
    _llm_type = "tonggpt"
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        
        messages_req = []
        for msg in messages:
            if type(msg) is SystemMessage:
                messages_req.append(
                    {"role": "system", "content": msg.content}
                )
            elif type(msg) is AIMessage:
                messages_req.append(
                    {"role": "assitant", "content": msg.content}
                )
            elif type(msg) is HumanMessage:
                messages_req.append(
                    {"role": "user", "content": msg.content}
                )
            else:
                raise ValueError("unk msg type")
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages_req,
            stop=stop
        )
        print(response)
        generation = ChatGeneration(
            text=response.choices[0].message.content,
            message=AIMessage(content=response.choices[0].message.content)
        )
        return ChatResult(generations=[generation],
                          llm_output=dict())

    
class RAGWebBrowser():
    def __init__(self) -> None:
        client, model_name = get_tonggpt_open_ai_client()
        llm = TongGPTChatModel(client=client, model_name=model_name)
        self.llm = llm
        
        
if __name__ == "__main__":
    browser = RAGWebBrowser()
    output = browser.llm.invoke(
        "Hello"
    )
    print(output)
    