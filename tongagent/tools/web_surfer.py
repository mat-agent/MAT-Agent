# Shamelessly stolen from Microsoft Autogen team: thanks to them for this great resource!
# https://github.com/microsoft/autogen/blob/gaia_multiagent_v01_march_1st/autogen/browser_utils.py
import os
import re
from typing import Tuple, Optional
from transformers.agents.agents import Tool
import time
import requests
import mimetypes
from tongagent.tools.browser import SimpleTextBrowser, google_custom_search
from tongagent.llm_engine.gpt import get_tonggpt_open_ai_client
from tongagent.utils import load_config, CACHE_FOLDER
# load_dotenv(override=True)

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

browser_config = {
    "viewport_size": 1024 * 5,
    "downloads_folder": CACHE_FOLDER,
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
}


browser_config["serpapi_key"] = os.environ["SERPAPI_API_KEY"]

browser = SimpleTextBrowser(**browser_config)


# Helper functions
def _browser_state() -> Tuple[str, str]:
    header = f"Address: {browser.address}\n"
    if browser.page_title is not None:
        header += f"Title: {browser.page_title}\n"

    current_page = browser.viewport_current_page
    total_pages = len(browser.viewport_pages)

    address = browser.address
    for i in range(len(browser.history)-2,-1,-1): # Start from the second last
        if browser.history[i][0] == address:
            header += f"You previously visited this page {round(time.time() - browser.history[i][1])} seconds ago.\n"
            break

    header += f"Viewport position: Showing page {current_page+1} of {total_pages}.\n"
    return (header, browser.viewport)




class SearchInformationTool(Tool):
    name="informational_web_search"
    description="Perform a google search query then return the search results."
    inputs = {
        "query": {
            "type": "string",
            "description": "The google web search query to perform."
        }
    }
    # inputs["filter_year"]= {
    #     "type": "text",
    #     "description": "[Optional parameter]: filter the search results to only include pages from a specific year. For example, '2020' will only include pages from 2020. Make sure to use this parameter if you're trying to search for articles from a specific date!"
    # }
    output_type = "string"

    def forward(self, query: str) -> str:
        filter_year = None
        browser.visit_page(f"google: {query}", filter_year=filter_year)
        header, content = _browser_state()
        return header.strip() + "\n=======================\n" + content
import re
    
class WebQATool(Tool):
    name = "web_qa"
    description = "Read the entire web page and try to answer the question."
    inputs = {"question": {"type": "string", "description": "The question that you want to answer with this web page."}}
    output_type = "string"
    
    
    client, _ = get_tonggpt_open_ai_client()
    model_name = load_config().web_qa.model_name
    system_prompt = '''You are a helpful assitant. You will be given a number of text chunks. Your goal is that answering the question based on the current text chunk and the chunk history. You need to determine if you want to read more text chunks if you cannot answer the question. Below are text chunk informations:
[Total Text Chunk {total}]
[Current Text Chunk {current_index}]
{text_chunk}
[Chunk History]
{previous_chunks}
The text chunk could be irrelevant to the question or partially relative to the question. Format your answer as
Thought: <your reasoning process>
Answer: <answer or extract relevant information to the question>
Next Action: <You should write 'page down' here if you think you need to read next text chunk. You should write 'stop' when you think you have enough information to answer the question.>
'''

    def forward(self, question):
        page = browser.viewport_current_page
        print("Total page", len(browser.viewport_pages))
        previous_chunks = []
        
        while page < len(browser.viewport_pages):
            # print("Current page content", browser.viewport)
            previous_chunks_prompt = ""
            if len(previous_chunks) > 0:
                previous_chunks_prompt = "You should also consider your reading history from previous chunks. Below are the reading history:\n"
                for chunk_id, chunk in enumerate(previous_chunks):
                    previous_chunks_prompt += f"Chunk {chunk_id}: {chunk}\n"
            system_prompt = self.system_prompt.format(
                text_chunk=browser.viewport,
                previous_chunks=previous_chunks_prompt,
                total=len(browser.viewport_pages),
                current_index=page+1)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}"}
            ]
            response = self.client.chat.completions.create(
                messages=messages,
                max_tokens=512,
                model=self.model_name
            )
            print(response.choices[0].message.content)
            print("=====")
            content = response.choices[0].message.content
            answer, next_action = self.extract_by_pattern(content)
            if next_action == "page down":
                previous_chunks.append(answer)
                browser.page_down()
                page += 1
            else:
                break
            
        if answer is None:
            answer = "This website cannot answer the question."
        print("web_qa:", answer)
        return answer
    
    def extract_by_pattern(self, text):
        answer_pattern = r'Answer:(.*?)Next Action'
        next_action_pattern = r'Next Action:(.*)'
        found = re.search(answer_pattern, text, re.DOTALL)
        answer = None
        next_action = None
        if found:
            answer = found.group(1).strip()
        
        found = re.search(next_action_pattern, text, re.DOTALL)
        if found:
            next_action = found.group(1).strip()
        return answer, next_action
    
class NavigationalSearchTool(Tool):
    name = "navigational_web_search"
    description = "Perform a google web search query then immediately navigate to the top result. Useful, for example, to navigate to a particular Wikipedia article or other known destination. Equivalent to Google's \"I'm Feeling Lucky\" button. You should print the response from this tool to see the content of the webpage."
    inputs = {"query": {"type": "string", "description": "The navigational web search query to perform."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        browser.visit_page(f"google: {query}")

        # Extract the first line
        m = re.search(r"\[.*?\]\((http.*?)\)", browser.page_content)
        if m:
            browser.visit_page(m.group(1))

        # Return where we ended up
        header, content = _browser_state()
        return header.strip() + "\n=======================\n" + content


class VisitTool(Tool):
    name="visit_page"
    description="Visit a webpage at a given URL and return its text."
    inputs = {"url": {"type": "string", "description": "The relative or absolute url of the webapge to visit."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        browser.visit_page(url)
        header, content = _browser_state()
        return header.strip() + "\n=======================\n" + content


class DownloadTool(Tool):
    name="download_file"
    description="""
Download a file at a given URL. The file should be of this format: [".xlsx", ".pptx", ".wav", ".mp3", ".png", ".docx"]
After using this tool, for further inspection of this page you should return the download path to your manager via final_answer, and they will be able to inspect it.
DO NOT use this tool for .pdf or .txt or .htm files: for these types of files use visit_page with the file url instead."""
    inputs = {"url": {"type": "string", "description": "The relative or absolute url of the file to be downloaded."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        if os.path.exists(url):
            return f"File is in local path {url}"
        if "arxiv" in url:
            url = url.replace("abs", "pdf")
        response = requests.get(url)
        content_type = response.headers.get("content-type", "")
        extension = mimetypes.guess_extension(content_type)
        os.makedirs(CACHE_FOLDER, exist_ok=True)
        if extension and isinstance(extension, str):
            new_path = f"{CACHE_FOLDER}/file{extension}"
        else:
            new_path = f"{CACHE_FOLDER}/file.object"

        with open(new_path, "wb") as f:
            f.write(response.content)

        if "pdf" in extension or "txt" in extension or "htm" in extension:
            raise Exception("Do not use this tool for pdf or txt or html files: use visit_page instead.")

        return f"File was downloaded and saved under path {new_path}."
    

class PageUpTool(Tool):
    name="page_up"
    description="Scroll the viewport UP one page-length in the current webpage and return the new viewport content."
    inputs = dict()
    output_type = "string"

    def forward(self) -> str:
        browser.page_up()
        header, content = _browser_state()
        return header.strip() + "\n=======================\n" + content

class ArchiveSearchTool(Tool):
    name="find_archived_url"
    description="Given a url, searches the Wayback Machine and returns the archived version of the url that's closest in time to the desired date."
    inputs={
        "url": {"type": "string", "description": "The url you need the archive for."},
        "date": {"type": "string", "description": "The date that you want to find the archive for. Give this date in the format 'YYYYMMDD', for instance '27 June 2008' is written as '20080627'."}
    }
    output_type = "string"

    def forward(self, url, date) -> str:
        archive_url = f"https://archive.org/wayback/available?url={url}&timestamp={date}"
        response = requests.get(archive_url).json()
        try:
            closest = response["archived_snapshots"]["closest"]
        except:
            raise Exception(f"Your url was not archived on Wayback Machine, try a different url.")
        target_url = closest["url"]
        browser.visit_page(target_url)
        header, content = _browser_state()
        return f"Web archive for url {url}, snapshot taken at date {closest['timestamp'][:8]}:\n" + header.strip() + "\n=======================\n" + content


class PageDownTool(Tool):
    name="page_down"
    description="Scroll the viewport DOWN one page-length in the current webpage and return the new viewport content."
    output_type = "string"
    inputs = dict()
    def forward(self, ) -> str:
        browser.page_down()
        header, content = _browser_state()
        return header.strip() + "\n=======================\n" + content


class FinderTool(Tool):
    name="find_on_page_ctrl_f"
    description="Scroll the viewport to the first occurrence of the search string and return the content as string. This is equivalent to Ctrl+F."
    inputs = {"search_string": {"type": "string", "description": "The string to search for on the page. This search string supports wildcards like '*'" }}
    output_type = "string"

    def forward(self, search_string: str) -> str:
        find_result = browser.find_on_page(search_string)
        header, content = _browser_state()

        if find_result is None:
            return header.strip() + f"\n=======================\nThe search string '{search_string}' was not found on this page."
        else:
            return header.strip() + "\n=======================\n" + content


class FindNextTool(Tool):
    name="find_next"
    description="Scroll the viewport to next occurrence of the search string and return the content as string. This is equivalent to finding the next match in a Ctrl+F search."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        find_result = browser.find_next()
        header, content = _browser_state()

        if find_result is None:
            return header.strip() + "\n=======================\nThe search string was not found on this page."
        else:
            return header.strip() + "\n=======================\n" + content
