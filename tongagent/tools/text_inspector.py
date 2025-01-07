from tongagent.tools.mdconvert import MarkdownConverter
from tongagent.llm_engine.gpt import TongGPTEngine

from typing import Optional
from transformers.agents.agents import Tool

class TextInspectorTool(Tool):
    name = "inspect_file_as_text"
    description = """You cannot load files yourself: instead call this tool to read a file as markdown text and ask questions about it.
This tool handles the following file extensions: [".html", ".htm", ".xlsx", ".pptx", ".wav", ".mp3", ".flac", ".pdf", ".docx"], and all other types of text files. IT DOES NOT HANDLE IMAGES."""

    inputs = {
        "question": {
            "description": "[Optional]: Your question, as a natural language sentence. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.",
            "type": "string",
        },
        "file_path": {
            "description": "The path to the file you want to read as text. Must be a '.something' file, like '.pdf'. If it is an image, use the visualizer tool instead! DO NOT USE THIS TOOL FOR A WEBPAGE: use the search tool instead!",
            "type": "string",
        },
    }
    output_type = "string"
    md_converter = MarkdownConverter()

    llm = TongGPTEngine()
    def forward(self, file_path, question: Optional[str] = None, initial_exam_mode: Optional[bool] = False) -> str:

        result = self.md_converter.convert(file_path)

        if file_path[-4:] in ['.png', '.jpg']:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if ".zip" in file_path:
            return result.text_content
        
        if not question:
            return result.text_content
        
        if initial_exam_mode:
            messages = [
                {
                    "role": "user",
                    "content": "Here is a file:\n### "
                    + str(result.title)
                    + "\n\n"
                    + result.text_content[:70000],
                },
                {
                    "role": "user",
                    "content": question,
                },
            ]
            return self.llm(messages)
        else:
            messages = [
                {
                    "role": "user",
                    "content": "You will have to write a short caption for this file, then answer this question:"
                    + question,
                },
                {
                    "role": "user",
                    "content": "Here is the complete file:\n### "
                    + str(result.title)
                    + "\n\n"
                    + result.text_content[:70000],
                },
                {
                    "role": "user",
                    "content": "Now answer the question below. Use these three headings: '1. Short answer', '2. Extremely detailed answer', '3. Additional Context on the document and question asked'."
                    + question,
                },
            ]
            return self.llm(messages)