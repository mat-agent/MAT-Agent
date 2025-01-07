from typing import Optional

from tongagent.tools.web_surfer import (
    SearchInformationTool,
    NavigationalSearchTool,
    VisitTool,
    PageUpTool,
    PageDownTool,
    FinderTool,
    FindNextTool,
    ArchiveSearchTool,
)
from tongagent.tools.visual_qa import VisualQAGPT4Tool
from tongagent.tools.new_added.face_det import FaceDetTool
from tongagent.tools.new_added.object_loc import ObjectLOCTool
from tongagent.tools.new_added.ocr import OCRTool
from tongagent.tools.new_added.seg import SegTool
from tongagent.tools.new_added.image_edit import ImageEditTool
from tongagent.tools.mdconvert import MarkdownConverter
from tongagent.tools.web_surfer import WebQATool, DownloadTool
from tongagent.llm_engine.gpt import TongGPTEngine

from transformers.agents.agents import Tool
from transformers.agents import load_tool
import os
class TextInspectorTool(Tool):
    name = "inspect_file_as_text"
    description = """You cannot load files yourself: instead call this tool to read a file as markdown text and ask questions about it. This tool handles the following file extensions: [".html", ".htm", ".xlsx", ".pptx", ".wav", ".mp3", ".flac", ".pdf", ".docx"], and all other types of text files. IT DOES NOT HANDLE IMAGES."""

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

    llm_engine = TongGPTEngine()
    
    def forward(self, file_path: str, question: Optional[str] = None) -> str:
        initial_exam_mode = False
        if not os.path.exists(file_path):
            return "File not found: " + file_path
        result = self.md_converter.convert(file_path)

        if file_path[-4:] in ['.png', '.jpg']:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if result is None:
            return "Get empty content from file: " + file_path
        
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
            return self.llm_engine(messages)
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
            return self.llm_engine(messages)

def get_general_tool_box():
    WEB_TOOLBOX = [
        SearchInformationTool(),
        VisitTool(),
        WebQATool()
        # PageUpTool(),
        # PageDownTool(),
        # FinderTool(),
        # FindNextTool(),
        # ArchiveSearchTool(),
    ]

    ti_tool = TextInspectorTool()
    image_generation_tool = load_tool("m-ric/text-to-image")

    MODEL_TOOLBOX = [
        # SearchTool(),
        VisualQAGPT4Tool(),  # VisualQATool(),
        FaceDetTool(),
        ObjectLOCTool(),
        ti_tool,
        image_generation_tool,
        OCRTool(),
        SegTool(),
        ImageEditTool()
    ]

    TOOL_BOX = WEB_TOOLBOX + MODEL_TOOLBOX
    return TOOL_BOX

def get_tool_box_gaia():
    TOOL_BOX_GAIA = [
        VisualQAGPT4Tool(),
        TextInspectorTool()
    ]
    return TOOL_BOX_GAIA


from tongagent.tools.new_added.image_generation import ImageGenerationTool

def get_visual_model_tool_box():
    MODEL_TOOLBOX = [
        VisualQAGPT4Tool(),
        FaceDetTool(),
        ObjectLOCTool(),
        TextInspectorTool(),
        ImageGenerationTool(),
        # OCRTool(),
        SegTool(),
        ImageEditTool()
    ]
    return MODEL_TOOLBOX

def get_visual_model_tool_box_for_gaia():
    MODEL_TOOLBOX = [
        VisualQAGPT4Tool(),
        #FaceDetTool(),
        #ObjectLOCTool(),
        TextInspectorTool(),
        #ImageGenerationTool(),
        # OCRTool(),
        #SegTool(),
        #ImageEditTool()
    ]
    return MODEL_TOOLBOX