from tongagent.tools.tool_box import TextInspectorTool

data_path = "data/GAIA/2023/validation/da52d699-e8d2-4dc5-9191-a2199e0b6a9b.xlsx"

tool = TextInspectorTool()

question = "What is the list of books read in 2022 along with their reading speeds?"
result = tool.forward(file_path=data_path, question=question)
print(result)