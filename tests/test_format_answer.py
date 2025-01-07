import unittest

from tongagent.prompt import FORMAT_ANSWER_PROMPT_GAIA

from langchain.prompts import ChatPromptTemplate

class TestFormat(unittest.TestCase):
    def test_case(self):
        template = ChatPromptTemplate.from_template(FORMAT_ANSWER_PROMPT_GAIA)
        prompt = template.invoke({
            "question": "hi",
            "answer": "hi"
        })
        print(prompt)
        print(prompt.to_string())
        print(prompt.to_messages()[0].content)
if __name__ == "__main__":
    unittest.main()