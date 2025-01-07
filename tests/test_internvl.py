import unittest
from tongagent.llm_engine.internvl2 import InternVL2Engine
from transformers.agents.llm_engine import MessageRole, HfApiEngine, get_clean_message_list

class TestInternVL(unittest.TestCase):
    def test_internvl(self):
        engine = InternVL2Engine()
        messages = [
            {"role": MessageRole.SYSTEM, "content": "You are a helpful assistant."},
            {"role": MessageRole.USER, "content": "What is the capital of France?"},
        ]
        answer = engine(messages)
        print(answer)
    
    def test_internvl_with_image(self):
        engine = InternVL2Engine()
        messages = [
            {"role": MessageRole.SYSTEM, "content": "Respond in chinese"},
            {"role": MessageRole.USER, "content": "What airplane in the image?"},
        ]
        answer = engine(messages, image_paths=["tests/data/airplane.jpeg"])
        print(answer)

if __name__ == "__main__":
    unittest.main()