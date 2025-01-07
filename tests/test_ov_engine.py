import unittest
from tongagent.llm_engine.llava import LLaVAEngine
from transformers.agents.llm_engine import MessageRole, HfApiEngine, get_clean_message_list

class TestOvEngine(unittest.TestCase):
    def test_llava1(self):
        engine = LLaVAEngine("Lin-Chen/open-llava-next-llama3-8b")
        prompt = [
            {"role": MessageRole.USER, "content": "How are you doing?"},
        ]
        answer = engine(prompt, image_path=[])
        print(answer)
    
    def test_llava2(self):
        engine = LLaVAEngine("Lin-Chen/open-llava-next-llama3-8b")
        image_path = "tests/data/airplane.jpeg"
        prompt = [
            {"role": MessageRole.USER, "content": "What is the image?"},
        ]
        answer = engine(prompt, image_paths=[image_path])
        print(answer)


if __name__ == "__main__":
    unittest.main()