import unittest

from tongagent.llm_engine.mini_cpm import MiniCPMEngine
class TestEngine(unittest.TestCase):
    def test_case(self):
        engine = MiniCPMEngine()
        messages = [
            {"role": "user", "content": "You are a helpful assistant. Response in chinese."},
            {"role": "user", "content": "Tell me the model of this aircraft."}
        ]
        output = engine(messages, stop_sequences=[], image_paths = ["tests/data/airplane.jpeg"])
        
        print(output)