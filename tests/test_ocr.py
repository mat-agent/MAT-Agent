import unittest
from tongagent.tools.new_added.ocr import OCRTool

class TestOCR(unittest.TestCase):
    def test_case(self):
        tool =  OCRTool()
        texts = tool.forward("tests/data/254.jpg", debug=True)
        print(texts)
        
if __name__ == "__main__":
    unittest.main()