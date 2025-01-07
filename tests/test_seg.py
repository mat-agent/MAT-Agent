import unittest

from tongagent.tools.new_added.seg import SegTool

class TestSeg(unittest.TestCase):
    
    def test_seg(self):
        tool = SegTool()
        result = tool.forward("tests/data/cars.png")        
        print(result)
        
    def test_seg_2(self):
        tool = SegTool()
        prompt = [[200.68, 451.94, 354.71, 545.96], [192.86, 359.01, 953.82, 738.95], [908.56, 197.47, 1555.35, 993.67]]
        result = tool.forward("tests/data/cars.png", prompt=prompt)        
        print(result)

if __name__ == "__main__":
    unittest.main()