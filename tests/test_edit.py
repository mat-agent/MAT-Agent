import unittest
from tongagent.tools.new_added.image_edit import ImageEditTool
class TestEdit(unittest.TestCase):
    def test_edit(self):
        tool = ImageEditTool()
     
        output_image = tool.forward("turn him into cyborg", "tests/data/draw.jpg")
        print(output_image)
    
if __name__ == "__main__":
    unittest.main()