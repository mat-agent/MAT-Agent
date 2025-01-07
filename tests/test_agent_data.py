import unittest

from tongagent.agents.data_sampling_agent import create_agent

class TestAgent(unittest.TestCase):
    def test_case(self):
        agent = create_agent()
        agent.set_image_paths(["tests/data/watch.png"])
        result = agent.run("What is the price of the gadget shown in the picture according to the online store's current listing\nAttachement: tests/data/watch.png")
        print(result)
        agent.save_trajectory()
        
    def test_case2(self):
        agent = create_agent()
        agent.set_image_paths([]) # everytime you call this
        result = agent.run("Generate a photo-realistic street view")
        print(result)
        agent.save_trajectory()
        
    def test_case3(self):
        agent = create_agent()
        agent.set_image_paths(["tests/data/draw.jpg"]) # everytime you call this
        result = agent.run("Turn him into cyborg\nAttachement: tests/data/draw.jpg")
        print(result)
        agent.save_trajectory()
    
    def test_case4(self):
        agent = create_agent()
        agent.set_image_paths(["tests/data/20240920-230749.png"]) # everytime you call this
        # Original: Can you find the average cost of the items listed in this receipt?
        result = agent.run("Can you find the average cost of the items listed in this receipt?\nAttachement: tests/data/20240920-230749.png")
        print(result)
        agent.save_trajectory()
        
    def test_case5(self):
        agent = create_agent()
        agent.set_image_paths(["tests/data/20240920-230749.png"]) # everytime you call this
        result = agent.run("Can you find the average value of the items listed in this plot?\nAttachement: tests/data/20240920-230749.png")
        print(result)
        agent.save_trajectory()
        
    def test_case6(self):
        from tongagent.tools.visual_qa import VisualQAGPT4Tool
        
        tool = VisualQAGPT4Tool()
        result = tool.forward(
            question="""Do you think if the image is relevant to the question? 
Question: Can you find the average cost of the items listed in this receipt?
Format your answer as:
Thought: <the reasoning process about if the image content is relevant to the question>
Answer: <yes or no>""",
            image_path="tests/data/20240920-230749.png"
        )
        print(result)
    
    def test_case7(self):
        agent = create_agent(error_tolerance=3)
        agent.set_image_paths(["tests/data/burger.jpeg"]) # every time you call this
        result = agent.run("I want to know which dish in the menu has the highest caloric value. Please highlight that item in the image of the menu.\nAttachment: tests/data/burger.jpeg; ")
        print(result)
        
        agent.save_trajectory()
        print(agent.logs[-1])
if __name__ == "__main__":
    unittest.main()