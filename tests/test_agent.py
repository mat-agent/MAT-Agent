import unittest

from tongagent.agents.general_agent import create_agent
class TestAgent(unittest.TestCase):
    def test_ocr(self):
        agent = create_agent()

        result = agent.run("Can you try to extract text from the image path? Image path: tests/data/254.jpg")
        print(result)
    
    def test_sg(self):
        agent = create_agent()

        result = agent.run("Can you try to extract mask from the image path to a pickle file?  Image path: tests/data/cars.png. Show me the file name you generated is good.")
        print(result)
        
    def test_edit(self):
        agent = create_agent()

        result = agent.run("Can you edit the image to turn him into cyborg? Image path: tests/data/draw.jpg.")
        print(result)
        
    def test_loc(self):
        agent = create_agent()

        result = agent.run("Can you try to first detect cars shown in the images and then extract masks for cars?  Image path: tests/data/cars.png.")
        print(result)
    
    def test_web_search(self):
        agent = create_agent()
        question = """If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, how many thousand hours would it take him to run the distance between the Earth and the Moon its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. Round your result to the nearest 1000 hours and do not use any comma separators if necessary."""
        
        result = agent.run(question)
        print(result)
    
    def test_web_search2(self):
        agent = create_agent()
        question = """How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia."""
        
        result = agent.run(question)
        print(result)
    
    def test_gaia_case1(self):
        agent = create_agent()
        question = """A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes, where each axis has a label word at both ends. Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016?"""
        result = agent.run(question)
        print(result)
        # answer = egalitarian
    
    def test_gaia_case2(self):
        question = "In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?"
        agent = create_agent()
        result = agent.run(question)
        print(result)
if __name__ == "__main__":
    unittest.main()