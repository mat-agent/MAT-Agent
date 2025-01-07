import unittest
from tongagent.agents.gaia_agent import create_agent_gaia, create_agent_simple_gaia
from tongagent.agents.general_agent import create_agent
class TestGAIA(unittest.TestCase):
    def test_case(self):
        agent = create_agent_gaia()
        
        question = """As a comma separated list with no whitespace, using the provided image provide all the fractions that use / as the fraction line and the answers to the sample problems. Order the list by the order in which the fractions appear. Attachement: data/GAIA/2023/validation/9318445f-fe6a-4e1b-acbf-c68228c9906a.png"""
        print(agent.run(question))
    
    def test_case_2(self):
        from tongagent.tools.new_added.ocr import OCRTool
        tool = OCRTool()
        result = tool.forward("data/GAIA/2023/validation/9318445f-fe6a-4e1b-acbf-c68228c9906a.png", debug=True)
        print(result)
        
    def test_case_3(self):
        question = """How many slides in this PowerPoint presentation mention crustaceans? Attachement: data/GAIA/2023/validation/a3fbeb63-0e8c-4a11-bff6-0e3b484c3e9c.pptx"""
        agent = create_agent_gaia()
        print(agent.run(question))
    
    def test_case_4(self):
        question = """Of the cities within the United States where U.S. presidents were born, which two are the farthest apart from the westernmost to the easternmost going east, giving the city names only? Give them to me in alphabetical order, in a comma-separated list"""
        agent = create_agent_gaia()
        print(agent.run(question))
        
    def test_case_5(self):
        question = """Here's a fun riddle that I think you'll enjoy.

You have been selected to play the final round of the hit new game show "Pick That Ping-Pong". In this round, you will be competing for a large cash prize. Your job will be to pick one of several different numbered ping-pong balls, and then the game will commence. The host describes how the game works.

A device consisting of a winding clear ramp and a series of pistons controls the outcome of the game. The ramp feeds balls onto a platform. The platform has room for three ping-pong balls at a time. The three balls on the platform are each aligned with one of three pistons. At each stage of the game, one of the three pistons will randomly fire, ejecting the ball it strikes. If the piston ejects the ball in the first position on the platform the balls in the second and third position on the platform each advance one space, and the next ball on the ramp advances to the third position. If the piston ejects the ball in the second position, the ball in the first position is released and rolls away, the ball in the third position advances two spaces to occupy the first position, and the next two balls on the ramp advance to occupy the second and third positions on the platform. If the piston ejects the ball in the third position, the ball in the first position is released and rolls away, the ball in the second position advances one space to occupy the first position, and the next two balls on the ramp advance to occupy the second and third positions on the platform.

The ramp begins with 100 numbered ping-pong balls, arranged in ascending order from 1 to 100. The host activates the machine and the first three balls, numbered 1, 2, and 3, advance to the platform. Before the random firing of the pistons begins, you are asked which of the 100 balls you would like to pick. If your pick is ejected by one of the pistons, you win the grand prize, $10,000.

Which ball should you choose to maximize your odds of winning the big prize? Please provide your answer as the number of the ball selected."""
        agent = create_agent_gaia()
        print(agent.run(question))
    
    def test_case_6(self):
        question = """How many times was a Twitter/X post cited as a reference on the english Wikipedia pages for each day of August in the last June 2023 versions of the pages?"""
        agent = create_agent_gaia()
        print(agent.run(question))
    
    def test_case_7(self):
        question = """How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia."""
        agent = create_agent_simple_gaia()
        # agent.set_plan(
        #     """1. I did a search for Mercedes Sosa\n2. I went to the Wikipedia page for her\n3. I scrolled down to \"Studio albums\"\n4. I counted the ones between 2000 and 2009"""
        # )
        print(agent.run(question))
    
    def test_case_8(self):
        question = """What was the complete title of the book in which two James Beard Award winners recommended the restaurant where Ali Khan enjoyed a New Mexican staple in his cost-conscious TV show that started in 2015? Write the numbers in plain text if there are some in the title."""
        agent = create_agent()
        result = agent.run(question)
        print(result)
        
if __name__ == "__main__":
    unittest.main()