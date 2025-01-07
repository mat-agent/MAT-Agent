import argparse
from tongagent.agents.general_agent import create_agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", 
        required=True, 
        help="Instructions that you want agent to execute.")
    args = parser.parse_args()
    agent = create_agent()
    result = agent.run(args.prompt)
    print("Agent Response:", result)

if __name__ == "__main__":
    main()