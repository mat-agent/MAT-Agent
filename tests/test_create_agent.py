from tongagent.agents.data_sampling_agent import create_agent

agent = create_agent(
    llm_engine="tonggpt",
    error_tolerance=10,
    task="gaia"
)

print(agent.authorized_imports)
print(agent.additional_authorized_imports)

