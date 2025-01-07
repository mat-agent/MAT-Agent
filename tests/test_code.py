from tongagent.agents.data_sampling_agent import evaluate_python_code_modify


code = '''
import pandas as pd

df = pd.read_csv("data/GAIA/2023/validation/da52d699-e8d2-4dc5-9191-a2199e0b6a9b.xlsx")

print(df.head())
'''

result = evaluate_python_code_modify(
    code,
    authorized_imports=["pandas"]
)
print(result)
