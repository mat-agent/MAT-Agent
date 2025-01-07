import wandb
import time
wandb.init(project="occupy", name="occupy")
print("Get GPU!")
wandb.alert(title="Get GPU!", text="Get GPU!")
wandb.finish()
