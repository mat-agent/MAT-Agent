import json
with open("data/gaia_1107_train.json", "r") as f:
    data = json.load(f)

import random
with open("data/gaia_1107_train.json", "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
    

with open("data/gaia_1107_train_subset.json", "w") as f:
    json.dump(random.sample(data, 500), f, indent=4, ensure_ascii=False)


