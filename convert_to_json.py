from kronrod import kronrod_points_dict
import numpy as np
import json

json_dict = {}
rng = np.random.default_rng()
val = set(rng.choice(list(kronrod_points_dict.keys()), 50))

for key, value in kronrod_points_dict.items():
    if key not in val:
        continue
    temp = {}
    temp["x"] = value[0].tolist()
    temp["w"] = value[1].tolist()
    temp["wg"] = value[2].tolist()

    json_dict[key] = temp

    
with open("kronrod.json", "w") as f:
    json.dump(json_dict, f)