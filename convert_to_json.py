from kronrod import kronrod_points_dict
import numpy as np
import json
from quad_tuning import load_tuned_quad_h5

model_name = "cylinder_accurate"
quad = load_tuned_quad_h5(model_name)

param_order = list(quad.reg_params.keys())
reg_params = {param: quad.reg_params[param].tolist() for param in param_order}
tuned_mat = quad.tuned_mat.flatten().tolist()
dims = quad.dims.tolist()
print(param_order)
print(reg_params)
print(tuned_mat)
print(dims)

json_dict = {
    model_name: {
        "params_order": param_order,
        "reg_params": reg_params,
        "tuned_mat": tuned_mat
        "dims": dims,
    }
}

with open("tuned_quad.json", "w") as f:
    json.dump(json_dict, f)
# print(quad)
# json_dict = {}
# val = set(quad.tuned_mat.flatten().tolist())
#
# print(val)
# print(len(val))
#
# # for key, value in kronrod_points_dict.items():
#     if key not in val:
#         continue
#
#     json_dict[key] = {"x": value[0].tolist(), "w": value[1].tolist(), "wg": value[2].tolist()}
#
#
# with open("kronrod.json", "w") as f:
#     json.dump(json_dict, f)
