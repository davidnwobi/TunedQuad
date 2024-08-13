import importlib

def integrate_from_model(model_name, func, a, b, params=()):
    model = importlib.import_module(model_name)
    return model.integrate(func, a, b, params)
