import tomllib
from recapuccino.importing import dynamically_import_foo


# path = "tmp/configs/template_for_xvalidation.toml"
def parse_model_config(path: str):
    with open(path, "rb") as f:
        config = tomllib.load(f)
    config["ModelFactory_hyperparameters_tuples"] = []
    for model_config in config["model"]:
        ModelFactory = dynamically_import_foo(model_config["name"])
        hyperparams = model_config["hyperparameter"]
        config["ModelFactory_hyperparameters_tuples"].append(
            (ModelFactory, hyperparams)
        )
    del config["model"]
    return config


# find_optimal_models_using_xvalidation(
#     X, Y, **parse_model_config("tmp/configs/template_for_xvalidation.toml")
# )
