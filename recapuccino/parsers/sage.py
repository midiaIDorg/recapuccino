import numpy as np
import pandas as pd


def _unfold(vals, split_token: str = ";"):
    res = []
    for x in vals:
        for y in x.split(split_token):
            if y != "":
                res.append(y)
    return res


def unfold(vals, split_token: str = ";"):
    return np.array(_unfold(vals, split_token)).astype(float)


def parse_fragments(sage, fragment_types: str):
    def _iter():
        for fragment_type in fragment_types:
            fr = pd.DataFrame(
                {
                    "exp_mz": unfold(sage[f"{fragment_type}_exp_mz"]),
                    "calc_mz": unfold(sage[f"{fragment_type}_calc_mz"]),
                }
            )
            fr["fragment_type"] = fragment_type
            yield fr

    return pd.concat(_iter(), ignore_index=True)
