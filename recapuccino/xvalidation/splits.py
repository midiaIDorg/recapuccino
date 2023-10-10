import numpy as np
import numpy.typing as npt
import typing
import pandas as pd
from math import inf


class Model(typing.Protocol):
    def fit(X, y, *args, **kwargs):
        ...

    def predict(X):
        ...


def create_group_assignments_at_random(
    number_of_elements_to_assign_groups_to: int,
    group_assignment_probabilities: npt.NDArray = np.array([0.8, 0.1, 0.1]),
    replace: bool = True,
) -> npt.NDArray:
    return np.random.choice(
        a=len(group_assignment_probabilities),
        size=number_of_elements_to_assign_groups_to,
        p=group_assignment_probabilities,
        replace=replace,
    )


# AA BB CC
# (X[AABB], Y[AABB]), (X[CC], Y[CC])
# A[0] .. A[99]


def iterate_train_evaluation_datasets(
    X: pd.DataFrame,
    Y: pd.Series,
    assignments: npt.NDArray,
) -> typing.Iterator[
    tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, npt.NDArray]
]:
    for chunk_No in range(max(assignments) + 1):
        evaluation_set_mask = assignments == chunk_No
        yield (
            X.loc[~evaluation_set_mask],
            Y.loc[~evaluation_set_mask],
            X.loc[evaluation_set_mask],
            Y.loc[evaluation_set_mask],
            evaluation_set_mask,
        )


# assignments = create_group_assignments_at_random(100)


def find_optimal_models_using_xvalidation(
    X: pd.DataFrame,
    Y: pd.Series,
    chunks_cnt: int,
    ModelFactory_hyperparameters_tuples: list[
        tuple[typing.Callable[..., Model], dict[str, typing.Any]]
    ],
    scoring: typing.Callable = lambda x, y: np.abs(x - y).median(),
    _replace=True,
    **kwargs
) -> list[tuple[Model, float]]:
    """
    Find k optimal models for each x-validation data chunk.

    We implement here simply a procedure described by Granholm et al.
    https://pubs.acs.org/doi/full/10.1021/acs.jproteome.0c01010

    Scoring: the smaller the better.
    """
    assert len(X) == len(Y), "X has dim N0*D and Y has dim N1 and N0 != N1."
    assert chunks_cnt <= len(X), "There are groups than data rows. WTF."

    group_assignment_probabilities = np.ones(chunks_cnt, dtype=float) / chunks_cnt
    group_assignment_probabilities /= sum(group_assignment_probabilities)

    assignments_of_data_points_to_individual_chunks = (
        create_group_assignments_at_random(
            number_of_elements_to_assign_groups_to=len(X),
            group_assignment_probabilities=group_assignment_probabilities,
            replace=_replace,
        )
    )
    N = len(ModelFactory_hyperparameters_tuples)
    split_probs_in_chunk = np.ones(N, dtype=float) / N

    optimal_predictions = np.zeros(len(Y), dtype=Y.values.dtype)
    best_model_and_score_per_chunk = []
    for (
        train_X,
        train_Y,
        eval_X,
        eval_Y,
        chunk_mask,
    ) in iterate_train_evaluation_datasets(
        X, Y, assignments_of_data_points_to_individual_chunks
    ):
        assignments_of_data_points_within_a_chunk = create_group_assignments_at_random(
            number_of_elements_to_assign_groups_to=len(train_X),
            group_assignment_probabilities=split_probs_in_chunk,
            replace=_replace,
        )

        best_model = None
        best_score = inf

        for (ModelFactory, hyperparameters), (
            hyper_train_X,
            hyper_train_Y,
            hyper_eval_X,
            hyper_eval_Y,
            hyper_chunk_mask,
        ) in zip(
            ModelFactory_hyperparameters_tuples,
            iterate_train_evaluation_datasets(
                train_X,
                train_Y,
                assignments_of_data_points_within_a_chunk,
            ),
        ):
            model = ModelFactory(**hyperparameters)
            model.fit(hyper_train_X, hyper_train_Y)
            hyper_pred_Y = model.predict(hyper_eval_X)
            score = scoring(hyper_pred_Y, hyper_eval_Y)
            if score < best_score:
                best_model = model
                best_score = score

        pred_Y = best_model.predict(eval_X)
        best_model_score = scoring(pred_Y, eval_Y)
        best_model_and_score_per_chunk.append((best_model, best_model_score))

        optimal_predictions[chunk_mask] = pred_Y

    for _, score in best_model_and_score_per_chunk:
        assert score < inf, "We fucked up: the scores were all infinite."

    return best_model_and_score_per_chunk, optimal_predictions


# from sklearn.linear_model import LinearRegression
# ModelFactory = LinearRegression
