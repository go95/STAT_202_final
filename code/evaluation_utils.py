"""Utilities for evaluating Kaggle submissions."""

import os
from typing import Dict, Text

import numpy as np
import pandas as pd


class Evaluator(object):
  """Evaluator of Kaggle submissions."""
  def __init__(self, solution_file: Text="../test_solutions.csv", missing_value: float=-1.0):
    """Constructor."""
    if not os.path.isfile(solution_file):
      raise ValueError(f"Unable to find {solution_file}.")

    self.solutions = pd.read_csv(solution_file, index_col="id")
    self.solutions["day"] = self.solutions.index.map(
        lambda x: int(x.split('-')[1])
    )
    self.solutions["period"] = self.solutions.day.apply(
        lambda x: "Period 1" if x <= 3 else "Period 2"
    )
    self.solutions.loc[self.solutions.open == -1.0, "open"] = np.nan
      
  def _evaluate_submission(self, submission):
    submission.rename(columns={'open': 'predicted_open'}, inplace=True)
    joint_data = pd.concat([self.solutions, submission], axis=1)
    # Imputes values where missing.
    joint_data.loc[joint_data.predicted_open.isnull(), "predicted_open"] = 0.
    joint_data["error"] = (joint_data.predicted_open - joint_data.open)**2
    # n.b. I didn't divide by 10 in the eval equation in the instructions.
    daily_avg_error = 10*joint_data.groupby(["period", "day"]).error.apply(
        np.nanmean
    )
    period_errors = daily_avg_error.groupby("period").mean()
    return period_errors.to_dict()

  def evaluate_submission(self, submission_file: Text):
    """Evaluates submitted entry.

    Args:
      submission_file: location of submission file. Must follow expected
        format for Kaggle.

    Returns:
      A dict mapping the period to the evaluated error.
    """
    if not os.path.isfile(submission_file):
      raise ValueError(f"Unable to find {submission_file}.")

    submission = pd.read_csv(submission_file, index_col="id")
    return _evaluate_submission(submission)
