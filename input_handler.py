import pandas as pd
import os, sys
from dataclasses import dataclass
import argparse

# Constants
DEFAULT_DATA_PATH = "data/Sample Data Set.xlsx"
DEFAULT_SEP = ";"
DEFAULT_NO_ID_COLS = 2
DEFAULT_LAST_SAVED = 0
DEFAULT_BATCH_SIZE = 50


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers and stuff.")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="The path to the file containing the data.",
    )
    parser.add_argument(
        "--n",
        type=int,
        help="The number of columns to be used for creating unique id.",
    )
    parser.add_argument(
        "--sep",
        "-s",
        type=str,
        help="The separator used in the csv file.",
    )
    parser.add_argument(
        "--last_saved",
        "-l",
        type=int,
        default=DEFAULT_LAST_SAVED,
        help="The number of unique ids saved in the last run.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="The number of unique ids to be processed before saving the results.",
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="%(prog)s (version 0.1)",
    )

    args = parser.parse_args()

    return args


@dataclass
class UserInputs:
    dataset_url: str = DEFAULT_DATA_PATH
    number_of_id_cols: int = DEFAULT_NO_ID_COLS
    last_saved: int = DEFAULT_LAST_SAVED
    batch_size: int = DEFAULT_BATCH_SIZE
    sep: str = DEFAULT_SEP


def get_user_inputs() -> UserInputs:
    args = parse_args()

    user_inputs = UserInputs(
        dataset_url=args.file,
        number_of_id_cols=args.n,
        last_saved=args.last_saved,
        batch_size=args.batch_size,
        sep=args.sep,
    )

    if not user_inputs.dataset_url:
        user_inputs.dataset_url = input(
            f"Enter the path to the dataset (default: '{DEFAULT_DATA_PATH}'): "
        ).strip()

        if not user_inputs.dataset_url:
            user_inputs.dataset_url = "data/Sample Data Set.xlsx"
        if not os.path.exists(user_inputs.dataset_url):
            sys.exit(f"File not found at {user_inputs.dataset_url}. Exiting...")
        if not user_inputs.dataset_url.endswith((".csv", ".xlsx")):
            sys.exit(
                f"{user_inputs.dataset_url} is not a supported file format (only .csv and .xlsx are supported). Exiting..."
            )

    if user_inputs.dataset_url.endswith(".csv") and not user_inputs.sep:
        user_inputs.sep = input(
            f"Enter the separator used in the dataset (default: '{DEFAULT_SEP}'): "
        ).strip()
        if not user_inputs.sep:
            user_inputs.sep = ";"

    if not user_inputs.number_of_id_cols:
        user_inputs.number_of_id_cols = input(
            f"Enter the number columns not in the time series (default: {DEFAULT_NO_ID_COLS}): "
        )
        if not user_inputs.number_of_id_cols:
            user_inputs.number_of_id_cols = DEFAULT_NO_ID_COLS
        else:
            user_inputs.number_of_id_cols = int(user_inputs.number_of_id_cols)

    return user_inputs
