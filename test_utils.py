import pytest
import pandas as pd
from pandas.testing import assert_series_equal
from pandas.testing import assert_frame_equal
import numpy as np
from utils import update_join


@pytest.fixture(scope="module")
def data1():
    return pd.DataFrame(
        np.array(
            [
                ["a", "one", 9, 8],
                ["b", "two", 61, 10],
                ["c", "three", 9, 225],
                ["d", "four", 1, 30],
            ]
        ),
        columns=["key", "attr11", "attr12", "attr13"],
    )


@pytest.fixture(scope="module")
def data2():
    return pd.DataFrame(
        {
            "key": ["a", "b", "c", "d", "a", "b", "c", "d", "c", "a"],
            "attr21": [x * 2 for x in range(10)],
            "attr22": [x ** 2 for x in range(10)],
            "attr23": [0, 1, None, 3, 4, 5, 6, None, 9, None],
        }
    )


@pytest.fixture(scope="module")
def data_no_key():
    return pd.DataFrame(
        {
            "dnk1": [x * 2 for x in range(10)],
            "dnk2": [x ** 2 for x in range(10)],
            "dnk3": [0, 1, None, 3, 4, 5, 6, None, 9, None],
        }
    )


@pytest.fixture(scope="module")
def bad_data():
    return pd.DataFrame(
        np.array([["1", 15, 49], ["2", 4, 36], ["3", 14, 9]]),
        columns=["key", "bd1", "bd2"],
    )


def test_update_join_with_warn(data1, data2):

    test = pd.Series(
        ["one", "two", "three", "four", "one", "two", "three", "four", "three", "one"]
    )

    # Check warning raised for key subset
    with pytest.warns(UserWarning, match="Not all keys matching may result in Nans"):
        col = update_join(
            left_df=data2,
            right_df=data1,
            on="key",
            right_col="attr11",
            fillna=True,
            validate=True,
        )
    # Make sure we get expected results
    assert_series_equal(test, col, check_names=False)


def test_update_join_no_match_key(data1, data_no_key):

    with pytest.raises(KeyError, match="Key is not shared between dataframes"):
        update_join(
            left_df=data1,
            right_df=data_no_key,
            on="key",
            right_col="dnk1",
            fillna=True,
            validate=True,
        )

