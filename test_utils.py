import pytest
import pandas as pd
from pandas.testing import assert_series_equal
from pandas.testing import assert_frame_equal
import numpy as np
from utils import SQLUpdate


@pytest.fixture(scope="module")
def df1():
    return pd.DataFrame(
        np.array(
            [["a", 5, 9, None], ["b", 14, 61, 10], ["c", 4, 9, None], ["d", 3, 1, 30]]
        ),
        columns=["key", "key2", "attr11", "attr12"],
    ).astype({"key": "str", "key2": "int"})


@pytest.fixture(scope="module")
def df2():
    return pd.DataFrame(
        np.array([["a", 5, 19], ["b", 14, 16], ["c", 4, 9], ["d", 3, 1], ["c", 3, 19]]),
        columns=["key", "key2", "attr21"],
    ).astype({"key": "str", "key2": "int"})


@pytest.fixture(scope="module")
def df2_nulls():
    return pd.DataFrame(
        np.array(
            [["a", None, 19], ["b", 14, 16], ["c", 4, 9], ["d", 3, 1], ["c", None, 19]]
        ),
        columns=["key", "key2", "attr21"],
    ).astype({"key": "str", "key2": "int"})


def test_new_col_no_ind(df1, df2):
    # Case where new column, no index specified, should raise warning.
    series = pd.Series(
        ["19", "16", "9", "1"],
        name="new",
        index=pd.Index(["a", "b", "c", "d"], name="key"),
    )
    with pytest.warns(UserWarning):
        test = SQLUpdate.update_join(
            df1=df1,
            df2=df2,
            update_col="new",
            target_index="key",
            source_col="attr21",
            on=["key", "key2"],
            how="inner",
            overwrite=False,
            validate_indexes=False,
        )

    assert_series_equal(test["new"], series)


def test_existing_col_no_ind(df1, df2):
    # Case where exiting column, no index specified, should raise warning.
    series = pd.Series(
        ["a", 10, "c", 30],
        name="attr12",
        index=pd.Index(["a", "b", "c", "d"], name="key"),
    )
    with pytest.warns(UserWarning):
        test = SQLUpdate.update_join(
            df1=df1,
            df2=df2,
            update_col="attr12",
            target_index="key",
            source_col="key",
            on=["key", "key2"],
            how="inner",
            overwrite=False,
            validate_indexes=False,
        )
    assert_series_equal(test["attr12"], series)

