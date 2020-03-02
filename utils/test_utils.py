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


#############################################Smaller dataframe assigned from larger dataframe#######################################


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


def test_existing_col_w_ind(df1, df2):
    # Case where exiting column, with index specified, should not raise warning.
    series = pd.Series(
        ["a", 10, "c", 30],
        name="attr12",
        index=pd.Index(["a", "b", "c", "d"], name="key"),
    )
    with pytest.warns(UserWarning):
        test = SQLUpdate.update_join(
            df1=df1.set_index("key"),
            df2=df2.set_index("key"),
            update_col="attr12",
            target_index="key",
            source_col="key",
            on=["key", "key2"],
            how="inner",
            overwrite=False,
            validate_indexes=False,
        )
    assert_series_equal(test["attr12"], series)


def test_existing_col_overwrite_no_ind(df1, df2):
    # Case where exiting column, no index specified, should raise warning.
    series = pd.Series(
        ["a", "b", "c", "d"],
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
            overwrite=True,
            validate_indexes=False,
        )
    assert_series_equal(test["attr12"], series)


#############################################Larger dataframe assigned from smaller dataframe#######################################


def test_bts_new_col_no_ind(df1, df2):
    # Case where new column, no index specified, should raise warning.
    series = pd.Series(
        [None, 10, None, 30, None],
        name="new",
        index=pd.Index(["a", "b", "c", "d", "c"], name="key"),
    )
    with pytest.warns(UserWarning):
        test = SQLUpdate.update_join(
            df1=df2,
            df2=df1,
            update_col="new",
            target_index="key",
            source_col="attr12",
            on=["key", "key2"],
            how="inner",
            overwrite=False,
            validate_indexes=False,
        )
    # Nan behavior means datatypes will differ
    assert_series_equal(test["new"], series, check_dtype=False)
    # Make sure the length of dataframe is unchanged
    assert len(test) == len(df2)


def test_bts_new_col_w_ind(df1, df2):
    # Case where new column, no index specified, should raise warning.
    series = pd.Series(
        [None, 10, None, 30, None],
        name="new",
        index=pd.Index(["a", "b", "c", "d", "c"], name="key"),
    )
    with pytest.warns(UserWarning):
        test = SQLUpdate.update_join(
            df1=df2.set_index("key"),
            df2=df1.set_index("key"),
            update_col="new",
            target_index="key",
            source_col="attr12",
            on=["key", "key2"],
            how="inner",
            overwrite=False,
            validate_indexes=False,
        )
    # Nan behavior means datatypes will differ
    assert_series_equal(test["new"], series, check_dtype=False)
    # Make sure the length of dataframe is unchanged
    assert len(test) == len(df2)


def test_bts_new_col_w_2_ind(df1, df2):
    # Case where new column, no index specified, should raise warning.
    series = pd.Series(
        ["a", "b", "c", "d", np.NaN], #note if just using one index, will populate where one index matches
        name="new",
        index= pd.MultiIndex.from_tuples(
            [("a", 5), ("b", 14), ("c", 4), ("d", 3), ("c", 3)], names=["key", "key2"]
        )
    )
    test = SQLUpdate.update_join(
        df1=df2,
        df2=df1,
        update_col="new",
        target_index=["key", "key2"],
        source_col="key",
        on=["key", "key2"],
        how="inner",
        overwrite=False,
        validate_indexes=False,
    )
    assert_series_equal(test["new"], series)
