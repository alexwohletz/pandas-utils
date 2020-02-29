import warnings
import itertools
from pandas_schema import Column, Schema
import pandas as pd
import numpy as np
from pandas_schema.validation import (
    CanConvertValidation,
    InListValidation,
    InRangeValidation,
    IsDistinctValidation,
    LeadingWhitespaceValidation,
    MatchesPatternValidation,
    TrailingWhitespaceValidation,
)


class PandasJoin:
    @staticmethod
    def check_join_cols(df1, df2, on):
        schema = Schema(
            [
                Column(
                    col,
                    [
                        LeadingWhitespaceValidation(),
                        TrailingWhitespaceValidation(),
                        IsDistinctValidation(),
                    ],
                )
                for col in on
            ]
        )
        results = [schema.validate(df) for df in [df1[on], df2[on]]]

        if len(results) > 0:
            errors = [error.__str__() for error in itertools.chain(*results)]
            warnings.warn(f"The Following Problems exist in the index {errors}")

    @staticmethod
    def update_join(
        df1,
        df2,
        update_col,
        source_col,
        target_index,
        on,
        how="inner",
        overwrite=False,
        validate_indexes=False,
    ):

        df1 = df1.copy()
        df2 = df2.copy()

        if validate_indexes:
            PandasJoin.check_join_cols(df1, df2, on)

        if not update_col in df1.columns:
            print(f"New column assignment detected, creating: '{update_col}''")
            df1[update_col] = None

        if not df1.index.name:
            warnings.warn(
                f"Index not set on df1, attempting to set index to {target_index}"
            )
            df1 = df1.set_index(target_index)

        if df1.index.name != target_index:
            warnings.warn(
                f"Index of update column does not match that of source column, attempting to set to {target_index}"
            )
            df1 = df1.reset_index()
            df1 = df1.set_index(target_index)

        if source_col in on or source_col == target_index:
            print("Index column is being used as a key, creating temporary column")
            df2["temp_col"] = df2.reset_index()[source_col]
            temp = (
                df1.merge(df2, on=on, how=how)
                .set_index(target_index)["temp_col"]
                .rename(update_col)
            )
        else:
            temp = (
                df1.merge(df2, on=on, how=how)
                .set_index(target_index)[source_col]
                .rename(update_col)
            )

        if len(temp) == 0:
            raise ValueError("Join failed, check indexes and try again")
        else:
            print(
                f"Assigning the following values to '{update_col}' in df1 \n {temp[df1.loc[temp.index,update_col].isnull()]}"
            )

        try:
            df1.update(other=temp, join="left", overwrite=overwrite)
            return df1
        except ValueError:
            print(
                f"Cannot assign {update_col} when duplicates {df2[on].duplicated().sum()} exist in join column(s) {on}, you might try setting on more than one column that forms a unique key"
            )


if __name__ == "__main__":

    df1 = pd.DataFrame(
        np.array(
            [["a", 5, 9, None], 
            ["b", 14, 61, 10], 
            ["c", 4, 9, None], 
            ["d", 3, 1, 30],]
        ),
        columns=["key", "key2", "attr12", "attr13"],
    )

    df2 = pd.DataFrame(
        np.array(
            [["a", 5, 19], 
            ["b", 14, 16], 
            ["c", 4, 9], 
            ["d", 3, 1], 
            ["c", 3, 19]]
        ),
        columns=["key", "key2", "attr22"],
    )

    test_df = PandasJoin.update_join(
        df1=df1,
        df2=df2,
        update_col="new",
        target_index="key",
        source_col="key2",
        on=["key", "key2"],
        how="inner",
        overwrite=False,
        validate_indexes=False,
    )
    print(test_df)

