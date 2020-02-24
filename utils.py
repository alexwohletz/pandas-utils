def update_join(left_df, right_df, on, right_col, fillna=True, validate=True):
    import warnings

    left_df = left_df.copy()
    right_df = right_df.copy()

    if right_col not in right_df.columns:
        raise KeyError("Column not found in right dataframe")

    if on not in set(left_df.columns).intersection(right_df.columns):
        raise KeyError("Key is not shared between dataframes")

    if validate:

        if set(right_df[on]).issubset(set(left_df[on])):
            warnings.warn("Not all keys matching may result in Nans")

        if not set(right_df[on]).intersection(set(left_df[on])):
            raise KeyError("No matching keys between indexes")

    if fillna:

        return (
            right_df.set_index(on)
            .reindex(left_df[on])
            .reset_index()[right_col]
            .fillna(right_df[right_col])
        )

    else:

        return right_df.set_index(on).reindex(left_df[on]).reset_index()[right_col]

