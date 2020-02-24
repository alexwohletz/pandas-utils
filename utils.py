def update_join(left_df, right_df, on, right_col, fillna=True, validate=True):
    """Simple function to mimic a SQL update set col1 = col2 from df1 join df2 on df1.key == df2.key. Does not modify dataframes in place.
    
    Arguments:
        left_df {dataframe} -- Left Dataframe.
        right_df {dataframe} -- Right Dataframe.
        on {str} -- Key or shared column to 'join' both dataframes on.
        right_col {str} -- Col2, the column from which we are updating column 1.
    
    Keyword Arguments:
        fillna {bool} -- Whether or not to fill gaps in Col1. (default: {True})
        validate {bool} -- Checks to make sure the indexes share values and warn if something is off. (default: {True})
    
    Raises:
        KeyError: No shared key.
        KeyError: No shared indexes in that key.
        KeyError: Right column does not exist in right dataframe.
    
    Returns:
        pd.Series -- A pandas Series with the length of left dataframe updated with the values of the right column based on the key.
    """
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

