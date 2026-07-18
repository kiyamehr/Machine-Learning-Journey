from sklearn.preprocessing import StandardScaler

def scale_data(*cols):

    """
    Scale the numeric columns of one or more pandas DataFrames using
    a single StandardScaler.

    The scaler is fitted on the first DataFrame provided and then used
    to transform the remaining DataFrames. This prevents data leakage
    by ensuring that statistics (mean and standard deviation) are
    learned only from the training data.

    Only columns with dtype ``int64`` or ``float64`` are scaled.
    Boolean and categorical columns are left unchanged.

    Parameters
    ----------
    *cols : pandas.DataFrame
        One or more DataFrames to scale. The first DataFrame is used
        to fit the scaler, while all subsequent DataFrames are only
        transformed.

    Returns
    -------
    None
        The input DataFrames are modified in place.
    """
    
    
    scaler = StandardScaler()

    for i, col in enumerate(cols):
        numeric_cols = col.select_dtypes(['int64', 'float64']).columns
        
        if i == 0:
            col[numeric_cols] = scaler.fit_transform(col[numeric_cols])
        else:
            col[numeric_cols] = scaler.transform(col[numeric_cols])
