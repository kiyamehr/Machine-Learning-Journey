from sklearn.preprocessing import StandardScaler

def scale_data(*cols):

    scaler = StandardScaler()

    for i, col in enumerate(cols):
        numeric_cols = col.select_dtypes(['int64', 'float64']).columns
        
        if i == 0:
            col[numeric_cols] = scaler.fit_transform(col[numeric_cols])
        else:
            col[numeric_cols] = scaler.transform(col[numeric_cols])
