import numpy as np # type: ignore

def gen_data():
    np.random.seed(12)

    x = np.linspace(-6,6,100)

    y = (
        np.cos(x)
        + x*0.15
        + np.random.normal(0,0.2,100)
    )

    X = x.reshape(-1,1)
    
    return X, y

def show_first_five(arrays):
    for i in range (len(arrays)):
        print(arrays[i][:5])
