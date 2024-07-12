def get_linear(device="cpu"):
    assert device == "cpu", "Linear regression can only be run on CPU"
    from sklearn.linear_model import LinearRegression
    return LinearRegression()