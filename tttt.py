import os
import sys
def add(i, j):
    print("hello,pybind11")
    return i + j

def client():
    from paddle_serving_client import Client
    import numpy as np
    client = Client()
    client.load_client_config("uci_housing_client/serving_client_conf.prototxt")
    client.connect(["127.0.0.1:9393"])
    data = [0.0137, -0.1136, 0.2553, -0.0692, 0.0582, -0.0727,
            -0.1583, -0.0584, 0.6283, 0.4919, 0.1856, 0.0795, -0.0332]
    fetch_map = client.predict(feed={"x": np.array(data).reshape(1,13,1)}, fetch=["price"])
    print(fetch_map)
    return 1
