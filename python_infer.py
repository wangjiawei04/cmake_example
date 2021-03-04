import os
import sys
from paddle_serving_app.reader import *
def add(i, j):
    print("hello,pybind11")
    return i + j

def client2():
    from paddle_serving_client import Client
    import numpy as np
    client = Client()
    client.load_client_config("uci_housing_client/serving_client_conf.prototxt")
    client.connect(["127.0.0.1:9393"])
    data = [0.0137, -0.1136, 0.2553, -0.0692, 0.0582, -0.0727,
            -0.1583, -0.0584, 0.6283, 0.4919, 0.1856, 0.0795, -0.0332]
#    fetch_map = client.predict(feed={"x": np.array(data).reshape(1,13,1)}, fetch=["price"])
    print(fetch_map)
    return 1

def _preprocess():
    import sys
    import numpy as np
    preprocess = Sequential([
        File2Image(), BGR2RGB(), Div(255.0),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], False),
        Resize((640, 640)), Transpose((2, 0, 1))
    ])

    postprocess = RCNNPostprocess("label_list.txt", "output")
    im = preprocess("000000570688.jpg")
    feed={
        "image": np.array(im)[np.newaxis,:],
        "im_shape": np.array(list(im.shape[1:])).astype(np.float32).reshape([1,2]),
        "scale_factor": np.array([1.0, 1.0]).astype(np.float32).reshape([1,2]),
    }
    return feed

def preprocess():
    #for i in range(1000):
    #    _preprocess()
    #return 1
    #feed = _preprocess()

    import numpy as np
    from paddle.inference import Config
    from paddle.inference import create_predictor
    config = Config("serving_server/__model__", "serving_server/__params__")
    config.disable_gpu()
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)
    predictor = create_predictor(config)

    for i in range(10):
        feed = _preprocess()
        # input 0 im shape
        input_names = predictor.get_input_names()
        input_handle = predictor.get_input_handle(input_names[0])
        input_t = feed["im_shape"]
        input_handle.reshape(input_t.shape)
        input_handle.copy_from_cpu(input_t)
        # input 1 image
        input_names = predictor.get_input_names()
        input_handle = predictor.get_input_handle(input_names[1])
        input_t = feed["image"]
        input_handle.reshape(input_t.shape)
        input_handle.copy_from_cpu(input_t)

        # input 2 scale factor
        input_names = predictor.get_input_names()
        input_handle = predictor.get_input_handle(input_names[2])
        input_t = feed["scale_factor"]
        input_handle.reshape(input_t.shape)
        input_handle.copy_from_cpu(input_t)

        predictor.run()
        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu() 

if __name__ == "__main__":
    import time
    a = time.time()
    preprocess()
    b = time.time()
    print("python time cost: {} sec".format(b-a))
