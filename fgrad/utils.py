import struct
import numpy as np


def layer_init_uniform(*x):
    ret = np.random.uniform(-1., 1., size=x).astype(np.float32) / np.sqrt(np.prod(x))
    return ret

def fetch_mnist():

    def fetch(url):
        import requests, gzip, os, hashlib, numpy
        fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
        if os.path.isfile(fp):
            with open(fp, 'rb') as f:
                data = f.read()
        else:
            with open(fp, 'wb') as f:
                response = requests.get(url)
                if response.status_code != 200:
                    raise RuntimeError(f"Failed to download {url}, status code: {response.status_code}")
                data = response.content
                f.write(data)
        return numpy.frombuffer(gzip.decompress(data), dtype=np.uint8)

    urls = [
        ("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz", 
         "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
         "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
         "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"),
        ("https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz",
         "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz", 
         "https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz",
         "https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz"),
    ]
    
    for train_images_url, train_labels_url, test_images_url, test_labels_url in urls:
        try:
            file_data = fetch(train_images_url)
            _, num_images, rows, cols = struct.unpack(">IIII", file_data[:0x10])
            X_train = file_data[0x10:].reshape(-1, rows, cols)
            
            file_data = fetch(train_labels_url)
            _, num_labels = struct.unpack(">II", file_data[:0x8])
            Y_train = file_data[0x8:]
            
            # 测试数据图像
            file_data = fetch(test_images_url)
            _, num_images, rows, cols = struct.unpack(">IIII", file_data[:0x10])
            X_test = file_data[0x10:].reshape(-1, rows, cols)
            
            # 测试数据标签
            file_data = fetch(test_labels_url)
            _, num_labels = struct.unpack(">II", file_data[:0x8])
            Y_test = file_data[0x8:]
            
            return X_train, Y_train, X_test, Y_test
            
        except Exception as e:
            print(f"Failed to download from mirror, trying next one... Error: {e}")
            continue