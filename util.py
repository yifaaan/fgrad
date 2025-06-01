import struct
import numpy as np

def fetch_mnist():

    def fetch(url):
        import requests, gzip, os, hashlib, numpy
        fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
        if os.path.isfile(fp):
            with open(fp, 'rb') as f:
                data = f.read()
        else:
            with open(fp, 'wb') as f:
                data = requests.get(url).content
                f.write(data)
        return numpy.frombuffer(gzip.decompress(data), dtype=np.uint8)

    # 16 字节的文件头, 前 4 字节是 magic number, 后面依次为图像数量, 行数, 列数 (都用大端表示)
    file_data = fetch('https://storage.googleapis.com/kaggle-data-sets/102285/242592/upload/train-images-idx3-ubyte.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250531%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250531T051117Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=23ffd289ca39356b793279e5b8de4b0ae86c48870508f96a4482cd24eb477909581ef45417ecfa4fb89fa259c69cf3d996b8d34c1120203488bb642c9b895e7868ba028397bc4c82afebcce6d52c92f0dc37e64778e50459e4ff684fe13a5c9847410c064e3b8d98bd77a5bbc81313ff42033935462ce8dc3863191d57b4a084a1bd1314d06d1c524d11a84f7de6fdce2e35b0fc6dbec043bd3d1d512fe8823122da6f55d57e884e786aca405a1094f1680e5b25d68025ba13ae646352040fc680a2f32be7d28ce75b4045460c100e814715289635da69c04cb66bf3d1095b672cf2ed1dca465dbff8a1855d3961c6218261ff3e50308c67f2f62daa82cd745b')
    _, num_images, rows, cols = struct.unpack(">IIII", file_data[:0x10])
    X_train = file_data[0x10:].reshape(-1, rows, cols)
    file_data = fetch('https://storage.googleapis.com/kaggle-data-sets/102285/242592/upload/train-labels-idx1-ubyte.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250531%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250531T051144Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=3488764f5aa676f7591f2b2c14fb917b5038a79e09dca7fecdaf95e2fe8635908783047939cebaf98d769044033dccee5b227a7ad4b7cb2b03c6d9cf42cc0c5abacf01313ddbc19967b523c103077de72ac5f502d2a511bc7ab5c467b1d71f99e885a6fa0568610084ebea24a8ceeff5a090724beb9748335837f6b0761a9c9f53821421f1848c21dd4424efe3589f632b60a0d9fca72639de5aa68b19ab59f77d88d7c01785fb2a8844fb26afe1efcde46685374cb300edd1738242910d460adb495d4018830872dcfd41f756efcca0907998a644da8340ff6a434903124012d33052e18f6b12a7add7b4374979c6641aea13378dceba323e612c20902e5a45')
    _, num_labels = struct.unpack(">II", file_data[:0x8])
    Y_train = file_data[0x8:]
    file_data = fetch('https://storage.googleapis.com/kaggle-data-sets/102285/242592/upload/t10k-images-idx3-ubyte.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250531%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250531T045831Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=39f58cb3a2e8034659d5a4534eddd606e970c78e2063b61976ebea0efc5afa9edee8db6844fabca8a0461aa24e9121060cbe889be083b58ff3c4c1635af3b35bb4e9ab47b273fb8764b481f3640689a34d279e810653fadd94299f54a5a72846f0a5d7d904543243c21781c17a2ef3b7ac6e3bf9c882377aa94677dc50283b96fbccd721614a108dd83656563b28dccc3280c3e034bcab219a56a474fdf66a9a1f0e9662b105cb5c74a1c1f67aaa06ea7ff50297a53551575b742cb3f4c682a2614d534f3eb690130d12547b2ce752971ebe618f8ef8949f43d07b22ce4385c427ce6852f9b281582fdbefc2769c3b6d45110c633d1940cc017240be668c255b')
    _, num_images, rows, cols = struct.unpack(">IIII", file_data[:0x10])
    X_test = file_data[0x10:].reshape(-1, rows, cols)
    file_data = fetch('https://storage.googleapis.com/kaggle-data-sets/102285/242592/upload/t10k-labels-idx1-ubyte.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250531%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250531T045908Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=a28f7a0eb3d41cf0aeb47d9755eefa259b5c35a6e1e52be4913873cdfe451b5695d0ae1bf80b52e49923decc8443b224f63492c97a6dca211620882421d18f9d32baa1e3dbc01dc899a4260c582e8be09fb66e2fd05335ed54759c04d61baea0997f06cf7775b954c49e5df7fc7a7ff114e4c1554aef0390ba34c907acc73a47032cf4cf5d360c8c84e22321f06f21d003b48c789a72590914557b7a3cd6f0842005acb2db857a3115cb7e74529416e1926ed367d75eb632f4b59ae6b001f06fc97b5e772bb285c667928f697314c96c9a3c61b84aea4ae3eac7e091f6768a11cbd1dd68a0e577c655a7059e37354c3e589512cc10f4d951574980bae38b0f44')
    _, num_labels = struct.unpack(">II", file_data[:0x8])
    Y_test = file_data[0x8:]
    return X_train, Y_train, X_test, Y_test
    # print(X_test.shape)
    # print(Y_test.shape)
    # print(X_train.shape)
    # print(Y_train.shape)



