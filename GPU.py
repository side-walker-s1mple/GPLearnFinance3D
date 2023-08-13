from numba import cuda
cuda.select_device(0)
import numpy as np
import time

@cuda.jit
def gpu(a,b,DATA_LENGHTH):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < DATA_LENGHTH:
        b[idx] += a[idx]

if __name__ == '__main__':
    np.random.seed(1)
    DATA_EXP_LENGTH = 20
    DATA_DIMENSION = 2**DATA_EXP_LENGTH
    np_time = 0.0
    nb_time = 0.0
    for i in range(100):
        a = np.random.randn(DATA_DIMENSION).astype(np.float32)
        b = np.random.randn(DATA_DIMENSION).astype(np.float32)
        a_cuda = cuda.to_device(a)
        b_cuda = cuda.to_device(b)
        time0 = time.time()
        gpu[DATA_DIMENSION,4](a_cuda,b_cuda,DATA_DIMENSION)
        time1 = time.time()
        c = b_cuda.copy_to_host()
        time2 = time.time()
        d = np.add(a,b)
        time3 = time.time()
        if i == 0:
            print ('The error between numba and numpy is: ', sum(c-d))
            continue
        np_time += time3 - time2
        nb_time += time1 - time0
    print ('The time cost of numba is: {}s'.format(nb_time))
    print ('The time cost of numpy is: {}s'.format(np_time))