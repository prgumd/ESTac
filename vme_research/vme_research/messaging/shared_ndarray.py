from multiprocessing import (Queue,
                             resource_tracker,
                             shared_memory,
                             Value)
import platform
from typing import Any
import numpy as np
import time
from typing import Union, Tuple, Dict
from queue import Empty

class SharedNDArray:
    def __init__(self, array: np.ndarray, id: Union[int, None], shm: shared_memory.SharedMemory, shm_indices: Tuple[int,int]) -> None:
        self.x = array
        self.id = id
        self.shm = shm
        self.shm_indices = shm_indices
        self.shape = None # type: ignore
        self.dtype = None # type: ignore

    def __getstate__(self)-> Dict[str, Any]:
        return {
            'id': self.id,
            'shm': self.shm,
            'shm_indices': self.shm_indices,
            'dtype': self.x.dtype,
            'shape': self.x.shape
        }
         

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.x = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf[self.shm_indices[0]:self.shm_indices[1]]) # type: ignore

# Used when a SharedNDArray is expected but can't be provided
class NonSharedNDArray:
    def __init__(self, array: np.ndarray) -> None:
        self.x = array

class SharedNDArrayPool:
    def __init__(self, shape, dtype: np.dtype, num_arrays, close_queue=None): 
        self.shape = shape
        self.dtype = dtype
        self.num_arrays = num_arrays
        self.close_queue = close_queue

        temp = np.ndarray(self.shape, dtype=self.dtype)
        self.expected_nbytes = temp.nbytes

        self.shm_nbytes = self.expected_nbytes*self.num_arrays
        self.shm = shared_memory.SharedMemory(size=self.shm_nbytes, create=True)
        if self.close_queue: self.free = list(range(self.num_arrays))
        else: self.free = None

    def get(self, array_id=None, N=1):
        # Attempt to empty the close queue
        if self.close_queue:
            while not self.close_queue.empty():
                try:
                    close_array_id = self.close_queue.get()
                    self.free.append(close_array_id)
                except Empty:
                    break

        if array_id is None:
            assert len(self.free) > 0
            assert N == 1
            array_id = self.free.pop(0)

        low_i  = self.expected_nbytes *  array_id
        high_i = self.expected_nbytes * (array_id + N)

        if N == 1:
            shape = self.shape
        else:
            shape = (N, *self.shape)

        if high_i < self.shm_nbytes + 1:
            shm_array = SharedNDArray(
                            np.ndarray(shape, dtype=self.dtype, buffer=self.shm.buf[low_i:high_i]),
                            array_id,
                            self.shm,
                            (low_i, high_i))
            return shm_array
        # N wraps around, we will have to make a copy since NumPy does
        # not support views of noncontiguous memory
        else:
            shape_lo = ((self.shm_nbytes - low_i) // self.expected_nbytes, *self.shape)
            shm_array_lo = SharedNDArray(
                            np.ndarray(shape_lo, dtype=self.dtype, buffer=self.shm.buf[low_i:self.shm_nbytes]),
                            array_id,
                            self.shm,
                            (low_i, self.shm_nbytes))
            wrapped_high_i = high_i - self.shm_nbytes + 1
            shape_hi = (wrapped_high_i // self.expected_nbytes, *self.shape)
            shm_array_hi = SharedNDArray(
                            np.ndarray(shape_hi, dtype=self.dtype, buffer=self.shm.buf[0:wrapped_high_i]),
                            array_id,
                            self.shm,
                            (0, wrapped_high_i))

            array = np.concatenate((shm_array_lo.x, shm_array_hi.x), axis=0)
            return NonSharedNDArray(array)

    def close(self, id):
        if self.close_queue:
            self.close_queue.put(id)
        else:
            self.free.append(id)

    def cleanup(self):
        if self.shm is not None:
            self.shm.close()
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass

    def __del__(self):
        self.cleanup()

class SharedNDArrayPubSub:
    def __init__(self, max_q_size: int = 10, zero_copy: bool = False):
        self.max_q_size = max_q_size
        self.queue = Queue(max_q_size)
        if platform.system() == 'Darwin':
            self.seperate_queue_size = True
        else:
            self.seperate_queue_size = False
        if self.seperate_queue_size: self.queue_size = Value('i', 0)
        self.zero_copy = zero_copy
        self.ndarray_pools = {}
        self.close_queue = Queue(self.max_q_size)

    def pub(self, data, use_shm=None):
        if not self.zero_copy:
            queue_list = []
            for i, d in enumerate(data):
                if use_shm and use_shm[i] and type(d) == np.ndarray:
                    if i not in self.ndarray_pools:
                        self.ndarray_pools[i] = SharedNDArrayPool(d.shape, d.dtype, self.max_q_size, close_queue=self.close_queue)

                    shm_array = self.ndarray_pools[i].get()
                    shm_array.x[:] = d
                    queue_list.append(shm_array)
                else:
                    queue_list.append(d)
        else:
            queue_list = data

        t = time.time()
        if self.seperate_queue_size:
            with self.queue_size.get_lock():
                self.queue_size.value += 1
        self.queue.put((t, queue_list))

    def get(self, timeout=None, verbose=False):
        t00 = time.time()
        if self.seperate_queue_size:
            with self.queue_size.get_lock():
                self.queue_size.value -= 1
        t, data = self.queue.get(timeout=timeout)
        t05 = time.time()

        if not self.zero_copy:
            data_out = []
            for d in data:
                if type(d) == SharedNDArray:
                    data_out.append(np.copy(d.x))
                    self.close_queue.put(d.id)
                else:
                    data_out.append(d)
        else:
            data_out = data
        t10 = time.time()

        t_pubsub = t10 - t
        tget = t05 - t00
        t_copy = t10 - t05

        if verbose:
            print('t_pubsub', t_pubsub, 'tget', tget, 't_copy', t_copy)

        return data_out

    def size(self):
        if self.seperate_queue_size:
            return self.queue_size.value
        else:
            return self.queue.qsize()

    def cleanup(self):
        for ndarray_pool in self.ndarray_pools.values():
            ndarray_pool.cleanup()

class SharedNDArrayPipe:
    def __init__(self, ndarray_pools=None, sample_data=None, max_messages=10, check_overflow=True):
        self.pub_message_i = 0 # integers in python3 have unlimited precision
        self.get_message_i = 0
        self.pool_i = Value('Q', 0, lock=False) # unsigned long long
        self.pool_get_i = Value('Q', 0, lock=False)
        self.max_messages = max_messages
        self.sample_data = sample_data
        self.check_overflow = check_overflow

        if ndarray_pools is not None and sample_data is None:
            self.ndarray_pools = ndarray_pools
        elif ndarray_pools is None and sample_data is not None:
            self.ndarray_pools = [SharedNDArrayPool(d.shape, d.dtype, self.max_messages) for d in sample_data]
        else:
            raise ValueError('Invalid combination of ndarray_pools and sample_message')

    def pub(self, data):
        assert len(data) == len(self.ndarray_pools)
        pool_index = self.pub_message_i % self.max_messages
        N = 1
        for d, p, sd in zip(data, self.ndarray_pools, self.sample_data):
            assert type(d) == np.ndarray
            if d.ndim == sd.ndim:
                shm_array = p.get(pool_index)
                shm_array.x[:] = d
            else:
                if N != 1: assert N == d.shape[0]
                else: N = d.shape[0] if len(d.shape) > 0 else 1
                if pool_index + N > p.num_arrays:
                    N_lo = p.num_arrays - pool_index
                    N_hi = N - N_lo
                    shm_array_lo = p.get(pool_index, N=N_lo)
                    shm_array_hi = p.get(0, N=N_hi)
                    # assert type(shm_array_lo) == SharedNDArray
                    # assert type(shm_array_hi) == SharedNDArray
                    shm_array_lo.x[:] = d[0:N_lo]
                    shm_array_hi.x[:] = d[N_lo:N]
                else:
                    shm_array = p.get(pool_index, N=N)
                    # assert type(shm_array) == SharedNDArray
                    shm_array.x[:] = d

        self.pub_message_i += N
        self.pool_i.value = self.pub_message_i
        # Detect if receiver has fallen a full buffer behind
        # error out without handling it
        if self.check_overflow:
            assert (self.pub_message_i - self.pool_get_i.value) < self.max_messages

    # this is a stateful call (we incremenet get_message_i and will not return the same data again)
    def get(self, N=1):
        pub_message_i = self.pool_i.value
        if N < 0 or self.get_message_i + N > pub_message_i:
            N = pub_message_i - self.get_message_i

        if N > 0:
            pool_index = self.get_message_i % self.max_messages
            data = [p.get(pool_index, N=N).x for p in self.ndarray_pools]
            self.get_message_i += N
            self.pool_get_i.value = self.get_message_i
        else:
            data = None

        return data

    # this is a stateless call, it will always return the previous min(N, available_messages)
    def get_latest_N(self, N=1):
        pub_message_i = self.pool_i.value # latest index
        # get the number of available messages (e.g. if the pool is full, it will be the max_messages)
        N_avail = min(self.max_messages, pub_message_i)
        if N < 0 or N > N_avail:
            # get the earliest possible index
            low_message_i = pub_message_i - N_avail
            N = N_avail # get the latest available messages
        else:
            # get the desired earliest index
            low_message_i = pub_message_i - N # get the latest N messages

        if N > 0:
            # get the index into the pool
            # (e.g. if i is greater than the size of the pool, it will wrap around)
            # (so if the pool is 10, and i is 11, the index will be 1)
            pool_index = low_message_i % self.max_messages
            data = [p.get(pool_index, N=N).x for p in self.ndarray_pools]

            # these are commented out so that this function is not stateful
            # (e.g. it will not increment the get_message_i 
            #       which would prevent preiously return data from being returned again)
            # self.get_message_i += N
            # self.pool_get_i.value = self.get_message_i
        else:
            data = None

        return data

    def size(self):
        return self.pool_i.value - self.pool_get_i.value

    def cleanup(self):
        for ndarray_pool in self.ndarray_pools:
            ndarray_pool.cleanup()
