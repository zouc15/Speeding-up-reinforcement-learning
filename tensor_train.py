import tensorflow as tf
from functools import reduce

class TensorTrain:
    def __init__(self, tt_cores, shape=None, tt_ranks=None, convert_to_tensor=True):
        tt_cores = list(tt_cores)
        if convert_to_tensor:
            with tf.name_scope("TensorTrain", tt_cores):
                for i in range(len(tt_cores)):
                    name = "core%d" % i
                    tt_cores[i] = tf.convert_to_tensor(tt_cores[i], name=name)
        self._tt_cores = tuple(tt_cores)
        self._raw_shape = []
        for i in range(len(shape)):
            self._raw_shape.append(tf.TensorShape(shape[i]))
        self._tt_ranks = tf.TensorShape(tt_ranks)
        
    def ndims(self):
        return len(self._tt_cores)

    def tt_cores(self):
        return self._tt_cores

    def get_raw_shape(self):
        return self._raw_shape

    def get_tt_ranks(self):
        return self._tt_ranks

    def get_shape(self):
        raw_shape=self.get_raw_shape()
        prod_f=lambda arr: reduce(lambda x,y: x*y,arr)
        m=prod_f(raw_shape[0].as_list())
        n=prod_f(raw_shape[1].as_list())
        return tf.TensorShape((m,n))
