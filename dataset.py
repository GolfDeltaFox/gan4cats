import numpy as np
from os import listdir
from os.path import isfile, join
import scipy

class Dataset():
    def __init__(self, data, one_hot=False):
        self._len = len(data)
        self._cur = 0
        if one_hot:
            data = np.array(data).reshape([data.shape[0], -1])
        self.data = np.array(data)

    def next_batch(self, batch_size):
        if self._cur + batch_size > self._len:
            res = np.append(self.data[self._cur:],self.data[:batch_size-(self._len-self._cur)], axis=0)
            self._cur = self._len-self._cur
        else:
            res = self.data[self._cur:self._cur+batch_size]
            self._cur = self._cur+batch_size
        return np.array(res, dtype='f')

    @classmethod
    def get_images(cls, path, grey_scale=False):
        images =[]
        for file in listdir(path):
            image_path = join(path, file)
            if isfile(image_path):
                image = scipy.ndimage.imread(image_path, flatten=grey_scale)
                images.append(image/256)
        return cls(images)
