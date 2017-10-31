import numpy as np
from os import listdir, walk
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
        for dir_path, _, files in walk(path):
            for afile in files:
                image_path = join(dir_path, afile)
                image = scipy.ndimage.imread(image_path, flatten=grey_scale)
                images.append(image/256)
        return cls(images)
