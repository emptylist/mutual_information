import numpy as np

#Good god this is a morass of bad style and efficiency hacks

def normalize_feature(data_vec):
    vec_min = np.min(data_vec)
    vec_max = np.max(data_vec)
    midrange = (vec_max + vec_min) / 2
    range_scale = (vec_max - vec_min) / 2
    return (data_vec - midrange) / range_scale

class JointHistogram(object):
    def __init__(self, dataset1, dataset2, bins_per_dim1, bins_per_dim2):
        self._dataset1 = self._normalize(dataset1)
        self._dataset2 = self._normalize(dataset2)
        self._dim1 = self._dataset1.shape[0]
        self._dim2 = self._dataset2.shape[0]
        self._b1 = self._partition_dataset(1, bins_per_dim1)
        self._b2 = self._partition_dataset(2, bins_per_dim2)
        self._h1, self._h2, self._j = self._build_distributions()
        self._MI_Cache = False

    def _normalize(self, dataset):
        return np.apply_along_axis(normalize_feature, 0, dataset)

    def _bin_function(self, bins_per_dim):
        #generates a function that maps a vector in [-1,1] to a bin id
        #bins are identically sized hypercube segments of the [-1,1]^n hypercube
        #note: the algorithm performs a shift to [0,2] first so no errors occur due
        #to sign changes.
        partition_size = 2.0 / bins_per_dim
        def bin_observation(data_vec):
            data_vec += 1 # shift [-1,1]^n -> [0,2]^n
            bin = np.sum(np.floor_divide(data_vec, partition_size).astype(int) \
                         * (bins_per_dim ** np.arange(data_vec.shape[0])))
            # bin = sum_i ( n^(i-1) * m_i)
            # where m_i is the bin of the i-th dimension, n is the number of bins,
            # and i is the dimension
            return bin
        return bin_observation

    def _partition_dataset(self, id, bins_per_dim):
        datasets = {1:self._dataset1, 2:self._dataset2}
        bin_func = self._bin_function(bins_per_dim)
        binned_data = np.apply_along_axis(bin_func, 1, datasets[id])
        return binned_data

    def _build_distributions(self):
        h1 = np.bincount(self._b1)
        h2 = np.bincount(self._b2)
        #BWAHAHA, Fuck you maintainer/future me!
        j = np.bincount(self._b1 * (h1.shape[0] ** self._b2)).reshape([h1.shape[0], h2.shape[0]])
        #But damn, I can compute the joint distribution in almost linear time (bincount aside)!
        return h1, h2, j

    def mutual_information(self):
        if not self._MI_Cache:
            tmp = self._j * np.log2(self._j / np.outer(self._h1,self._h2))
            self._MI = np.sum(tmp[np.isfinite(tmp)])
            self._MI_Cache = True
        return self._MI

    ##Yeah, I have no idea why I made this an object.
    ##I think a lack of strong typing is making me paranoid about top-level functions in Python.
    ##Maybe I should have just written it in C++ after all.
    ##Then I could actually use loops and label things clearly.
    ##Or Haskell, if I could get fast array ops.
    ##Although I guess a large part of this already derives from the meditate -> fold expression
    ##approach to programming.

## Function wrapper for JH object, since it currently doesn't do anything after it's created
def mutual_information(dataset1, dataset2, bins_per_dim1, bins_per_dim2):
    calculator = JointHistogram(dataset1, dataset2, bins_per_dim1, bins_per_dim2)
    return calculator.mutual_information()
