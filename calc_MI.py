import numpy as np

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
        self._h1 = self._partition_dataset(1, bins_per_dim1)
        self._h2 = self._partition_dataset(2, bins_per_dim2)
        self._j = self._build_joint()
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
            # bin_id = sum_i ( n^(i-1) * m_i)
            # where m_i is the bin of the i-th dimension, n is the number of bins,
            # and i is the dimension
            return bin
        return bin_observation

    def _partition_dataset(self, id, bins_per_dim):
        datasets = {1:self._dataset1, 2:self._dataset2}
        bin_func = self._bin_function(bins_per_dim)
        binned_data = np.apply_along_axis(bin_func, 1, datasets[id])
        histogram = np.bincount(binned_data)
        return histogram

    def mutual_information(self):
        if not self._MI_Cache:
            tmp = self._j * np.log2(self._j / np.outer(self._h1,self._h2))
            self._MI = np.sum(tmp[np.isfinite(tmp)])
            self._MI_Cache = True
        return self._MI
