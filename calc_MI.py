import numpy as np

def normalize_feature(data_vec):
    vec_min = np.min(data_vec)
    vec_max = np.max(data_vec)
    midrange = (vec_max + vec_min) / 2
    range_scale = (vec_max - vec_min) / 2
    return (data_vec - midrange) / range_scale

class JointHistogram(object):
    def __init__(self, dataset1, dataset2, partition_size1, partition_size2):
        self._dataset1 = self._normalize(dataset1)
        self._dataset2 = self._normalize(dataset2)
        self._dim1 = self._dataset1.shape[0]
        self._dim2 = self._dataset2.shape[0]
        self._h1 = self._partition_dataset(1, partition_size1)
        self._h2 = self._partition_dataset(2, partition_size2)
        self._j = self._build_joint()
        self._MI_Cache = False

    def _normalize(dataset):
        return np.apply_along_axis(normalize_feature, 0, dataset)

    def _bin_function(partition_size):
        def bin_observation(data_vec):
            pass
        return bin_observation
            

    def _partition_dataset(id, partition_size):
        datasets = {1:self._dataset1, 2:self._dataset2}
        binned_data = np.apply_along_axis(self._bin_function(partition_size), 1, datasets[id])
        histogram = np.bincount(binned_data)
        return histogram

    def mutual_information(self):
        if not self._MI_Cache:
            tmp = self._j * np.log2(self._j / np.outer(self._h1,self._h2))
            self._MI = np.sum(tmp[np.isfinite(tmp)])
            self._MI_Cache = True
        return self._MI
