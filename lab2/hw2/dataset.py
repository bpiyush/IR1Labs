
import os
import gc
import json
import os.path
import zipfile

import requests
import numpy as np
from tqdm import tqdm
import pickle

FOLDDATA_WRITE_VERSION = 4


def _add_zero_to_vector(vector):
    return np.concatenate([np.zeros(1, dtype=vector.dtype), vector])


def get_dataset(num_folds=1,
                num_relevance_labels=5,
                num_nonzero_feat=519,
                num_unique_feat=501,
                query_normalized=False):

    fold_paths = [get_data_path()]
    return DataSet(
        "ir1-2020",
        fold_paths,
        num_relevance_labels,
        num_unique_feat,
        num_nonzero_feat,
    )


class DataSet(object):

    """
    Class designed to manage meta-data for datasets.
    """

    def __init__(self,
                 name,
                 data_paths,
                 num_rel_labels,
                 num_features,
                 num_nonzero_feat,
                 feature_normalization=True,
                 purge_test_set=True):
        self.name = name
        self.num_rel_labels = num_rel_labels
        self.num_features = num_features
        self.data_paths = data_paths
        self.purge_test_set = purge_test_set
        self._num_nonzero_feat = num_nonzero_feat

    def num_folds(self):
        return len(self.data_paths)

    def get_data_folds(self):
        return [DataFold(self, i, path) for i, path in enumerate(self.data_paths)]


class DataFoldSplit(object):
    def __init__(self, datafold, name, doclist_ranges, feature_matrix, label_vector):
        self.datafold = datafold
        self.name = name
        self.doclist_ranges = doclist_ranges
        self.feature_matrix = feature_matrix
        self.label_vector = label_vector

    def num_queries(self):
        return self.doclist_ranges.shape[0] - 1

    def num_docs(self):
        return self.feature_matrix.shape[0]

    def query_range(self, query_index):
        s_i = self.doclist_ranges[query_index]
        e_i = self.doclist_ranges[query_index+1]
        return s_i, e_i

    def query_size(self, query_index):
        s_i = self.doclist_ranges[query_index]
        e_i = self.doclist_ranges[query_index+1]
        return e_i - s_i

    def query_sizes(self):
        return (self.doclist_ranges[1:] - self.doclist_ranges[:-1])

    def query_labels(self, query_index):
        s_i = self.doclist_ranges[query_index]
        e_i = self.doclist_ranges[query_index+1]
        return self.label_vector[s_i:e_i]

    def query_feat(self, query_index):
        s_i = self.doclist_ranges[query_index]
        e_i = self.doclist_ranges[query_index+1]
        return self.feature_matrix[s_i:e_i, :]

    def doc_feat(self, query_index, doc_index):
        s_i = self.doclist_ranges[query_index]
        e_i = self.doclist_ranges[query_index+1]
        assert s_i + doc_index < self.doclist_ranges[query_index+1]
        return self.feature_matrix[s_i + doc_index, :]

    def doc_str(self, query_index, doc_index):
        doc_feat = self.doc_feat(query_index, doc_index)
        feat_i = np.where(doc_feat)[0]
        doc_str = ''
        for f_i in feat_i:
            doc_str += '%f ' % (doc_feat[f_i])
        return doc_str

    def subsample_by_ids(self, qids):
        feature_matrix = []
        label_vector = []
        doclist_ranges = [0]
        for qid in qids:
            feature_matrix.append(self.query_feat(qid))
            label_vector.append(self.query_labels(qid))
            doclist_ranges.append(self.query_size(qid))

        doclist_ranges = np.cumsum(np.array(doclist_ranges), axis=0)
        feature_matrix = np.concatenate(feature_matrix, axis=0)
        label_vector = np.concatenate(label_vector, axis=0)
        return doclist_ranges, feature_matrix, label_vector

    def random_subsample(self, subsample_size):
        if subsample_size > self.num_queries():
            return DataFoldSplit(self.datafold, self.name + '_*', self.doclist_ranges, self.feature_matrix, self.label_vector, self.data_raw_path)
        qids = np.random.randint(0, self.num_queries(), subsample_size)

        doclist_ranges, feature_matrix, label_vector = self.subsample_by_ids(
            qids)

        return DataFoldSplit(None, self.name + str(qids), doclist_ranges, feature_matrix, label_vector)


class DataFold(object):

    def __init__(self, dataset, fold_num, data_path):
        self.name = dataset.name
        self.num_rel_labels = dataset.num_rel_labels
        self.num_features = dataset.num_features
        self.fold_num = fold_num
        self.data_path = data_path
        self._data_ready = False
        self._num_nonzero_feat = dataset._num_nonzero_feat

    def data_ready(self):
        return self._data_ready

    def clean_data(self):
        del self.train
        del self.validation
        del self.test
        self._data_ready = False
        gc.collect()

    def read_data(self):
        """
        Reads data from a fold folder (letor format).
        """

        pickle_name = 'ltr_data.npz'

        pickle_path = os.path.join(self.data_path,  pickle_name)

        if os.path.isfile(pickle_path):
            loaded_data = np.load(pickle_path, allow_pickle=True)
            if loaded_data['format_version'] == FOLDDATA_WRITE_VERSION:
                train_feature_matrix = loaded_data['train_feature_matrix']
                train_doclist_ranges = loaded_data['train_doclist_ranges']
                train_label_vector = loaded_data['train_label_vector']
                valid_feature_matrix = loaded_data['valid_feature_matrix']
                valid_doclist_ranges = loaded_data['valid_doclist_ranges']
                valid_label_vector = loaded_data['valid_label_vector']
                test_feature_matrix = loaded_data['test_feature_matrix']
                test_doclist_ranges = loaded_data['test_doclist_ranges']
                test_label_vector = loaded_data['test_label_vector']
            del loaded_data

        self.train = DataFoldSplit(self,
                                   'train',
                                   train_doclist_ranges,
                                   train_feature_matrix,
                                   train_label_vector)
        self.validation = DataFoldSplit(self,
                                        'validation',
                                        valid_doclist_ranges,
                                        valid_feature_matrix,
                                        valid_label_vector)
        self.test = DataFoldSplit(self,
                                  'test',
                                  test_doclist_ranges,
                                  test_feature_matrix,
                                  test_label_vector)
        self._data_ready = True


def get_data_path():
    folder_path = os.environ.get("IR1_DATA_PATH")
    if not folder_path:
        folder_path = "./datasets/"
    return folder_path


def download_dataset():
    folder_path = get_data_path()
    os.makedirs(folder_path, exist_ok=True)

    file_location = os.path.join(folder_path, "ltr_data.npz")

    # download file if it doesn't exist
    if not os.path.exists(file_location):

        url = "https://surfdrive.surf.nl/files/index.php/s/5LLTWa5RBm3R8Pa/download"

        with open(file_location, "wb") as handle:
            print(f"Downloading file from {url} to {file_location}")
            response = requests.get(url, stream=True)
            for data in tqdm(response.iter_content(chunk_size=4096)):
                handle.write(data)
            print("Finished downloading file")


def load_production_ranker():
    data_path = get_data_path()
    with open(os.path.join(get_data_path(), "ranks.pkl"), 'rb') as f:
        ranks = pickle.load(f)
    return ranks


def clean_and_sort(ranks, data_fold_split, topk):
    """sorts the items based on the 'ranks' input and removes ranks after 'topk'
    Returns
    -------
    argsorted: the inverse ranking of items shown to the user. feature_matrix[argsorted] would be aligned to the 'clicks'
    doclist_ranges: the new ranges of doclist, consistent with argsorted

    See Also
    --------
    load_clicks_pickle"""
    argsorted = []
    new_doclist_range = [0]

    for qid in range(data_fold_split.doclist_ranges.shape[0] - 1):
        irank = np.argsort(
            ranks[data_fold_split.doclist_ranges[qid]:data_fold_split.doclist_ranges[qid+1]])
        shown_len = min(irank.shape[0], topk)
        argsorted.append(
            data_fold_split.doclist_ranges[qid] + irank[:shown_len])
        new_doclist_range.append(shown_len)

    _argsorted = np.concatenate(argsorted, axis=0)
    _doclist_range = np.cumsum(np.array(new_doclist_range), axis=0)

    return _argsorted, _doclist_range


if __name__ == "__main__":
    download_dataset()
    dataset = get_dataset()
    data = dataset.get_data_folds()[0]
    data.read_data()

    print(f"Number of features: {data.num_features}")
    # print some statistics
    for split in ["train", "validation", "test"]:
        print(f"Split: {split}")
        split = getattr(data, split)
        print(f"\tNumber of queries {split.num_queries()}")
        print(f"\tNumber of docs {split.num_docs()}")
