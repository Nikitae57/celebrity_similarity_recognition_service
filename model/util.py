import os
import numpy as np
from scipy.spatial import KDTree


NUMPY_FILE_EXTENSION = 'npy'


def get_closest_vector_index(target_vector: np.ndarray, search_tree: KDTree) -> int:
    _, closest_vector_index = search_tree.query(target_vector)
    return closest_vector_index


def build_kd_tree_from_np(vectors: np.ndarray) -> KDTree:
    return KDTree(vectors)


def get_files_from_dir_by_extension(directory: str, extension: str) -> [str]:
    files_in_dir = os.listdir(directory)
    matching_files = filter(lambda dir_name: dir_name.endswith(extension), files_in_dir)

    return matching_files


def get_np_arrays_from_dir(directory: str) -> np.ndarray:
    np_file_names = get_files_from_dir_by_extension(directory, NUMPY_FILE_EXTENSION)
    np_file_paths = [os.path.join(directory, np_file_name) for np_file_name in np_file_names]
    np_arrays = [np.fromfile(np_file_path) for np_file_path in np_file_paths]

    return np.asarray(np_arrays)


def assert_dirs_exist(dirs: [str]):
    for directory in dirs:
        if not os.path.isdir(directory):
            raise ValueError(f'Not a directory: {directory}')


def assert_dirs_contain_same_items_count(dirs: [str]):
    target_items_count = None

    for directory in dirs:

        items_count = len(os.listdir(directory))
        if target_items_count is None:
            target_items_count = items_count
            continue

        if target_items_count != items_count:
            raise ValueError('Directories have unequal items count')


def get_same_basename_files(target_file: str, search_path: str) -> [str]:
    files_in_search_path = os.listdir(search_path)
    target_basename = os.path.basename(target_file)
    same_basename_files = []

    for file in files_in_search_path:
        current_file_basename = os.path.basename(file)
        if current_file_basename == target_basename:
            same_basename_files.append(file)

    return same_basename_files
