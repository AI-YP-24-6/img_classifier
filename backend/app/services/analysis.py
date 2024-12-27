import hashlib
import os
from typing import Any

DATASET_DIR = 'temp'


def classes_info() -> dict[str, int]:
    classes = os.listdir(DATASET_DIR)
    class_counts = {cl: len(os.listdir(os.path.join(DATASET_DIR, cl))) for cl in classes}
    return class_counts


def get_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def find_duplicates(folder_path: str) -> list[tuple[str, Any]]:
    hashes = {}
    duplicates = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        hash = get_hash(file_path)

        if hash in hashes:
            duplicates.append((file_path, hashes[hash]))
        else:
            hashes[hash] = file_path
    return duplicates


def duplicates_info() -> dict[str, int]:
    duplicates = dict()
    classes = os.listdir(DATASET_DIR)
    for cls in classes:
        path = os.path.join(DATASET_DIR, cls)
        dup = find_duplicates(path)
        duplicates[cls] = len(dup)
    return duplicates
