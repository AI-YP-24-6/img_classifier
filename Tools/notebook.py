import json
from pathlib import Path


def set_cell_id(notebook_path: str | Path) -> int:
    """
    Обновляет идентификаторы ячеек в файле ноутбука от 1 до N
    :param notebook_path: путь к файлу ipynb
    :return: 1 - файл обновлен
    """
    if isinstance(notebook_path, str):
        notebook_path = Path(notebook_path)

    with open(notebook_path, "rt") as f_in:
        doc = json.load(f_in)
    cnt = 1

    for cell in doc["cells"]:
        if "execution_count" in cell:
            cell["execution_count"] = cnt

            for o in cell.get("outputs", []):
                if "execution_count" in o:
                    o["execution_count"] = cnt

        cnt += 1

    with open(notebook_path, "wt") as f_out:
        json.dump(doc, f_out, indent=1)

    return 1
