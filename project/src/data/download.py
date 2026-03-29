"""
Загрузка датасета Goodreads-books с Kaggle.

Использование:
    python -m src.data.download
"""

import shutil
from pathlib import Path

import kagglehub

KAGGLE_DATASET = "jealousleopard/goodreadsbooks"
DEST_DIR = Path("data")


def download() -> Path:
    print(f"Скачиваем датасет {KAGGLE_DATASET} ...")
    src = Path(kagglehub.dataset_download(KAGGLE_DATASET))

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for file in src.iterdir():
        shutil.copy(file, DEST_DIR / file.name)
        print(f"  {file.name} → {DEST_DIR / file.name}")

    print("Готово.")
    return DEST_DIR


if __name__ == "__main__":
    download()
