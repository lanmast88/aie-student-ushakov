"""
Утилиты для работы с батчами туров.

Команды:
    # Разбить на батчи для отправки в Claude
    python batch_utils.py split --input tours_seed.json --batch-size 25

    # Смёржить описания обратно после получения от Claude
    python batch_utils.py merge --input tours_seed.json --descriptions-dir descriptions/
"""

import json
import argparse
from pathlib import Path


def split_into_batches(input_path: str, batch_size: int = 25) -> None:
    """
    Разбивает tours_seed.json на батчи.
    Для каждого батча создаёт компактный файл — только поля нужные для описания.
    """
    with open(input_path, encoding="utf-8") as f:
        tours = json.load(f)

    batches_dir = Path("batches")
    batches_dir.mkdir(exist_ok=True)

    total = len(tours)
    n_batches = (total + batch_size - 1) // batch_size

    print(f"Всего туров: {total}")
    print(f"Размер батча: {batch_size}")
    print(f"Батчей: {n_batches}")
    print()

    for i in range(n_batches):
        batch = tours[i * batch_size : (i + 1) * batch_size]

        # Для Claude оставляем только поля нужные для написания описания
        compact = []
        for t in batch:
            compact.append({
                "id": t["id"],
                "city_name": t["city_name"],
                "country": t["country"],
                "hotel_name": t["hotel_name"],
                "hotel_stars": t["hotel_stars"],
                "duration_nights": t["duration_nights"],
                "meal_type": t["meal_type"],
                "category": t["category"],
                "price": t["price"],
            })

        out_path = batches_dir / f"batch_{i+1:02d}_of_{n_batches:02d}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(compact, f, ensure_ascii=False, indent=2)

        print(f"  ✓ {out_path}  ({len(batch)} туров)")

    print(f"\nГотово. Папка: batches/")
    print(f"\nДальше:")
    print(f"  1. Открой каждый файл batch_XX_of_{n_batches:02d}.json")
    print(f"  2. Скинь содержимое в Claude со словами:")
    print(f"     'напиши описания для этих туров'")
    print(f"  3. Сохрани ответ в descriptions/batch_XX.json")
    print(f"  4. После всех батчей: python batch_utils.py merge")


def merge_descriptions(input_path: str, descriptions_dir: str) -> None:
    """
    Мёржит описания от Claude обратно в исходный JSON.

    Ожидает файлы вида descriptions/batch_01.json со структурой:
    [{"id": "...", "description": "..."}, ...]

    Или просто список строк в том же порядке что батч.
    """
    with open(input_path, encoding="utf-8") as f:
        tours = json.load(f)

    # Строим индекс id → тур
    tour_index = {t["id"]: t for t in tours}

    desc_dir = Path(descriptions_dir)
    desc_files = sorted(desc_dir.glob("*.json"))

    if not desc_files:
        print(f"Нет файлов в {descriptions_dir}/")
        print("Создай файлы вида: descriptions/batch_01.json")
        return

    filled = 0
    for desc_file in desc_files:
        with open(desc_file, encoding="utf-8") as f:
            data = json.load(f)

        # Поддерживаем два формата ответа от Claude:
        # Формат 1: [{"id": "...", "description": "..."}]
        # Формат 2: {"descriptions": {"id": "description"}}
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "id" in item and "description" in item:
                    if item["id"] in tour_index:
                        tour_index[item["id"]]["description"] = item["description"]
                        filled += 1

        elif isinstance(data, dict):
            # {"id": "description", ...}
            for tour_id, description in data.items():
                if tour_id in tour_index:
                    tour_index[tour_id]["description"] = description
                    filled += 1

        print(f"  ✓ {desc_file.name}")

    # Проверяем что все описания заполнены
    missing = [t["id"] for t in tours if not t.get("description")]
    if missing:
        print(f"\n⚠ Без описания: {len(missing)} туров")
        print("  Они сохранятся с description=null")
    else:
        print(f"\n✅ Все {filled} описаний заполнены")

    # Сохраняем финальный файл
    output_path = Path(input_path).stem + "_final.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tours, f, ensure_ascii=False, indent=2)

    print(f"\nСохранено → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    # split
    sp = sub.add_parser("split", help="Разбить на батчи")
    sp.add_argument("--input",      default="tours_seed.json")
    sp.add_argument("--batch-size", type=int, default=25)

    # merge
    mp = sub.add_parser("merge", help="Смёржить описания")
    mp.add_argument("--input",            default="tours_seed.json")
    mp.add_argument("--descriptions-dir", default="descriptions")

    args = parser.parse_args()

    if args.cmd == "split":
        split_into_batches(args.input, args.batch_size)
    elif args.cmd == "merge":
        merge_descriptions(args.input, args.descriptions_dir)
    else:
        parser.print_help()
