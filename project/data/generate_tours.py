"""
Генератор синтетических данных туров для recommendation-service.

Принципы реалистичности:
- Цены коррелируют с городом, отелем, сезоном и типом питания
- Длительность зависит от дальности направления
- Описания строятся из шаблонов с реальными деталями
- Сезонность влияет на даты и цены
- ~15% туров недоступны (status=inactive) — как в реальной жизни

Запуск:
    python generate_tours.py                  # → data/tours_seed.json
    python generate_tours.py --count 200      # больше туров
    python generate_tours.py --seed 99        # другой random seed
"""

import json
import uuid
import random
import argparse
from datetime import date, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Справочники (бизнес-логика заложена в структуру данных)
# ---------------------------------------------------------------------------

CITIES = {
    "barcelona": {
        "name": "Barcelona",
        "country": "Spain",
        "region": "europe",
        "base_price": 1200,          # базовая цена тура (€)
        "price_spread": 0.5,         # разброс ±50% от базовой
        "duration_range": (5, 10),   # типичная длительность (ночей)
        "peak_months": [6, 7, 8],    # высокий сезон
        "photo_keyword": "barcelona",
    },
    "paris": {
        "name": "Paris",
        "country": "France",
        "region": "europe",
        "base_price": 1400,
        "price_spread": 0.55,
        "duration_range": (4, 8),
        "peak_months": [5, 6, 9],
        "photo_keyword": "paris-travel",
    },
    "prague": {
        "name": "Prague",
        "country": "Czech Republic",
        "region": "europe",
        "base_price": 700,
        "price_spread": 0.4,
        "duration_range": (3, 7),
        "peak_months": [4, 5, 9, 10],
        "photo_keyword": "prague",
    },
    "istanbul": {
        "name": "Istanbul",
        "country": "Turkey",
        "region": "europe_asia",
        "base_price": 900,
        "price_spread": 0.45,
        "duration_range": (5, 10),
        "peak_months": [4, 5, 9, 10],
        "photo_keyword": "istanbul",
    },
    "dubai": {
        "name": "Dubai",
        "country": "UAE",
        "region": "middle_east",
        "base_price": 2800,
        "price_spread": 0.6,
        "duration_range": (6, 12),
        "peak_months": [11, 12, 1, 2, 3],
        "photo_keyword": "dubai",
    },
    "bali": {
        "name": "Bali",
        "country": "Indonesia",
        "region": "asia",
        "base_price": 1600,
        "price_spread": 0.5,
        "duration_range": (9, 14),
        "peak_months": [6, 7, 8],
        "photo_keyword": "bali-indonesia",
    },
    "tokyo": {
        "name": "Tokyo",
        "country": "Japan",
        "region": "asia",
        "base_price": 2500,
        "price_spread": 0.4,
        "duration_range": (10, 14),
        "peak_months": [3, 4, 10, 11],
        "photo_keyword": "tokyo",
    },
    "maldives": {
        "name": "Maldives",
        "country": "Maldives",
        "region": "asia",
        "base_price": 4500,
        "price_spread": 0.55,
        "duration_range": (7, 14),
        "peak_months": [12, 1, 2, 3, 4],
        "photo_keyword": "maldives",
    },
    "cairo": {
        "name": "Cairo",
        "country": "Egypt",
        "region": "africa",
        "base_price": 800,
        "price_spread": 0.45,
        "duration_range": (7, 12),
        "peak_months": [10, 11, 12, 1, 2, 3],
        "photo_keyword": "cairo-egypt",
    },
    "new_york": {
        "name": "New York",
        "country": "USA",
        "region": "americas",
        "base_price": 2200,
        "price_spread": 0.5,
        "duration_range": (7, 12),
        "peak_months": [5, 6, 9, 10, 12],
        "photo_keyword": "new-york",
    },
}

HOTELS = {
    "barcelona": [
        {"id": "arts_bcn",     "name": "Hotel Arts Barcelona", "stars": 5, "price_mult": 1.6},
        {"id": "w_barcelona",  "name": "W Barcelona",          "stars": 5, "price_mult": 1.5},
        {"id": "mandarin_bcn", "name": "Mandarin Oriental",    "stars": 5, "price_mult": 1.8},
        {"id": "nobu_bcn",     "name": "Nobu Hotel Barcelona", "stars": 5, "price_mult": 1.4},
        {"id": "room_mate",    "name": "Room Mate Anna",       "stars": 4, "price_mult": 1.0},
        {"id": "eixample_bcn", "name": "Hotel Eixample",       "stars": 4, "price_mult": 0.9},
        {"id": "hostal_bcn",   "name": "Hostal Grau",          "stars": 3, "price_mult": 0.6},
    ],
    "paris": [
        {"id": "ritz_paris",    "name": "The Ritz Paris",        "stars": 5, "price_mult": 2.2},
        {"id": "le_bristol",    "name": "Le Bristol Paris",      "stars": 5, "price_mult": 2.0},
        {"id": "pullman_paris", "name": "Pullman Paris Tour",    "stars": 4, "price_mult": 1.1},
        {"id": "ibis_paris",    "name": "Ibis Paris Gare",       "stars": 3, "price_mult": 0.6},
        {"id": "novotel_paris", "name": "Novotel Paris Centre",  "stars": 4, "price_mult": 0.95},
    ],
    "prague": [
        {"id": "four_seasons_prg", "name": "Four Seasons Prague", "stars": 5, "price_mult": 1.7},
        {"id": "augustine_prg",    "name": "Augustine Prague",    "stars": 5, "price_mult": 1.5},
        {"id": "mosaic_house",     "name": "Mosaic House",        "stars": 4, "price_mult": 0.9},
        {"id": "old_town_prg",     "name": "Hotel Old Town",      "stars": 3, "price_mult": 0.65},
    ],
    "istanbul": [
        {"id": "ciragan_ist",  "name": "Çırağan Palace Kempinski", "stars": 5, "price_mult": 1.9},
        {"id": "pera_palace",  "name": "Pera Palace Hotel",         "stars": 5, "price_mult": 1.6},
        {"id": "hilton_ist",   "name": "Hilton Istanbul",           "stars": 5, "price_mult": 1.3},
        {"id": "novotel_ist",  "name": "Novotel Istanbul",          "stars": 4, "price_mult": 0.85},
        {"id": "sultan_ist",   "name": "Sultan Hostel",             "stars": 2, "price_mult": 0.45},
    ],
    "dubai": [
        {"id": "burj_al_arab",  "name": "Burj Al Arab",            "stars": 7, "price_mult": 3.5},
        {"id": "atlantis_dxb",  "name": "Atlantis The Palm",       "stars": 5, "price_mult": 2.0},
        {"id": "address_dxb",   "name": "Address Downtown Dubai",  "stars": 5, "price_mult": 1.7},
        {"id": "rove_dxb",      "name": "Rove Downtown",           "stars": 3, "price_mult": 0.7},
    ],
    "bali": [
        {"id": "como_bali",     "name": "COMO Uma Ubud",           "stars": 5, "price_mult": 1.8},
        {"id": "alila_bali",    "name": "Alila Villas Uluwatu",    "stars": 5, "price_mult": 2.0},
        {"id": "komaneka_bali", "name": "Komaneka at Bisma",       "stars": 5, "price_mult": 1.6},
        {"id": "ibis_bali",     "name": "ibis Styles Bali",        "stars": 3, "price_mult": 0.55},
    ],
    "tokyo": [
        {"id": "park_hyatt_tky", "name": "Park Hyatt Tokyo",       "stars": 5, "price_mult": 1.9},
        {"id": "aman_tky",       "name": "Aman Tokyo",             "stars": 5, "price_mult": 2.4},
        {"id": "shibuya_excel",  "name": "Shibuya Excel Hotel",    "stars": 4, "price_mult": 1.0},
        {"id": "capsule_tky",    "name": "9h Nine Hours",          "stars": 3, "price_mult": 0.4},
    ],
    "maldives": [
        {"id": "soneva_fushi",   "name": "Soneva Fushi",           "stars": 5, "price_mult": 2.5},
        {"id": "gili_maldives",  "name": "Gili Lankanfushi",       "stars": 5, "price_mult": 2.2},
        {"id": "centara_mldv",   "name": "Centara Grand Maldives", "stars": 5, "price_mult": 1.8},
        {"id": "lux_maldives",   "name": "LUX* South Ari Atoll",  "stars": 5, "price_mult": 1.6},
    ],
    "cairo": [
        {"id": "four_seasons_cai", "name": "Four Seasons Cairo",   "stars": 5, "price_mult": 1.6},
        {"id": "marriott_cai",     "name": "Cairo Marriott Hotel", "stars": 5, "price_mult": 1.4},
        {"id": "kempinski_cai",    "name": "Kempinski Nile Hotel", "stars": 5, "price_mult": 1.5},
        {"id": "ibis_cai",         "name": "ibis Cairo City",      "stars": 3, "price_mult": 0.5},
    ],
    "new_york": [
        {"id": "plaza_nyc",      "name": "The Plaza",              "stars": 5, "price_mult": 2.1},
        {"id": "four_seasons_nyc","name": "Four Seasons New York", "stars": 5, "price_mult": 2.3},
        {"id": "citizenm_nyc",   "name": "citizenM New York",      "stars": 4, "price_mult": 1.0},
        {"id": "pod_nyc",        "name": "Pod 51 Hotel",           "stars": 3, "price_mult": 0.55},
    ],
}

MEAL_TYPES = {
    "all_inclusive": {"price_add": 350, "label": "All Inclusive"},
    "breakfast":     {"price_add": 80,  "label": "Breakfast Included"},
    "none":          {"price_add": 0,   "label": "No Meals"},
}

# Шаблоны описаний — для каждого тура собираем из частей
DESCRIPTION_TEMPLATES = {
    "intro": {
        "luxury": [
            "Откройте для себя {city} в исключительном комфорте.",
            "Роскошное путешествие в самое сердце {city}.",
            "Незабываемый отдых высшего класса в {city}.",
        ],
        "comfort": [
            "Комфортное путешествие в {city} для настоящих ценителей.",
            "Идеальный отдых в {city} с отличным соотношением цены и качества.",
            "Откройте {city} вместе с нашим тщательно продуманным туром.",
        ],
        "budget": [
            "Бюджетное путешествие в {city} без потери качества.",
            "Доступный тур в {city} — максимум впечатлений за минимум денег.",
            "Познакомьтесь с {city} по доступной цене.",
        ],
    },
    "hotel_desc": {
        7: [
            "Остановитесь в легендарном {hotel} — самом роскошном отеле в мире.",
            "Ваш дом — {hotel}, непревзойдённый символ ультра-люкса.",
        ],
        5: [
            "Остановитесь в отеле {hotel} — символе роскоши и безупречного сервиса.",
            "Ваш дом — {hotel}, легендарный пятизвёздочный отель с мировым именем.",
        ],
        4: [
            "Комфортабельный четырёхзвёздочный {hotel} в отличном расположении.",
            "Отель {hotel} предлагает первоклассный сервис и удобные номера.",
        ],
        3: [
            "Уютный {hotel} — оптимальный выбор для активных путешественников.",
            "Практичный {hotel} в удобном месте для знакомства с городом.",
        ],
        2: [
            "Бюджетный {hotel} — всё необходимое для комфортного отдыха.",
            "Простой и удобный {hotel} в шаговой доступности от центра.",
        ],
    },
    "activities": {
        "europe": [
            "Посетите знаковые исторические достопримечательности и музеи.",
            "Прогулки по старому городу, шоппинг на местных рынках.",
            "Дегустация местной кухни, вечерние прогулки по набережной.",
        ],
        "europe_asia": [
            "Уникальное сочетание европейской и азиатской культур.",
            "Исторические мечети, базары, круиз по Босфору.",
            "Незабываемые закаты и традиционная кухня.",
        ],
        "middle_east": [
            "Шоппинг в роскошных молах и посещение старого базара.",
            "Сафари в пустыне, небоскрёбы и традиционная арабская кухня.",
            "Аттракционы мирового класса и архитектура будущего.",
        ],
        "asia": [
            "Храмы, рисовые поля, аутентичная местная кухня.",
            "Погружение в уникальную азиатскую культуру и традиции.",
            "Удивительная природа, духовные практики и местные рынки.",
        ],
        "africa": [
            "Пирамиды, сфинкс и тысячелетняя история цивилизации.",
            "Круиз по Нилу, базары Хан аль-Халили, Египетский музей.",
            "Незабываемое погружение в мир древних фараонов.",
        ],
        "americas": [
            "Таймс-сквер, Центральный парк, Статуя Свободы.",
            "Музеи мирового класса, Бродвей и культовые районы.",
            "Бесконечные возможности для шоппинга и гастрономии.",
        ],
    },
    "meal_desc": {
        "all_inclusive": "Полный пансион включён — никаких дополнительных расходов на питание.",
        "breakfast":     "Завтрак включён. Обеды и ужины — отличный повод попробовать местные рестораны.",
        "none":          "Питание не включено, что даёт полную свободу в выборе местной кухни.",
    },
}


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def get_category(stars: int, price_mult: float) -> str:
    """Определяет категорию тура по звёздности отеля и мультипликатору цены."""
    if stars >= 5 and price_mult >= 1.5:
        return "luxury"
    elif stars >= 4 or price_mult >= 1.0:
        return "comfort"
    else:
        return "budget"


def generate_price(city_data: dict, hotel: dict, meal_type: str,
                   month: int, rng: random.Random) -> float:
    """
    Цена = base_price × hotel_multiplier × season_multiplier + meal_add + jitter.

    Логика:
    - Высокий сезон: +20-35% к цене
    - Дорогой отель: умножаем на price_mult
    - All-inclusive: +350€ фиксированно
    - Случайный шум ±10% для реализма
    """
    base = city_data["base_price"] * hotel["price_mult"]

    # Сезонная надбавка
    if month in city_data["peak_months"]:
        season_mult = rng.uniform(1.2, 1.35)
    else:
        season_mult = rng.uniform(0.85, 1.05)

    meal_add = MEAL_TYPES[meal_type]["price_add"]

    # Шум ±10%
    jitter = rng.uniform(0.9, 1.1)

    price = (base * season_mult + meal_add) * jitter
    return round(price, 2)


def generate_duration(city_data: dict, rng: random.Random) -> int:
    """Длительность из реалистичного диапазона города, кратная 1 ночи."""
    lo, hi = city_data["duration_range"]
    return rng.randint(lo, hi)


def generate_start_date(month: int, rng: random.Random) -> str:
    """Случайная дата в указанном месяце 2025 или 2026 года."""
    year = rng.choice([2025, 2025, 2026])  # больше 2025
    max_day = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    day = rng.randint(1, max_day)
    return str(date(year, month, day))


def generate_name(city_data: dict, hotel: dict, category: str,
                  duration: int, rng: random.Random) -> str:
    """Название тура — сжато и информативно."""
    prefixes = {
        "luxury":  ["Luxury", "Premium", "Exclusive", "Elite"],
        "comfort": ["Classic", "Comfort", "Popular", "Standard"],
        "budget":  ["Budget", "Economy", "Express", "Essential"],
    }
    prefix = rng.choice(prefixes[category])
    city = city_data["name"]
    nights = f"{duration} nights"
    return f"{prefix} {city} — {nights}"


def generate_description(city_key: str, city_data: dict, hotel: dict,
                         category: str, meal_type: str,
                         rng: random.Random) -> str:
    """Описание тура собирается из тематических блоков."""
    region = city_data["region"]
    city = city_data["name"]
    hotel_name = hotel["name"]
    stars = hotel["stars"]

    intro = rng.choice(DESCRIPTION_TEMPLATES["intro"][category]).format(city=city)
    hotel_desc = rng.choice(DESCRIPTION_TEMPLATES["hotel_desc"][stars]).format(hotel=hotel_name)
    activity = rng.choice(DESCRIPTION_TEMPLATES["activities"][region])
    meal_desc = DESCRIPTION_TEMPLATES["meal_desc"][meal_type]

    return f"{intro} {hotel_desc} {activity} {meal_desc}"


def generate_photo_url(city_data: dict) -> str:
    """
    Для разработки используем picsum.photos — детерминированные красивые фото.
    В production заменить на Unsplash API.
    """
    keyword = city_data["photo_keyword"]
    # seed по keyword → стабильное фото для одного города
    seed = abs(hash(keyword)) % 10000
    return f"https://picsum.photos/seed/{seed}/800/600"


# ---------------------------------------------------------------------------
# Основной генератор
# ---------------------------------------------------------------------------

def generate_tours(count: int = 80, random_seed: int = 42) -> list[dict]:
    """
    Генерирует список туров с реалистичными распределениями.

    count: общее количество туров
    random_seed: для воспроизводимости
    """
    rng = random.Random(random_seed)
    tours = []

    # Распределение туров по городам — не равномерное, как в реальности
    city_weights = {
        "barcelona": 12, "paris": 10, "prague": 8,  "istanbul": 8,
        "dubai": 10,     "bali": 10,  "tokyo": 8,   "maldives": 8,
        "cairo": 8,      "new_york": 8,
    }
    total_weight = sum(city_weights.values())
    city_counts = {}
    remaining = count
    for i, (city, weight) in enumerate(city_weights.items()):
        if i == len(city_weights) - 1:
            city_counts[city] = remaining
        else:
            n = round(count * weight / total_weight)
            city_counts[city] = n
            remaining -= n

    for city_key, n in city_counts.items():
        city_data = CITIES[city_key]
        hotels = HOTELS[city_key]

        for _ in range(n):
            hotel = rng.choice(hotels)
            meal_type = rng.choices(
                ["all_inclusive", "breakfast", "none"],
                weights=[30, 45, 25]  # реалистичное распределение
            )[0]

            # Месяц: с учётом сезонности города
            all_months = list(range(1, 13))
            peak = city_data["peak_months"]
            month_weights = [3 if m in peak else 1 for m in all_months]
            month = rng.choices(all_months, weights=month_weights)[0]

            duration = generate_duration(city_data, rng)
            price = generate_price(city_data, hotel, meal_type, month, rng)
            start_date = generate_start_date(month, rng)
            category = get_category(hotel["stars"], hotel["price_mult"])

            tour = {
                "id": str(uuid.uuid4()),
                "name": generate_name(city_data, hotel, category, duration, rng),
                "description": generate_description(
                    city_key, city_data, hotel, category, meal_type, rng
                ),
                "city_id": city_key,
                "city_name": city_data["name"],
                "country": city_data["country"],
                "hotel_id": hotel["id"],
                "hotel_name": hotel["name"],
                "hotel_stars": hotel["stars"],
                "price": price,
                "duration_nights": duration,
                "meal_type": meal_type,
                "start_date": start_date,
                "category": category,           # luxury / comfort / budget
                "photo_url": generate_photo_url(city_data),
                # ~15% туров неактивны — реалистично для живой БД
                "status": rng.choices(
                    ["active", "inactive"],
                    weights=[85, 15]
                )[0],
            }
            tours.append(tour)

    # Перемешать, чтобы города не шли блоками
    rng.shuffle(tours)
    return tours


# ---------------------------------------------------------------------------
# Валидация и статистика
# ---------------------------------------------------------------------------

def validate_and_print_stats(tours: list[dict]) -> None:
    """Выводит статистику для проверки реалистичности данных."""
    active = [t for t in tours if t["status"] == "active"]

    print(f"\n{'='*50}")
    print(f"Всего туров:   {len(tours)}")
    print(f"Активных:      {len(active)} ({len(active)/len(tours)*100:.0f}%)")
    print(f"Неактивных:    {len(tours)-len(active)}")

    prices = [t["price"] for t in active]
    print(f"\nЦены (активные):")
    print(f"  min:    €{min(prices):.0f}")
    print(f"  max:    €{max(prices):.0f}")
    print(f"  median: €{sorted(prices)[len(prices)//2]:.0f}")

    print(f"\nПо городам (активные):")
    city_counts: dict[str, int] = {}
    for t in active:
        city_counts[t["city_id"]] = city_counts.get(t["city_id"], 0) + 1
    for city, cnt in sorted(city_counts.items(), key=lambda x: -x[1]):
        print(f"  {CITIES[city]['name']:15s} {cnt:3d} туров")

    print(f"\nПо типу питания:")
    meal_counts: dict[str, int] = {}
    for t in active:
        meal_counts[t["meal_type"]] = meal_counts.get(t["meal_type"], 0) + 1
    for meal, cnt in meal_counts.items():
        print(f"  {meal:15s} {cnt:3d} ({cnt/len(active)*100:.0f}%)")

    print(f"\nПо категории:")
    cat_counts: dict[str, int] = {}
    for t in active:
        cat_counts[t["category"]] = cat_counts.get(t["category"], 0) + 1
    for cat, cnt in cat_counts.items():
        print(f"  {cat:10s} {cnt:3d} ({cnt/len(active)*100:.0f}%)")

    # Проверки здравого смысла
    assert all(t["price"] > 0 for t in tours), "Найдены туры с нулевой ценой!"
    assert all(t["duration_nights"] >= 3 for t in tours), "Слишком короткие туры!"
    assert len(set(t["id"] for t in tours)) == len(tours), "Дублирующиеся ID!"
    print(f"\n✅ Все проверки пройдены")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генератор туров для recommendation-service")
    parser.add_argument("--count", type=int, default=80, help="Количество туров (default: 80)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default="data/tours_seed.json",
                        help="Путь к выходному файлу")
    parser.add_argument("--stats", action="store_true", default=True,
                        help="Показать статистику")
    args = parser.parse_args()

    # Создаём папку если нет
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Генерирую {args.count} туров (seed={args.seed})...")
    tours = generate_tours(count=args.count, random_seed=args.seed)

    if args.stats:
        validate_and_print_stats(tours)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tours, f, ensure_ascii=False, indent=2)

    print(f"Сохранено → {output_path}")
    print(f"Пример тура:")
    import pprint
    pprint.pprint(tours[0], width=80)
