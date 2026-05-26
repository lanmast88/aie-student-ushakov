# Описание данных — Similar Tours Recommendation

## Датасет: tours.json

- **Источник:** синтетические данные, сгенерированные скриптом `generate_tours.py`
- **Размер:** 200 туров
- **Реальных персональных данных нет** — соответствует правилам README

### Файлы

| Файл | Описание |
|---|---|
| `tours.json` | Синтетический датасет туров |
| `generate_tours.py` | Скрипт генерации датасета |
| `batch_utils.py` | Утилита разбивки и мёржа батчей |

### Структура одного тура

| Поле | Тип | Описание |
|---|---|---|
| id | string | UUID тура |
| name | string | Название тура |
| description | string | Текстовое описание (2–3 предложения) |
| city_id | string | Идентификатор города |
| city_name | string | Название города |
| country | string | Страна |
| hotel_id | string | Идентификатор отеля |
| hotel_name | string | Название отеля |
| hotel_stars | int | Звёздность отеля (2, 3, 4, 5, 7) |
| price | float | Цена тура в евро |
| duration_nights | int | Длительность в ночах |
| meal_type | string | `all_inclusive` / `breakfast` / `none` |
| start_date | string | Дата начала тура (YYYY-MM-DD) |
| category | string | `luxury` / `comfort` / `budget` |
| photo_url | string | URL фото (picsum.photos placeholder) |
| status | string | `active` / `inactive` |

### Как воспроизвести датасет

```bash
python3 generate_tours.py --count 200 --seed 42 --output tours.json
```

### Происхождение данных

Синтетические данные сгенерированы скриптом с реалистичной бизнес-логикой:
цены коррелируют с городом, отелем и сезоном.
Описания написаны с помощью Claude (Anthropic).
Фотографии — плейсхолдеры от picsum.photos.
