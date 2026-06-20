# Similar Tours Recommender — Итоговый проект

Сервис рекомендаций похожих туров на основе content-based подхода.
По идентификатору тура REST API возвращает Top-K наиболее похожих туров из каталога.

---

## 1. Паспорт проекта

- **Название:** Сервис рекомендаций похожих туров (Similar Tours Recommender)
- **Автор:** Ушаков Глеб Максимович
- **Группа:** ЭФБО-06-24
- **Контакт:** @lanmast8
- **Курс:** Инженерия Искусственного Интеллекта

**Краткое описание:**
Content-based рекомендательная система для туристической платформы.
Модель комбинирует структурные признаки тура (город, категория, звёздность, цена, длительность, тип питания) и TF-IDF-векторы текстовых описаний.
Cosine similarity вычисляется заранее и хранится в памяти — запрос к `/predict` занимает 1–5 мс.

---

## 2. Структура проекта

```
project/
├── configs/
│   └── config.yaml                  # параметры модели и сервиса
├── data/
│   ├── tours.json                   # синтетический датасет (200 туров)
│   └── data_description.md          # описание полей датасета
├── notebooks/
│   ├── exp01_eda.ipynb              # разведочный анализ данных
│   ├── exp02_baseline.ipynb         # сравнение моделей по метрикам
│   ├── exp03_weight_tuning.ipynb    # подбор весов комбинированной модели
│   └── exp04_object_map.ipynb       # карта объектов (t-SNE / PCA)
├── src/
│   ├── models/
│   │   ├── features.py              # построение матриц признаков (one-hot, TF-IDF)
│   │   ├── recommender.py           # логика рекомендаций (get_similar_tours)
│   │   └── metrics.py               # метрики качества (CHR, Coverage, ILD, Pers)
│   └── service/
│       └── app.py                   # FastAPI-приложение (/health, /tours, /predict)
├── artifacts/
│   ├── exp01/                       # графики EDA (9 PNG)
│   ├── exp02/                       # матрица и метрики baseline
│   ├── exp03/                       # финальные артефакты сервиса
│   │   ├── sim_combined.npy         # матрица схожести 170×170
│   │   ├── scaler.pkl               # MinMaxScaler
│   │   ├── tfidf_vectorizer.pkl
│   │   ├── tours_active.json        # 170 активных туров
│   │   ├── best_config.json         # оптимальные веса
│   │   └── weight_search.csv        # результаты grid search
│   └── exp04/                       # карта объектов
│       ├── 11_object_map.png        # t-SNE и PCA, 4 панели
│       ├── 12_similarity_heatmap.png # тепловая карта схожести запроса
│       ├── tsne_coords.npy          # 2D-координаты t-SNE
│       └── pca_coords.npy           # 2D-координаты PCA
├── tests/
│   ├── conftest.py                  # shared fixtures (mini_df, mini_sim)
│   ├── test_features.py             # unit: build_feature_matrices
│   ├── test_recommender.py          # unit: get_similar_tours, get_popularity_recs
│   ├── test_metrics.py              # unit: CHR, Coverage, MeanSim, ILD, Personalization
│   └── test_api.py                  # integration: /health, /tours, /predict
├── Dockerfile
├── pyproject.toml                   # зависимости для uv
├── requirements-service.txt
├── report.md                        # подробный технический отчёт
└── self-checklist.md
```

---

## 3. Требования и установка

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) (рекомендуется) **или** pip

### Установка зависимостей

```bash
cd project

# Вариант 1 — через uv (рекомендуется)
uv sync

# Вариант 2 — через pip
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-service.txt
```

**Зависимости сервиса** (`requirements-service.txt`):

| Библиотека | Версия | Назначение |
|---|---|---|
| fastapi | 0.136.3 | REST API фреймворк |
| uvicorn | 0.48.0 | ASGI-сервер |
| numpy | 2.3.4 | Работа с матрицами |
| pandas | 2.3.3 | Работа с табличными данными |
| scikit-learn | 1.8.0 | MinMaxScaler, TfidfVectorizer, t-SNE, PCA |
| pyyaml | 6.0.2 | Чтение config.yaml |

---

## 4. Как запустить проект

### 4.1. Модель уже обучена

Артефакты финальной модели сохранены в `artifacts/exp03/` и загружаются сервисом при старте автоматически.
**Переобучение не требуется.**

Чтобы воспроизвести эксперименты — запустить ноутбуки в порядке:

```
exp01_eda.ipynb → exp02_baseline.ipynb → exp03_weight_tuning.ipynb → exp04_object_map.ipynb
```

### 4.2. Запуск сервиса — без Docker

```bash
cd project
uv run uvicorn src.service.app:app --host 0.0.0.0 --port 8000
```

или через активированное виртуальное окружение:

```bash
cd project
source .venv/bin/activate
uvicorn src.service.app:app --host 0.0.0.0 --port 8000
```

### 4.3. Запуск сервиса — через Docker

```bash
cd project
docker build -t tour-recommender .
docker run -p 8000:8000 tour-recommender
```

Сервис поднимается на `http://localhost:8000`.

### 4.4. Проверка работоспособности

```bash
curl http://localhost:8000/health
# {"status": "ok", "tours_loaded": 170, "sim_matrix_shape": [170, 170]}
```

Интерактивная документация (Swagger UI): `http://localhost:8000/docs`

---

## 5. Тесты

49 тестов, покрывающих все модули проекта.

```bash
cd project
uv run python -m pytest tests/ -v
```

| Файл | Тестов | Что покрывается |
|---|---|---|
| `test_features.py` | 6 | формы матриц, MinMaxScaler [0,1], L2-норма TF-IDF |
| `test_recommender.py` | 12 | top_k, self-exclusion, порядок similarity, edge cases |
| `test_metrics.py` | 12 | блочная/анти-блочная матрица, ILD при top_k=1, монотонность |
| `test_api.py` | 19 | все эндпоинты, фильтры, clamping top_k, 404 |

Тесты используют минимальный синтетический датафрейм из 9 туров (`conftest.py`) — реальный датасет задействован только в интеграционных тестах API.

---

## 6. API — эндпоинты

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "tours_loaded": 170, "sim_matrix_shape": [170, 170]}
```

---

### `GET /tours`

Список туров с опциональной фильтрацией.

| Параметр | Тип | Описание |
|---|---|---|
| `city` | string (opt) | Фильтр по городу (case-insensitive) |
| `category` | string (opt) | Фильтр по категории: `luxury`, `comfort`, `budget` |

```bash
curl "http://localhost:8000/tours?city=Istanbul&category=comfort"
```

---

### `POST /predict`

Top-K похожих туров по `tour_id`.

| Поле | Тип | Описание |
|---|---|---|
| `tour_id` | string | UUID тура из каталога |
| `top_k` | int (opt) | Количество рекомендаций, 1–20, по умолчанию 5 |

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"tour_id": "362b3800-ccd0-47d2-b8cc-158bd0d22742", "top_k": 5}'
```

```json
{
  "tour_id": "362b3800-ccd0-47d2-b8cc-158bd0d22742",
  "recommendations": [
    {
      "id": "...",
      "name": "Comfort Istanbul — 10 nights",
      "city_name": "Istanbul",
      "category": "comfort",
      "hotel_stars": 3,
      "meal_type": "breakfast",
      "price": 1234.56,
      "similarity": 0.981
    }
  ],
  "elapsed_ms": 1.4
}
```

---

## 7. Конфигурация

Параметры задаются в `configs/config.yaml`:

```yaml
artifacts_dir: artifacts/exp03   # директория с артефактами модели

model:
  struct_weight: 0.7             # вес структурных признаков
  tfidf_weight: 0.3              # вес TF-IDF описаний
  top_k: 5                       # Top-K по умолчанию
  tfidf_max_features: 200
  tfidf_ngram_range: [1, 2]

service:
  port: 8000
  log_level: INFO
```

Путь к конфигу можно переопределить через переменную окружения:

```bash
CONFIG_PATH=/path/to/my_config.yaml uvicorn src.service.app:app ...
```

---

## 8. Данные

Синтетический датасет `data/tours.json`: **200 туров, 10 городов** (Bali, Barcelona, Cairo, Dubai, Istanbul, Maldives, New York, Paris, Prague, Tokyo).

- Активных туров: **170** (участвуют в рекомендациях).
- Категории: luxury (~50 %), comfort, budget.
- Цена: €353 – €14 000 (медиана €1 837).

Подробное описание полей: `data/data_description.md`.

---

## 9. Результаты экспериментов

### Сравнение моделей (`exp02_baseline.ipynb`)

| Модель | CHR@5 | Coverage | MeanSim@5 | ILD@5 | Pers@5 |
|---|---|---|---|---|---|
| Quality Baseline (популярность) | 0.531 | 0.400 | 0.709 | 0.187 | 0.922 |
| Структурные признаки | 0.934 | 0.982 | 0.878 | 0.176 | 0.969 |
| TF-IDF по описаниям | 0.725 | 1.000 | 0.427 | 0.707 | 0.970 |
| **Комбинированная (w=0.7)** | **0.936** | **1.000** | 0.720 | 0.355 | **0.972** |

### Подбор весов (`exp03_weight_tuning.ipynb`)

Grid search по `w_struct ∈ [0.1, 0.9]` подтвердил оптимальность `w=0.7` при ограничении Coverage=1.0.

### Карта объектов (`exp04_object_map.ipynb`)

t-SNE на матрице расстояний `1 − sim_combined` (KL=0.50) визуализирует пространство туров.
Результат: городские кластеры хорошо выражены, топ-5 рекомендаций расположены рядом с запросом.

---

## 10. Ограничения и дальнейшая работа

**Текущие ограничения:**
- Матрица схожести фиксирована — при добавлении нового тура требуется пересчёт и перезапуск.
- Content-based only — история взаимодействий пользователей не используется.
- `start_date` не используется как фильтр доступности.

**Направления развития:**
- Фильтрация по дате отправления на уровне сервиса.
- Приближённый поиск ближайших соседей (FAISS) для масштабирования.
- Гибридная модель: content-based + popularity boost.
- Sentence-embeddings вместо TF-IDF для более точного текстового сигнала.

---
