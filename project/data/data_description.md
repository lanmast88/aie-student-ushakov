# Описание данных

## Датасет: Goodbooks-10k

- **Источник:** Kaggle — [zygmuntz/goodbooks-10k](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k)
- **Размер:** 10 000 книг, 53 424 пользователя, ~6 млн оценок

### Файлы

| Файл | Описание |
|---|---|
| `books.csv` | Метаданные книг: название, автор, год, ISBN, рейтинг |
| `ratings.csv` | Индивидуальные оценки: `user_id`, `book_id`, `rating` (1–5) |
| `book_tags.csv` | Теги книг от пользователей: `goodreads_book_id`, `tag_id`, `count` |
| `tags.csv` | Справочник тегов: `tag_id`, `tag_name` |
| `to_read.csv` | Книги в списке "хочу прочитать": `user_id`, `book_id` |

### Ключевые колонки ratings.csv

| Колонка | Тип | Описание |
|---|---|---|
| user_id | int | ID пользователя |
| book_id | int | ID книги |
| rating | int | Оценка от 1 до 5 |

### Ключевые колонки books.csv

| Колонка | Тип | Описание |
|---|---|---|
| book_id | int | ID книги |
| title | str | Название |
| authors | str | Авторы |
| average_rating | float | Средний рейтинг |
| ratings_count | int | Число оценок |
| original_publication_year | float | Год публикации |

## Загрузка

```bash
# Настройте Kaggle API:
#   1. https://www.kaggle.com/settings → API → Create New Token
#   2. mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

python -m src.data.download
```
