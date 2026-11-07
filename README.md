# W Projects · ТОП-106Б
## Поймай мошенника

В корне находятся презентация, математический отчёт и scored.csv с итоговыми оценками для 10 000 операций. В project/lab находятся исходники, тесты, обученная модель и исследовательские материалы. В work/hackathon/data находятся входные CSV.

### Запуск демонстрации

Распакуйте весь архив, откройте терминал в его корне и выполните с Python 3.14:

```bash
cd project/lab
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -v
python -m antifraud serve --model models/mvp-001 --data-dir ../../work/hackathon/data
```

Откройте http://127.0.0.1:8000/ и выберите «Данные хакатона». Подробности обучения и проверки приведены в project/lab/README.md.

## Материалы для жюри

- [Презентация](Защита_Поймай_мошенника.pptx)
- [Математический отчёт](Математический_отчёт_Поймай_мошенника.docx)
- [Результаты анализа 10 000 операций](scored.csv)
- [Исходный код и инструкция](project/lab/README.md)
- [Исследования и проверка MVP](project/lab/docs/research/mvp-validation.md)
- [Техническое задание](materials/Техническое_задание.pdf)
