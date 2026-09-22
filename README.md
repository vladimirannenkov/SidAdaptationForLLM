# LocalLearning: обучение GPT без сквозного backprop

## Подходы

| Подход | Каталог | Идея | Лучший результат |
|---|---|---|---|
| E2E-baseline | `src/experiments/baseline_e2e/` | Обычный сквозной backprop, точка сравнения | val_ppl 105.68 (100% бюджета) |
| SID-S | `src/experiments/sid_s/`, `sid_s_blocks/` | Backbone с нуля + верхние блоки, multi-token loss | — (архитектурный поиск, см. memory.md) |
| SID-P-blocks | `src/experiments/sid_p_blocks/` | Backbone из E2E-чекпоинта, не заморожен, consistency-KL | +1.4% от глубины |
| SID-P-Boost | `src/experiments/sid_p_boost/` | То же, но boosting-лосс вместо consistency-KL | +4.6% от глубины |
| SID-F | `src/experiments/sid_f/` | Backbone заморожен, чистый локальный CE | +6.9% от глубины (k=3) |
| Newton-SID | `src/experiments/newton_sid/` | Boosting + квадратичный Newton-лосс на коррекцию блока | отрицательный (см. memory.md) |
| Cascade-SID | `src/experiments/cascade/` | Последовательные заморожены-навсегда stages, three-zone loss | smoke — чисто, полный прогон не зафиксирован |


## Структура репозитория

```
src/
  common/        device/dtype setup, LR-расписание, checkpoint I/O, Comet-логирование, config.yaml loader
  model/         ядро GPT (GPTConfig, Block, CausalSelfAttention, MLP)
  data/wikitext/ токенизация и подготовка WikiText-103 в бинарные token-стримы
  sid/           общая библиотека SID: forward по диапазону слоёв, chunked-лоссы, optimizers, newton, cka, checkpoint
  cascade/       лоссы и checkpoint-формат Cascade-SID
  visualization/ Tuned Lens + CKA layer-wise анализ обученных чекпоинтов
  experiments/   train.py + config.yaml на каждый подход (таблица выше)
tests/           короткие проверочные скрипты (сгруппированы как src/)
tools/
  profiling/     замеры throughput/VRAM/batch size
  analysis/      разовые аналитические скрипты (CKA, linear probe, Tuned Lens визуализация)
docs/            план, журнал (краткий + полный), отчёты об экспериментах, логи прогонов
reference/nanogpt_pinned/  пиннутые файлы апстрима nanoGPT (только для цитирования, не импортируются)
data/wikitext/   данные: tokenizer/ (в git), raw_cache/ и bin/ (в .gitignore, воспроизводимы)
checkpoints*/    чекпоинты обучения (в .gitignore, большие)
```

## Установка

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Требуется PyTorch с CUDA (проект калибровался под 6-7 ГБ VRAM), либо CPU —
тогда обучение будет на порядки медленнее, но код корректен на обоих.

## Подготовка данных (один раз)

```bash
python src/data/wikitext/train_tokenizer.py   # обучает BPE, data/wikitext/tokenizer/
python src/data/wikitext/prepare.py           # токенизирует WikiText-103 в data/wikitext/bin/
```

Ожидает parquet-шарды WikiText-103-raw-v1 в `data/wikitext/raw_cache/`.

## Запуск эксперимента

Каждый эксперимент представлен конфигурацией `train.py` + `config.yaml` рядом с ним:

```bash
python src/experiments/baseline_e2e/train.py
python src/experiments/sid_f/train.py --k 3
```

Гиперпараметры по умолчанию — в `config.yaml` эксперимента; специфичные для
подхода флаги (`--k`, `--freeze-readout`, `--lambda-div` и т.п.) — через CLI,
их значения по умолчанию тоже берутся из `config.yaml`, но обычно варьируются
между запусками.
