# Inference — deepfake detection

Скрипт `Inference.py`: детекция лица (SCRFD) → кроп → классификатор real/fake (TorchScript Xception).

## Установка

Нужны **uv** (≥0.9) и **Python 3.10**. Зависимости зафиксированы в `uv.lock`.

```bash
uv sync --frozen    # создаёт .venv и ставит пакеты
source .venv/bin/activate
```

Альтернатива:

```bash
uv pip install -r requirements.txt --python .venv \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

Системная зависимость для видео-оверлея: **ffmpeg** (`sudo apt install ffmpeg`).

Чекпоинт детектора (`checkpoint/epoch_45_model.pt`) в git не входит — положите файл вручную перед запуском.

## Структура репозитория

```
Inference.py              # точка входа
SCRFD/nets/nn.py          # детектор лиц
SCRFD/weights/model_1.onnx
checkpoint/epoch_45_model.pt   # нужно поместить чекпоинт сюда
pyproject.toml / uv.lock  # окружение
```

## Быстрый старт

```bash
source .venv/bin/activate

# Фото — JSON в stdout
CUDA_VISIBLE_DEVICES=0 python Inference.py /path/to/image.jpg

# Видео с оверлеем
CUDA_VISIBLE_DEVICES=0 python Inference.py /path/to/video.mp4 --overlay
```

## Параметры `Inference.py`

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `input` | path | — | Путь к изображению или видео (обязательный позиционный аргумент) |
| `--checkpoint` | path | `checkpoint/epoch_45_model.pt` | TorchScript-модель детектора (`.pt`) |
| `--scrfd-model` | path | `SCRFD/weights/model_1.onnx` | SCRFD ONNX для детекции лица |
| `--overlay` | flag | выкл. | Нарисовать bbox и подпись real/fake; сохранить файл |
| `--output` | path | `None` | Путь для оверлея; без флага — `<имя>_overlay.<ext>` рядом с входом |
| `--device` | `cuda` \| `cpu` | `cuda` | Устройство для детектора; при отсутствии GPU — CPU |
| `--bbox-scale` | float | **1.0** | Масштаб квадратного кропа: `side = max(w, h) × scale`. Должен совпадать с обучением |
| `--min-size` | int | `380` | Если меньшая сторона кропа &lt; min-size — апскейл до `min-size×min-size`; иначе без ресайза |
| `--scrfd-thresh` | float | `0.5` | Порог уверенности SCRFD |
| `--scrfd-input-size` | int int | `640 640` | Размер входа SCRFD (ширина высота) |
| `--keep-tmp` | flag | выкл. | Не удалять кадры видео из `tmp/` |
| `--json-out` | path | `None` | Дополнительно сохранить JSON с результатом в файл |

## Вывод

- **stdout** — JSON: `label` (`real` / `fake`), `fake_prob`, `confidence`, по кадрам для видео.
- **stderr** — `Saved overlay: ...` при `--overlay`.
- Цвет рамки: зелёный = real, красный = fake.

## Примеры

```bash

# Свой чекпоинт и JSON на диск
python Inference.py clip.mp4 \
  --checkpoint checkpoint/epoch_45_model.pt \
  --json-out results/clip.json

# Оверлей в заданный путь
python Inference.py clip.mp4 --overlay --output output/clip_overlay.mp4
```
