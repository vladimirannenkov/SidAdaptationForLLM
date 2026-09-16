"""Загрузка config.yaml экспериментов (PyYAML уже стоит в окружении py-dl).
Каждый src/experiments/<name>/config.yaml несёт архитектуру модели, гиперпараметры
обучения, путь к данным, каталог чекпоинтов и настройки Comet; специфичные
для подхода флаги (k, freeze-режимы и т.п.) остаются в argparse самого
train.py — они меняются от запуска к запуску чаще, чем стабильные настройки
здесь, и YAML на каждую комбинацию флагов был бы избыточен."""

import copy

import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_smoke_overrides(cfg, experiment_name):
    """--smoke-test: небольшой target_tokens и частые (но редко считаемые)
    проверки из cfg["smoke"], Comet отключён, чекпоинты пишутся в отдельный
    checkpoints_smoke/<experiment_name>/ — настоящие чекпоинты не трогаются."""
    cfg = copy.deepcopy(cfg)
    cfg["train"].update(cfg.get("smoke", {}))
    cfg["comet"]["enabled"] = False
    cfg["checkpoint"]["dir"] = f"checkpoints_smoke/{experiment_name}"
    return cfg
