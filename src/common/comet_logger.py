"""Общая обёртка над созданием comet_ml.Experiment — раньше каждый
train-скрипт повторял ``comet_ml.Experiment(workspace=..., disabled=...)``
+ ``set_name``/``add_tags``/``log_parameters`` по отдельности. Само имя
эксперимента, список тегов и словарь параметров всё ещё строит вызывающий
train.py (они у каждого подхода свои), эта функция только оборачивает
одинаковую часть и даёт единый способ отключить логирование (--smoke-test)."""

import comet_ml


def init_experiment(comet_cfg, name, tags, parameters):
    """comet_cfg — под-словарь config.yaml: {"enabled": bool, "workspace": str}.
    enabled=False -> comet_ml.Experiment(disabled=True) — no-op заглушка,
    ничего не отправляет и не создаёт запись на comet.com (используется для
    --smoke-test, чтобы не засорять проект короткими проверочными прогонами)."""
    experiment = comet_ml.Experiment(
        workspace=comet_cfg.get("workspace", "team-rl-exp"),
        disabled=not comet_cfg.get("enabled", True),
    )
    experiment.set_name(name)
    experiment.add_tags(tags)
    experiment.log_parameters(parameters)
    return experiment
