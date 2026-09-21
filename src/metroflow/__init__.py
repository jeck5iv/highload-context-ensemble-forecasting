from .config import ExperimentConfig, load_experiment_config


def run_experiment(*args, **kwargs):
    from .experiment import run_experiment as run
    return run(*args, **kwargs)
