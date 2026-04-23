from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from metroflow.config import load_experiment_config
from metroflow.research import run_and_materialize


if __name__ == '__main__':
    configs = sys.argv[1:] or [
        str(ROOT / 'configs' / 'sportivnaya_15m_15m.toml'),
        str(ROOT / 'configs' / 'sportivnaya_15m_60m.toml'),
        str(ROOT / 'configs' / 'sportivnaya_60m_360m.toml'),
    ]
    for cfg_path in configs:
        cfg = load_experiment_config(cfg_path)
        result = run_and_materialize(cfg)
        print(f'\n===== {cfg.name} =====')
        print(result['summary_table'].head(10))
        print(f'Output dir: {result["output_dir"]}')
