from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from metroflow.config import load_experiment_config
from metroflow.research import run_and_materialize


if __name__ == '__main__':
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / 'configs' / 'sportivnaya_15m_60m.toml')
    cfg = load_experiment_config(cfg_path)
    result = run_and_materialize(cfg)
    print(result['summary_table'])
    print(f'\nOutput dir: {result["output_dir"]}')
    for k, v in sorted(result['artifact_paths'].items()):
        print(f'- {k}: {v}')
