"""Record the CPU runtime and optionally read-only PyTRIO account capabilities."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', default='mini/reports/environment/runtime.json')
    parser.add_argument('--check-account', action='store_true', help='Read model metadata only; no paid model requests')
    args = parser.parse_args()
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'platform': platform.platform(), 'python': sys.version,
              'executable': sys.executable, 'project': str(Path.cwd()),
              'os_release': Path('/etc/os-release').read_text() if Path('/etc/os-release').exists() else None,
              'wsl_config': Path('/etc/wsl.conf').read_text() if Path('/etc/wsl.conf').exists() else None,
              'packages': {name: importlib.metadata.version(name) for name in
                           ['pytrio', 'transformers', 'tokenizers', 'huggingface-hub', 'pytest', 'pandas', 'pyarrow']},
              'tools': {name: shutil.which(name) for name in
                        ['bash', 'rg', 'grep', 'head', 'tail', 'sed', 'awk', 'find', 'wc', 'sort', 'cut', 'uniq', 'tr', 'git']},
              'model_weights_downloaded': False, 'paid_model_calls': 0}
    if args.check_account:
        import pytrio
        client = pytrio.ServiceClient()
        report['pytrio_capabilities'] = client.get_server_capabilities().model_dump(mode='json')
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
