import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

# Load the proposed Windows sources without replacing a running WSL evaluation's
# hash-locked source files. Also works when run directly from the main repository.
SOURCE = Path(__file__).parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SOURCE/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('mode', ['online', 'cloud', 'offline'])
def test_private_cloud_and_offline_fallback(tmp_path, monkeypatch, mode):
    config_module = load('cloud_test_config', 'config.py')
    tracking = load('cloud_test_tracking', 'tracking.py')
    config = config_module.Config(swanlab=mode).validate()
    (tmp_path/'mini').mkdir()
    (tmp_path/'mini/UPSTREAM.json').write_text('{}')
    monkeypatch.setattr(tracking, 'ROOT', tmp_path)
    monkeypatch.setattr(tracking, 'file_hash', lambda _: 'verified-test-digest')
    captured = {}
    def init(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(finish=lambda: None)
    monkeypatch.setitem(sys.modules, 'swanlab', SimpleNamespace(init=init))
    log = tracking.RunLog(tmp_path/'run', config)
    log.finish()
    assert captured['mode'] == ('online' if mode == 'cloud' else mode)
    assert captured['public'] is False
    assert captured['on_online_error'] == 'offline'
    assert captured['project'] == 'grepseek-mini'
    assert 'api_key' not in captured


@pytest.mark.parametrize('override,expected', [(None, 'online'), ('offline', 'offline')])
def test_rl_cli_online_default_and_explicit_override(tmp_path, monkeypatch, capsys, override, expected):
    import mini.pytrio_backend.config as installed_config
    import mini.pytrio_backend.data as data
    proposed = load('cloud_cli_config', 'config.py')
    monkeypatch.setattr(installed_config, 'Config', proposed.Config)
    monkeypatch.setattr(data, 'preflight', lambda _: (None, {'files_sha256': {}}))
    argv = ['backend', 'rl', '--out', str(tmp_path/'not-created'), '--initial-checkpoint', 'not-loaded.json']
    if override:
        argv.extend(['--swanlab', override])
    monkeypatch.setattr(sys, 'argv', argv)
    cli = load('cloud_test_cli', '__main__.py')
    assert cli.main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['config']['swanlab'] == expected
    assert plan['execute'] is False
    assert not (tmp_path/'not-created').exists()
