"""Enabled only by mini.launch's child PYTHONPATH; never changes training math."""
import os

if os.environ.get('MINI_OBSERVE_DIR'):
    from mini.observability import install
    install()
