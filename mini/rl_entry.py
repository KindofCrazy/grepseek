"""Use the official Hydra/Ray entrypoint, installing read-only diagnostics in Ray workers."""
from verl.trainer.main_ppo import main

if __name__ == '__main__':
    main()
