"""Optional sequential vLLM serving, so three checkpoints can share one GPU."""
from contextlib import contextmanager
import subprocess
import sys
import time


@contextmanager
def checkpoint_server(checkpoint, args, output_dir):
    if not args.serve_locally:
        yield checkpoint['base_url']
        return
    from openai import OpenAI
    url = f'http://127.0.0.1:{args.server_port}/v1'
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', checkpoint['model'],
               '--served-model-name', checkpoint['model'], '--host', '127.0.0.1', '--port', str(args.server_port),
               '--tensor-parallel-size', str(args.server_tp), '--max-model-len', str(args.context_length),
               '--max-num-seqs', '4', '--gpu-memory-utilization', str(args.server_memory),
               '--reasoning-parser', 'qwen3', '--language-model-only', '--enforce-eager']
    # Refuse to accidentally reuse an existing server/another checkpoint.
    import socket
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', args.server_port)) == 0:
            raise RuntimeError('local evaluation server port is already occupied')
    with (output_dir / f'{checkpoint["label"]}-server.log').open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + args.server_timeout
            client = OpenAI(base_url=url, api_key='EMPTY', timeout=3, max_retries=0)
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'vLLM exited; inspect {checkpoint["label"]}-server.log')
                try:
                    if checkpoint['model'] in {m.id for m in client.models.list().data}:
                        break
                except Exception:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('vLLM startup exceeded --server-timeout')
                time.sleep(2)
            yield url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
