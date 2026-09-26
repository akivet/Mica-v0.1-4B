"""Example in-process judge: Laya (github.com/NandhaKishorM/laya), which is a Python library rather than a server.

    pip install "git+https://github.com/NandhaKishorM/laya.git@c7527708f9f5220c669d8aa385077cd28d04708a" huggingface_hub safetensors
    python bench.py --judge mica=http://127.0.0.1:8010/v1/systemone --judge laya=py:examples/laya_judge.py

Any file with make_judge(arg) -> fn(state, questions) -> {"answers": {...}} (the /v1/systemone response shape) works the
same way; arg is whatever follows a second ':' in the --judge spec (here: the checkpoint subfolder).
"""
import time

LAYA_REPO = 'convaiinnovations/laya'
LAYA_REVISION = '1c5edc17a7acd8701df6fc341c0d179f1c62c982'


def make_judge(arg=None):
    import torch
    import laya.agent as la
    from huggingface_hub import snapshot_download
    sub = arg or 'typed-decisions'
    root = snapshot_download(LAYA_REPO, revision=LAYA_REVISION, allow_patterns=[
        f'{sub}/{n}' for n in ('rl_agent_config.json', 'model.safetensors', 'tokenizer/*', 'encoder/*')])
    agent = la.Agent(root, device='cuda' if torch.cuda.is_available() else 'cpu', subfolder=sub)

    def ask(state, questions):
        t = time.perf_counter()
        res = agent.system_one(state, questions)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res['latency_ms'] = (time.perf_counter() - t) * 1000
        return res
    return ask
