"""Run every curriculum retry from one committed, isolated source checkout.

No JAX import or training occurs until the final exec. Checkpoint contract
validation remains the trainer's responsibility; this is not a migration tool.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    repo = Path(__file__).resolve().parent.parent
    args = sys.argv[1:]
    trainer = repo / 'mjx/train_adaptive_curriculum.py'
    if '--help' in args or '-h' in args:
        os.execv(sys.executable, [sys.executable, str(trainer), *args])
    dirty = subprocess.check_output(
        ['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'],
        text=True).strip()
    if dirty:
        raise SystemExit('Commit tracked source changes before training. '
                         'The curriculum runs from an isolated committed checkout.')
    revision = subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    cache = Path.home() / '.cache/hexapod-training-sources'
    cache.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix=revision[:12] + '-', dir=cache)) / 'source'
    subprocess.run(['git', '-C', str(repo), 'worktree', 'add', '--detach',
                    str(destination), revision], check=True)
    # Preserve the ordinary output location rather than writing runs into cache.
    if not any(a == '--run-root' or a.startswith('--run-root=') for a in args):
        args.extend(['--run-root', str(repo / 'mjx/runs/adaptive-curriculum')])
    print(f'PINNED SOURCE: {destination}\nSOURCE REVISION: {revision}', flush=True)
    # Keep caller cwd: relative restore/run-root arguments retain their meaning.
    # Every child trainer is resolved by the pinned manager's __file__.
    os.execv(sys.executable, [sys.executable, '-u',
             str(destination / 'mjx/train_adaptive_curriculum.py'), *args])


if __name__ == '__main__':
    main()
