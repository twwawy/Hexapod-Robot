"""Nonblocking bridge between the existing explorer and archived PPO dynamics."""
from multiprocessing.connection import Connection
from pathlib import Path
import os
import socket
import subprocess
import sys

import mujoco
import numpy as np

from foothold_preview_runtime import SiteIK


class PolicyProcess:
    def __init__(self, args):
        parent, child = socket.socketpair()
        self.connection = Connection(parent.detach())
        self.process = subprocess.Popen(
            [sys.executable, '-u', str(Path(__file__).with_name('foothold_policy_worker.py')),
             '--fd', str(child.fileno())], pass_fds=(child.fileno(),),
            env={**os.environ, 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'})
        child.close()
        self.connection.send({key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()})
        try:
            ready = self.connection.recv()
            if ready['kind'] != 'ready':
                raise RuntimeError(ready.get('error', 'Policy worker failed to create the scene'))
            self.model = mujoco.MjModel.from_binary_path(ready['model'])
            self.data = mujoco.MjData(self.model)
            self.data.qpos[:] = ready['home']
            mujoco.mj_forward(self.model, self.data)
            self.manifest = ready['manifest']
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process.poll() is None:
            try:
                self.connection.send(dict(kind='close'))
            except (OSError, EOFError):
                pass
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=3)
        self.connection.close()


class LearnedFootholdGait:
    def __init__(self, backend):
        self.backend = backend
        self.model, self.data = backend.model, backend.data
        self.ik = SiteIK(self.model, self.data.qpos)
        self.home_height = float(self.data.qpos[2])
        self.duration, self.elapsed, self.phase_scale = 0.8, 0.0, 1.0
        self.plans = {}  # LiDAR plans are overlays; never claim they drive this PPO.
        self.targets = None
        self.action = np.zeros(18)
        self.anchors = self.ik.positions(self.data)
        self.completed_swings = 0
        self.status = 'Preparing stage31 policy / JAX compilation'
        self.pending, self.reset_requested, self.paused, self.terminal = True, False, False, False
        self.failed = False
        self.accumulator = 0.0
        self.policy_dt = 0.02
        self._applied = np.zeros(3)

    def tick(self, result, command, dt, target_height, **unused):
        del result, unused
        if self.failed:
            return
        try:
            while self.backend.connection.poll():
                message = self.backend.connection.recv()
                self.pending = False
                if message['kind'] == 'error':
                    raise RuntimeError(message['error'])
                self.data.qpos[:] = message['qpos']
                self.data.qvel[:] = message['qvel']
                self.data.ctrl[:] = message['ctrl']
                self.data.time = message['time']
                mujoco.mj_forward(self.model, self.data)
                self.targets, self.action = message['targets'], message['action']
                self.anchors = self.ik.positions(self.data)
                self.completed_swings = message['completed_swings']
                self.terminal, self.status = message['terminal'], message['status']
                self._applied = message['applied']
                self.policy_dt = message['dt']
                self.elapsed = float(np.max(message['progress']))*self.duration
            if self.reset_requested and not self.pending:
                self.backend.connection.send(dict(kind='reset'))
                self.pending, self.reset_requested = True, False
                self.status = 'Resetting policy and dynamics'
                return
            if self.paused:
                self.status = 'Dynamics paused; Enter resumes'
                return
            self.accumulator = min(self.accumulator+dt, self.policy_dt)
            if not self.pending and not self.terminal and self.accumulator >= self.policy_dt:
                height = float(np.clip(target_height-self.home_height, -0.05, 0.10))
                policy_command = [float(command[0]), float(command[2]), height, 0.0, 0.0]
                self.backend.connection.send(dict(kind='step', command=policy_command))
                self.pending = True
                self.accumulator = 0.0
        except (EOFError, OSError, RuntimeError) as error:
            self.failed = True
            self.status = 'Policy worker failed: restart viewer; see terminal'
            print(error, flush=True)

    def reset(self):
        self.reset_requested, self.paused = True, False

    def retry(self):
        if self.terminal:
            self.reset()

    def stop_body(self):
        # UI sets velocity command to zero; physics must keep evolving to settle.
        pass

    @property
    def applied_command(self):
        return np.zeros(3) if self.paused or self.terminal or self.failed else self._applied
