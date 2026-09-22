"""Exercise command forwarding and history recovery in a disposable repository."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hexapod cli ")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo with spaces"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "docs").mkdir()
        shutil.copy(ROOT / "hexapod", self.repo)
        shutil.copy(ROOT / "scripts/hexapod_cli.py", self.repo / "scripts")
        for name in ("train_adaptive_curriculum.sh", "view_foothold_planner.sh"):
            (self.repo / "scripts" / name).write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$HEXAPOD_PYTHON" "$@"\n')
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Launcher Test")
        (self.repo / "original.txt").write_text("original snapshot\n")
        self.git("add", ".")
        self.git("commit", "-qm", "original")
        self.original = self.git("rev-parse", "HEAD").strip()
        (self.repo / "docs/integration-sources.json").write_text(json.dumps({"sources": [
            {"id": "original", "sha": self.original, "branch": "old", "disposition": "preserved"}]}))
        (self.repo / "original.txt").write_text("current snapshot\n")
        self.git("add", ".")
        self.git("commit", "-qm", "integration")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True, stderr=subprocess.STDOUT)

    def cli(self, *args, success=True):
        result = subprocess.run(["bash", str(self.repo / "hexapod"), *map(str, args)],
                                cwd=self.temp.name, env={**os.environ, "HEXAPOD_PYTHON": sys.executable},
                                capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_help_is_inert_and_unknown_commands_fail(self):
        before = self.git("rev-parse", "HEAD")
        self.assertIn("train-rc", self.cli().stdout)
        self.cli("typo", success=False)
        self.cli("history", "original", success=False)
        self.assertEqual(self.git("rev-parse", "HEAD"), before)

    def test_forwarding_preserves_spaces_and_does_not_evaluate_shell(self):
        literal = "run name; $(touch unwanted)"
        result = self.cli("train", "--run-name", literal)
        lines = result.stdout.splitlines()
        self.assertIn(literal, lines)
        self.assertIn(sys.executable, lines)
        self.assertIn("hybrid", lines)
        self.assertFalse((Path(self.temp.name) / "unwanted").exists())

    def test_dry_run_does_not_execute_missing_script(self):
        (self.repo / "scripts/train_adaptive_curriculum.sh").unlink()
        self.assertIn("--profile rc", self.cli("--dry-run", "train-rc").stdout)

    def test_exact_snapshot_in_separate_worktree_and_no_overwrite(self):
        destination = Path(self.temp.name) / "restored snapshot"
        head = self.git("rev-parse", "HEAD")
        self.cli("history", "original", destination)
        self.assertEqual((destination / "original.txt").read_text(), "original snapshot\n")
        self.assertEqual(self.git("rev-parse", "HEAD"), head)
        self.cli("history", "original", destination, success=False)

    def test_bundle_roundtrip_preserves_original_without_other_branches(self):
        destination = Path(self.temp.name) / "source.bundle"
        self.cli("bundle", destination)
        restored = Path(self.temp.name) / "clone"
        subprocess.run(["git", "clone", "-q", str(destination), str(restored)], check=True)
        old = subprocess.check_output(["git", "-C", str(restored), "show",
                                       self.original + ":original.txt"], text=True)
        self.assertEqual(old, "original snapshot\n")
        self.assertEqual((restored / "original.txt").read_text(), "current snapshot\n")
        self.cli("bundle", destination, success=False)

    def test_bundle_rejects_dirty_source(self):
        (self.repo / "original.txt").write_text("uncommitted")
        self.assertIn("Commit source", self.cli("bundle", Path(self.temp.name) / "x.bundle", success=False).stderr)

    def test_non_ancestor_is_not_reported_as_preserved(self):
        self.git("checkout", "--orphan", "unrelated")
        self.git("commit", "-qm", "unrelated root")
        self.cli("verify-history", success=False)

    def test_resume_requires_checkpoint_and_help_does_not_train(self):
        for name in ("resume_stair5_path_v6.sh", "resume_stair5_v6_clearance.sh"):
            shutil.copy(ROOT / "scripts" / name, self.repo / "scripts")
        for command in ("resume-v5", "resume-v6"):
            self.assertIn("--checkpoint", self.cli(command, "--help").stdout)
            self.cli(command, success=False)
            self.cli(command, "--checkpoint", success=False)


if __name__ == "__main__":
    unittest.main()
