"""Exercise the manual runner's process boundary without CUDA or GitHub access."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


TOOL = Path(__file__).with_name("manual_cuda.py")


class ManualCudaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Test")
        (self.repo / "sentinel").write_text("committed")
        self.git("add", ".")
        self.git("-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
        self.sha = self.git("rev-parse", "HEAD").strip()
        (self.repo / "sentinel").write_text("dirty")
        self.out = self.root / "results"
        self.julia = self.executable("julia", '''
import os, pathlib, sys
assert pathlib.Path("sentinel").read_text() == "committed"
assert "--startup-file=no" in sys.argv
assert "--project=test/gpu/cuda" in sys.argv
code = sys.argv[-1]
stage = "setup" if "Pkg.instantiate()" in code else "test"
if stage == "test":
    assert "CUDA.functional() || error" in code
    assert 'include(' in code
    if "MPI.mpiexec()" in code:
        assert "Comm_size" in code and "length(devices) >= 2" in code
    else:
        assert "length(devices) >= 1" in code
print(stage, "devices=" + os.environ.get("CUDA_VISIBLE_DEVICES", "inherited"))
sys.exit(int(os.environ.get("FAKE_" + stage.upper() + "_EXIT", "0")))
''')
        self.capture = self.root / "github.json"
        self.gh = self.executable("gh", '''
import json, os, pathlib, sys
pathlib.Path(os.environ["GH_CAPTURE"]).write_text(json.dumps({
    "argv": sys.argv[1:], "payload": json.load(sys.stdin)}))
sys.exit(int(os.environ.get("FAKE_GH_EXIT", "0")))
''')

    def executable(self, name, body):
        path = self.root / name
        path.write_text("#!" + sys.executable + "\n" + body)
        path.chmod(0o755)
        return path

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True)

    def cli(self, *args, env=None):
        return subprocess.run([sys.executable, str(TOOL), *map(str, args)],
                              text=True, capture_output=True,
                              env={**os.environ, "GH_CAPTURE": str(self.capture), **(env or {})})

    def run_cuda(self, *args, env=None):
        return self.cli("run", "--repo", self.repo, "--output", self.out,
                        "--julia", self.julia, *args, env=env)

    def report(self, *args, env=None):
        return self.cli("report", self.out / "result.json", "--gh", self.gh,
                        "--repository", "owner/project", *args, env=env)

    def test_success_tests_committed_snapshot_and_preserves_checkout(self):
        proc = self.run_cuda("--devices", "0")
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        result = json.loads((self.out / "result.json").read_text())
        self.assertEqual(result["commit"], self.sha)
        self.assertTrue(result["completed"])
        self.assertEqual([s["exit_code"] for s in result["stages"]], [0, 0])
        self.assertEqual((self.repo / "sentinel").read_text(), "dirty")
        self.assertIn("test devices=0", (self.out / "cuda-test.log").read_text())

    def test_report_preview_has_no_side_effect_and_publish_uses_recorded_sha(self):
        self.assertEqual(self.run_cuda().returncode, 0)
        self.git("-c", "commit.gpgsign=false", "commit", "-qam", "later commit")
        preview = self.report()
        self.assertEqual(preview.returncode, 0, preview.stderr)
        self.assertFalse(self.capture.exists())
        self.assertIn(self.sha, preview.stdout)
        published = self.report("--publish", "--target-url", "https://example.org/log")
        self.assertEqual(published.returncode, 0, published.stderr)
        call = json.loads(self.capture.read_text())
        self.assertIn("repos/owner/project/statuses/" + self.sha, call["argv"])
        self.assertEqual(call["payload"]["state"], "success")
        self.assertEqual(call["payload"]["context"], "CUDA / remote manual")
        self.assertEqual(call["payload"]["target_url"], "https://example.org/log")

    def test_explicit_ref_selects_older_commit(self):
        self.git("-c", "commit.gpgsign=false", "commit", "-qam", "later commit")
        proc = self.run_cuda("--ref", self.sha)
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        self.assertEqual(json.loads((self.out / "result.json").read_text())["commit"], self.sha)

    def test_setup_and_test_failures_are_reportable(self):
        for stage, expected in [("SETUP", "error"), ("TEST", "failure")]:
            with self.subTest(stage=stage):
                self.out = self.root / stage
                proc = self.run_cuda(env={"FAKE_" + stage + "_EXIT": "7"})
                self.assertNotEqual(proc.returncode, 0)
                preview = self.report()
                self.assertEqual(preview.returncode, 0, preview.stderr)
                self.assertEqual(json.loads(preview.stdout)["payload"]["state"], expected)

    def test_missing_julia_records_setup_error(self):
        proc = self.run_cuda("--julia", self.root / "missing-julia")
        self.assertNotEqual(proc.returncode, 0)
        preview = self.report()
        self.assertEqual(preview.returncode, 0, preview.stderr)
        self.assertEqual(json.loads(preview.stdout)["payload"]["state"], "error")

    def test_mpi_is_optional_and_requires_two_visible_devices(self):
        proc = self.run_cuda("--mpi", "--devices", "0,1")
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        result = json.loads((self.out / "result.json").read_text())
        self.assertEqual([s["name"] for s in result["stages"]], ["setup", "cuda", "mpi_cuda"])
        self.assertIn("MPI", json.loads(self.report().stdout)["payload"]["context"])

    def test_reused_output_is_rejected(self):
        self.assertEqual(self.run_cuda().returncode, 0)
        original = (self.out / "result.json").read_bytes()
        self.assertNotEqual(self.run_cuda().returncode, 0)
        self.assertEqual((self.out / "result.json").read_bytes(), original)

    def test_incomplete_or_inconsistent_evidence_cannot_publish(self):
        self.assertEqual(self.run_cuda().returncode, 0)
        path = self.out / "result.json"
        original = json.loads(path.read_text())
        for change in [{"completed": False}, {"stages": []}, {"commit": "HEAD"},
                       {"stages": [{"name": "setup", "exit_code": 0}]},
                       {"stages": [{"name": "setup", "exit_code": True}]}]:
            with self.subTest(change=change):
                path.write_text(json.dumps({**original, **change}))
                self.assertNotEqual(self.report("--publish").returncode, 0)
                self.assertFalse(self.capture.exists())
        path.write_text(json.dumps(original))
        (self.out / "cuda-test.log").write_text("truncated")
        self.assertNotEqual(self.report("--publish").returncode, 0)
        self.assertFalse(self.capture.exists())

    def test_github_failure_is_not_reported_as_success(self):
        self.assertEqual(self.run_cuda().returncode, 0)
        self.assertNotEqual(self.report("--publish", env={"FAKE_GH_EXIT": "1"}).returncode, 0)


if __name__ == "__main__":
    unittest.main()
