"""Pinned downloads use hashes and do not replace existing weight files."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ModelDownloads(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copy(ROOT / "fetch-models.sh", self.root)
        self.payload = b"model fixture\n"
        self.sha = hashlib.sha256(self.payload).hexdigest()
        self.revision = "a" * 40
        self.manifest = self.root / "models.lock.tsv"
        self.manifest.write_text(
            "# group\tmodel_id\tquant\tdefault\trepo\trevision\tfilename\tbytes\tsha256\n"
            + self.row("bonsai", "old", "Q1_0", "yes", "old.gguf")
            + self.row("agents", "new", "Q4_K_M", "yes", "new.gguf")
            + self.row("browser", "large", "Q4_K_XL", "yes", "large-xl.gguf")
            + self.row("browser", "large", "IQ4_XS", "no", "large-xs.gguf")
        )
        self.bin = self.root / "bin"
        self.bin.mkdir()
        fake = self.bin / "hf"
        fake.write_text("#!/usr/bin/env python3\n"
                        "import os,pathlib,sys\n"
                        "a=sys.argv[1:]\n"
                        "pathlib.Path(os.environ['CALLS']).write_text(repr(a))\n"
                        "assert a[0]=='download' and a[3]=='--revision'\n"
                        "assert a[4]=='" + self.revision + "'\n"
                        "d=pathlib.Path(a[a.index('--local-dir')+1]);d.mkdir(parents=True,exist_ok=True)\n"
                        "(d/a[2]).write_bytes(b'wrong content' if os.environ.get('BAD_DOWNLOAD') else b'model fixture\\n')\n")
        fake.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        CALLS=str(self.root / "calls"))

    def row(self, group, model_id, quant, default, name):
        return (f"{group}\t{model_id}\t{quant}\t{default}\towner/repo\t{self.revision}"
                f"\t{name}\t{len(self.payload)}\t{self.sha}\n")

    def run_fetch(self, *args):
        return subprocess.run(["bash", str(self.root / "fetch-models.sh"), *args],
                              env=self.env, capture_output=True, text=True)

    def test_default_downloads_only_bonsai_and_pins_revision(self):
        result = self.run_fetch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "models/repo/old.gguf").read_bytes(), self.payload)
        self.assertFalse((self.root / "models/repo/new.gguf").exists())

    def test_agent_selector_does_not_download_bonsai(self):
        result = self.run_fetch("agents")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "models/repo/new.gguf").exists())
        self.assertFalse((self.root / "models/repo/old.gguf").exists())

    def test_model_selector_downloads_only_default_quant(self):
        result = self.run_fetch("--model", "large")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "models/repo/large-xl.gguf").exists())
        self.assertFalse((self.root / "models/repo/large-xs.gguf").exists())
        self.assertFalse((self.root / "models/repo/old.gguf").exists())

    def test_model_selector_can_choose_alternative_quant(self):
        result = self.run_fetch("--model", "large", "--quant", "IQ4_XS")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "models/repo/large-xs.gguf").exists())
        self.assertFalse((self.root / "models/repo/large-xl.gguf").exists())

    def test_all_flag_retains_all_artifact_behavior(self):
        result = self.run_fetch("--all")
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("old.gguf", "new.gguf", "large-xl.gguf", "large-xs.gguf"):
            self.assertTrue((self.root / "models/repo" / name).exists())

    def test_missing_model_or_quant_fails_before_download(self):
        for args in (("--model", "missing"), ("--model", "large", "--quant", "Q3_K_M"),
                     ("--quant", "IQ4_XS")):
            with self.subTest(args=args):
                result = self.run_fetch(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / "calls").exists())

    def test_valid_existing_file_is_not_downloaded(self):
        dest = self.root / "models/repo/old.gguf"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(self.payload)
        result = self.run_fetch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / "calls").exists())

    def test_same_size_wrong_hash_is_preserved_and_rejected(self):
        dest = self.root / "models/repo/old.gguf"
        dest.parent.mkdir(parents=True)
        original = b"x" * len(self.payload)
        dest.write_bytes(original)
        result = self.run_fetch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA256", result.stderr)
        self.assertEqual(dest.read_bytes(), original)
        self.assertFalse((self.root / "calls").exists())

    def test_bad_download_is_not_installed(self):
        self.env["BAD_DOWNLOAD"] = "1"
        result = self.run_fetch()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "models/repo/old.gguf").exists())

    def test_invalid_selector_fails_before_download(self):
        result = self.run_fetch("latest")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "calls").exists())

    def test_unpinned_revision_fails_before_download(self):
        self.manifest.write_text(self.row("bonsai", "old", "Q1_0", "yes", "old.gguf").replace(self.revision, "main"))
        result = self.run_fetch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("revision", result.stderr)
        self.assertFalse((self.root / "calls").exists())

    def test_repository_manifest_has_pinned_browser_variants(self):
        rows = [line.split("\t") for line in (ROOT / "models.lock.tsv").read_text().splitlines()
                if line and not line.startswith("#")]
        self.assertEqual(len(rows), 12)
        self.assertEqual(sum(row[0] == "agents" for row in rows), 3)
        self.assertEqual(sum(row[0] == "browser" for row in rows), 5)
        defaults = {row[1]: row[2] for row in rows if row[3] == "yes"}
        self.assertEqual(defaults["qwen3.6-35b-a3b"], "Q4_K_XL")
        self.assertEqual(defaults["gemma4-e4b"], "QAT_Q4_0")
        for row in rows:
            self.assertEqual(len(row), 9)
            self.assertRegex(row[5], r"^[a-f0-9]{40}$")
            self.assertRegex(row[8], r"^[a-f0-9]{64}$")
            self.assertGreater(int(row[7]), 0)


class OfflineVersions(unittest.TestCase):
    def test_configured_and_installed_versions_are_distinguished_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy(ROOT / "version.sh", root)
            (root / "runtime-versions.env").write_text("PRISM_SHA=" + "a" * 40 + "\nCUDA_VERSION=12.8.1\n")
            (root / "models.lock.tsv").write_text("agents\tfixture\tQ4_K_M\tyes\towner/model\t" + "b" * 40 + "\tmodel.gguf\t12\t" + "c" * 64 + "\n")
            (root / "build-revision").write_text("fixture-qcm-sha\n")
            binary = root / "llama-server"
            binary.write_text("#!/bin/sh\nprintf 'fixture-installed-build\\n'\n")
            binary.chmod(0o755)
            result = subprocess.run(["bash", str(root / "version.sh")], capture_output=True,
                                    text=True, env=dict(os.environ, LLAMA_SERVER=str(binary), API_KEY="must-not-print"))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Configured Prism", result.stdout)
            self.assertIn("fixture-qcm-sha", result.stdout)
            self.assertIn("Installed llama-server", result.stdout)
            self.assertIn("fixture-installed-build", result.stdout)
            self.assertIn("owner/model", result.stdout)
            self.assertIn("fixture Q4_K_M", result.stdout)
            self.assertNotIn("must-not-print", result.stdout)


if __name__ == "__main__":
    unittest.main()
