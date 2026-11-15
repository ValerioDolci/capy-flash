"""Release and model installation against a local stand-in for GitHub and Hugging Face (tests/fake_sources.py)."""

import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from capy_app import catalog, downloads  # noqa: E402
from capy_app.downloads import DownloadError, Fetcher  # noqa: E402
from capy_app.installer import Busy, Installer, _extract  # noqa: E402
from fake_sources import Sources, model_files, release_files, sha256, tarball  # noqa: E402

M27 = catalog.model("27b-16gb")
TARBALL = catalog.tarball_name("v0.12.0")


class InstallCase(unittest.TestCase):
    retries = 0

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "capy-flash"
        self.sources = Sources()
        self.files = model_files(M27["artifact"])
        self.sources.add_model(M27["hf_repo"], self.files)
        self.installer = self.make()

    def make(self, retries=None):
        fetcher = Fetcher(
            allow_http=True, timeout=5, retries=self.retries if retries is None else retries, backoff=0.01
        )
        return Installer(self.root, fetcher, github_api=self.sources.url, github=self.sources.url, hf=self.sources.url)

    def tearDown(self):
        self.installer.cancel()
        self.installer.wait(5)
        self.sources.close()
        self.tmp.cleanup()

    def run_job(self, model=None, installer=None, **kw):
        installer = installer or self.installer
        job = installer.start(model, **kw)
        installer.wait(30)
        return job

    def asked(self, name):
        return [(p, r) for p, r in self.sources.requests if p.endswith("/" + name)]

    def part(self, name, folder="27b-16gb"):
        """Where the app keeps a download in progress: its own corner of the root, never the model folder."""
        return self.root / ".capy-app" / "tmp" / folder / (name + ".part")

    @staticmethod
    def picture(folder):
        """Every file under `folder` with its bytes and times: equal pictures mean nothing was touched."""
        out = {}
        for path in sorted(Path(folder).rglob("*")):
            if path.is_file():
                st = path.stat()
                out[str(path.relative_to(folder))] = (path.read_bytes(), st.st_mtime_ns, st.st_mode)
        return out


class TestHttpsOnly(unittest.TestCase):
    def test_plain_http_refused(self):
        with self.assertRaises(DownloadError) as c:
            Fetcher().text("http://127.0.0.1:9/x")
        self.assertEqual(c.exception.code, "insecure")
        with self.assertRaises(DownloadError):
            Fetcher().download("ftp://example.org/x", Path(tempfile.gettempdir()) / "never")

    def test_redirect_to_http_refused(self):
        handler = downloads._HttpsOnlyRedirects()
        req = urllib.request.Request("https://example.org/a")
        with self.assertRaises(DownloadError) as c:
            handler.redirect_request(req, None, 302, "Found", {}, "http://example.org/b")
        self.assertEqual(c.exception.code, "insecure")
        self.assertIsNotNone(handler.redirect_request(req, None, 302, "Found", {}, "https://cdn.example.org/b"))

    def test_sums_and_paths(self):
        sums = downloads.parse_sums("%s  a.bin\n%s *./b/c.bin\nnot a line\n" % ("a" * 64, "B" * 64))
        self.assertEqual(sums, {"a.bin": "a" * 64, "b/c.bin": "b" * 64})
        for bad in ("", "/etc/passwd", "../x", "a/../../x", "a//b", "a\\b", "./a"):
            self.assertFalse(downloads.safe_relative(bad), bad)
        self.assertTrue(downloads.safe_relative("bin/ninfer-serve"))


class TestRelease(InstallCase):
    def test_install(self):
        job = self.run_job()
        self.assertEqual(job.state, "done", job.error)
        release = self.installer.release()
        self.assertEqual((release["version"], release["verified"]), ("v0.12.0", True))
        self.assertEqual(os.readlink(self.root / "current"), "capy-flash-v0.12.0")
        launcher = self.installer.launcher(catalog.model("flash-next"))
        self.assertTrue(os.access(launcher, os.X_OK))
        self.assertEqual(
            sorted(p.name for p in self.root.iterdir() if not p.name.startswith(".")),
            ["capy-flash-v0.12.0", "current"],
            "the tarball and its part are gone",
        )
        snap = job.snapshot()
        self.assertEqual(snap["bytes_done"], snap["bytes_total"])

    def test_corrupt_tarball_is_never_unpacked(self):
        self.sources.corrupt.add(TARBALL)
        job = self.run_job()
        self.assertEqual((job.state, job.error["code"]), ("failed", "checksum"))
        self.assertIsNone(self.installer.release())
        self.assertFalse((self.root / "current").exists())
        self.assertEqual([p.name for p in self.root.iterdir() if not p.name.startswith(".")], [], "nothing unpacked")
        self.assertEqual(list((self.root / ".capy-app" / "tmp").iterdir()), [], "the corrupt download is gone")
        # the published file is fine again: the same button works
        self.sources.corrupt.clear()
        self.assertEqual(self.run_job().state, "done")

    def test_release_that_fails_its_own_sums_is_not_adopted(self):
        files = release_files()
        files["bin/ninfer-serve"] = b"#!/bin/sh\necho tampered\n"  # SHA256SUMS still lists the old digest
        self.sources.release = tarball("v0.12.0", files)
        self.sources.release_sha = sha256(self.sources.release)
        job = self.run_job()
        self.assertEqual((job.state, job.error["code"]), ("failed", "checksum"))
        self.assertFalse((self.root / "current").exists())
        self.assertFalse((self.root / "capy-flash-v0.12.0").exists())

    def test_release_must_cover_engine_and_launchers(self):
        files = release_files()
        files["SHA256SUMS"] = ("%s  README.md\n" % sha256(files["README.md"])).encode()
        self.sources.release = tarball("v0.12.0", files)
        self.sources.release_sha = sha256(self.sources.release)
        job = self.run_job()
        self.assertEqual((job.state, job.error["code"]), ("failed", "checksum"))
        self.assertIsNone(self.installer.release())

    def test_unsafe_archives_refused(self):
        outside = Path(self.tmp.name) / "evil"
        cases = [
            tarball("v1", {"../../evil": b"x"}),
            tarball("v1", {"ok": b"x"}, extra=[("top2/file", tarfile.REGTYPE, "")]),
            tarball(
                "v1",
                {"ok": b"x"},
                extra=[("capy-flash-v1-linux-x86_64-sm120a-ubuntu26.04/link", tarfile.SYMTYPE, "/etc")],
            ),
            tarball(
                "v1",
                {"ok": b"x"},
                extra=[("capy-flash-v1-linux-x86_64-sm120a-ubuntu26.04/up", tarfile.SYMTYPE, "../../evil")],
            ),
            tarball(
                "v1", {"ok": b"x"}, extra=[("capy-flash-v1-linux-x86_64-sm120a-ubuntu26.04/dev", tarfile.CHRTYPE, "")]
            ),
        ]
        for n, data in enumerate(cases):
            archive = Path(self.tmp.name) / f"a{n}.tar.gz"
            archive.write_bytes(data)
            into = Path(self.tmp.name) / f"into{n}"
            into.mkdir()
            with self.assertRaises(DownloadError, msg=n) as c:
                _extract(archive, into)
            self.assertEqual(c.exception.code, "layout")
            self.assertFalse(outside.exists())
        # a link that stays inside is fine (lib/libx.so -> libx.so.1)
        good = Path(self.tmp.name) / "good.tar.gz"
        good.write_bytes(
            tarball(
                "v1",
                {"lib/libx.so.1": b"x"},
                extra=[("capy-flash-v1-linux-x86_64-sm120a-ubuntu26.04/lib/libx.so", tarfile.SYMTYPE, "libx.so.1")],
            )
        )
        into = Path(self.tmp.name) / "into-good"
        into.mkdir()
        top = _extract(good, into)
        self.assertEqual((top / "lib" / "libx.so").read_bytes(), b"x")

    def test_bad_version_refused_before_any_request(self):
        with self.assertRaises(ValueError):
            self.installer.start(None, version="v0.12.0/../../x")
        self.assertEqual(self.sources.requests, [])

    def test_network_failure_reported(self):
        self.sources.fail_status["/repos/"] = 404
        job = self.run_job()
        self.assertEqual((job.state, job.error["code"]), ("failed", "http"))


class TestModel(InstallCase):
    def setUp(self):
        super().setUp()
        self.assertEqual(self.run_job().state, "done")  # the release
        self.sources.requests.clear()
        self.folder = self.root / "models" / "27b-16gb"

    def test_download_and_verify(self):
        self.assertEqual(self.installer.model_status("27b-16gb")["installed"], False)
        job = self.run_job("27b-16gb")
        self.assertEqual(job.state, "done", job.error)
        for name, data in self.files.items():
            if name == ".gitattributes":
                self.assertFalse((self.folder / name).exists())
            else:
                self.assertEqual((self.folder / name).read_bytes(), data, name)
        status = self.installer.model_status("27b-16gb")
        self.assertEqual(
            (status["state"], status["installed"], status["verified"], status["deletable"]), ("ready", True, True, True)
        )
        self.assertEqual(status["bytes"], sum(len(d) for n, d in self.files.items() if n != ".gitattributes"))
        self.assertEqual(
            sorted(p.name for p in self.folder.iterdir()),
            sorted(n for n in self.files if n != ".gitattributes"),
            "only the repository's files",
        )
        self.assertFalse((self.root / ".capy-app" / "tmp" / "27b-16gb").exists(), "no temporary file is left")
        self.assertEqual(self.asked(TARBALL), [], "the installed release is not downloaded again")
        self.assertEqual(self.installer.installed_models(), ["27b-16gb"])

    def test_corrupt_file_is_deleted_and_never_installed(self):
        self.sources.corrupt.add(M27["artifact"])
        job = self.run_job("27b-16gb")
        self.assertEqual((job.state, job.error["code"]), ("failed", "checksum"))
        self.assertFalse((self.folder / M27["artifact"]).exists())
        self.assertFalse(self.part(M27["artifact"]).exists())
        status = self.installer.model_status("27b-16gb")
        self.assertEqual((status["installed"], status["verified"]), (False, False))
        self.sources.corrupt.clear()
        self.assertEqual(self.run_job("27b-16gb").state, "done")
        self.assertTrue(self.installer.model_status("27b-16gb")["verified"])

    def test_file_outside_sha256sums_checked_against_the_published_git_digest(self):
        self.sources.corrupt.add("README.md")
        job = self.run_job("27b-16gb")
        self.assertEqual((job.state, job.error["code"]), ("failed", "checksum"))
        self.assertFalse((self.folder / "README.md").exists())

    def test_resume_after_the_connection_breaks(self):
        self.sources.break_after[M27["artifact"]] = 120_000
        self.sources.on_break = lambda name: self.sources.fail_status.update({"/" + M27["hf_repo"] + "/resolve": 503})
        job = self.run_job("27b-16gb")  # the connection breaks, then the server is down
        self.assertEqual((job.state, job.error["code"]), ("failed", "network"))
        part = self.part(M27["artifact"])
        self.assertEqual(part.stat().st_size, 120_000)
        self.assertTrue(self.installer.model_status("27b-16gb")["partial"])
        self.assertFalse(list(self.folder.glob("*.part")), "no temporary file in the model folder")
        self.sources.requests.clear()
        self.sources.fail_status.clear()
        job = self.run_job("27b-16gb")
        self.assertEqual(job.state, "done", job.error)
        self.assertEqual(self.asked(M27["artifact"])[0][1], "bytes=120000-", "continued, not started again")
        self.assertEqual((self.folder / M27["artifact"]).read_bytes(), self.files[M27["artifact"]])
        self.assertEqual(self.asked("NOTICE"), [], "files already verified are not fetched again")
        self.assertEqual(job.snapshot()["bytes_done"], job.snapshot()["bytes_total"])

    def test_retries_continue_inside_one_job(self):
        self.sources.break_after[M27["artifact"]] = 50_000
        installer = self.make(retries=3)
        job = self.run_job("27b-16gb", installer)
        self.assertEqual(job.state, "done", job.error)
        self.assertEqual([r for _, r in self.asked(M27["artifact"])], [None, "bytes=50000-"])

    def test_server_that_ignores_range_restarts_cleanly(self):
        self.sources.break_after[M27["artifact"]] = 120_000
        self.sources.on_break = lambda name: self.sources.fail_status.update({"/" + M27["hf_repo"] + "/resolve": 503})
        self.run_job("27b-16gb")
        self.sources.fail_status.clear()
        self.sources.ignore_range = True
        job = self.run_job("27b-16gb")
        self.assertEqual(job.state, "done", job.error)
        self.assertEqual((self.folder / M27["artifact"]).read_bytes(), self.files[M27["artifact"]])
        self.assertEqual(job.snapshot()["bytes_done"], job.snapshot()["bytes_total"])

    def test_damaged_part_fails_verification_and_is_not_used(self):
        self.sources.break_after[M27["artifact"]] = 120_000
        self.sources.on_break = lambda name: self.sources.fail_status.update({"/" + M27["hf_repo"] + "/resolve": 503})
        self.run_job("27b-16gb")
        self.sources.fail_status.clear()
        part = self.part(M27["artifact"])
        part.write_bytes(b"x" * 120_000)  # the part was damaged on disk; the server continues it as asked
        job = self.run_job("27b-16gb")
        self.assertEqual((job.state, job.error["code"]), ("failed", "checksum"))
        self.assertFalse((self.folder / M27["artifact"]).exists())
        self.assertFalse(part.exists())
        self.assertEqual(self.run_job("27b-16gb").state, "done")  # the next attempt starts that file again

    def test_someone_elses_part_file_is_neither_continued_nor_touched(self):
        self.folder.mkdir(parents=True)
        stray = self.folder / (M27["artifact"] + ".part")
        stray.write_bytes(b"x" * 1000)
        before = self.picture(self.folder)
        job = self.run_job("27b-16gb")
        self.assertEqual(job.state, "done", job.error)
        self.assertEqual(self.asked(M27["artifact"])[0][1], None)
        self.assertEqual(self.picture(self.folder)[stray.name], before[stray.name])

    def test_cancel_keeps_the_part(self):
        self.sources.rate = 150_000
        job = self.installer.start("27b-16gb")
        deadline = time.time() + 10
        while time.time() < deadline and not (
            job.step == "download" and job.file == M27["artifact"] and job.bytes_done > 40_000
        ):
            time.sleep(0.02)
        self.assertTrue(self.installer.cancel(job.id))
        self.installer.wait(10)
        self.assertEqual((job.state, job.error), ("cancelled", None))
        part = self.part(M27["artifact"])
        self.assertTrue(0 < part.stat().st_size < len(self.files[M27["artifact"]]))
        self.assertFalse((self.folder / M27["artifact"]).exists())
        self.sources.rate = 0
        self.sources.requests.clear()
        self.assertEqual(self.run_job("27b-16gb").state, "done")
        self.assertRegex(self.asked(M27["artifact"])[0][1], r"^bytes=\d+-$")

    def test_disk_space_checked_before_downloading(self):
        real = shutil.disk_usage
        downloads.shutil.disk_usage = lambda p: type(real(p))(10**12, 10**12 - 1000, 1000)
        try:
            job = self.run_job("27b-16gb")
        finally:
            downloads.shutil.disk_usage = real
        self.assertEqual((job.state, job.error["code"]), ("failed", "disk"))
        self.assertEqual(self.asked(M27["artifact"]), [])

    def test_one_job_at_a_time_and_known_models_only(self):
        self.sources.rate = 100_000
        self.installer.start("27b-16gb")
        with self.assertRaises(Busy):
            self.installer.start("27b-16gb")
        self.installer.cancel()
        self.installer.wait(10)
        for bad in ("nope", "../x", ""):
            with self.assertRaises(ValueError):
                self.installer.start(bad)
            with self.assertRaises(ValueError):
                self.installer.model_dir(bad)

    def test_missing_published_checksum_stops_the_install(self):
        files = dict(self.files)
        files["SHA256SUMS"] = ("%s  NOTICE\n" % sha256(files["NOTICE"])).encode()  # the artifact is not listed
        self.sources.add_model(M27["hf_repo"], files)
        job = self.run_job("27b-16gb")
        self.assertEqual((job.state, job.error["code"]), ("failed", "layout"))
        self.assertEqual(self.asked(M27["artifact"]), [])

    def test_new_revision_drops_old_parts(self):
        self.sources.break_after[M27["artifact"]] = 120_000
        self.run_job("27b-16gb")
        newer = model_files(M27["artifact"])
        self.sources.add_model(M27["hf_repo"], newer, revision="b" * 40)
        self.sources.requests.clear()
        job = self.run_job("27b-16gb")
        self.assertEqual(job.state, "done", job.error)
        self.assertEqual(self.asked(M27["artifact"])[0][1], None, "a part of another revision is not continued")
        self.assertEqual((self.folder / M27["artifact"]).read_bytes(), newer[M27["artifact"]])

    def test_remove(self):
        self.run_job("27b-16gb")
        self.installer.remove_model("27b-16gb")
        self.assertFalse(self.folder.exists())
        self.assertEqual(self.installer.installed_models(), [])


OLD = "a" * 40  # the revision of what somebody else left on disk


class ThereAlready(InstallCase):
    """A release installed by install.sh and a model folder the app did not create."""

    def setUp(self):
        super().setUp()
        release = self.root / "capy-flash-v0.12.0"
        for name, data in release_files().items():
            path = release / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(0o755)
        os.symlink(str(release), self.root / "current")  # install.sh links the absolute path
        self.folder = self.root / "models" / "27b-16gb"
        self.put(self.folder, self.files)
        self.new = self.root / "models" / ("27b-16gb-" + self.sources.repos[M27["hf_repo"]]["revision"][:8])

    @staticmethod
    def put(folder, files):
        folder.mkdir(parents=True)
        for name, data in files.items():
            (folder / name).write_bytes(data)
            os.utime(folder / name, (1_700_000_000, 1_700_000_000))  # "last year": any rewrite would show

    def older(self, **changed):
        """The same file names with other contents: an earlier revision of the repository."""
        files = dict(self.files)
        for name in changed or {M27["artifact"]: 1, "README.md": 1}:
            files[name] = bytes(len(files[name])) if name == M27["artifact"] else files[name] + b"(older)\n"
        listed = [n for n in files if n not in ("README.md", ".gitattributes", "SHA256SUMS")]
        files["SHA256SUMS"] = "".join(f"{sha256(files[n])}  {n}\n" for n in sorted(listed)).encode()
        return files

    def replace_folder(self, files):
        shutil.rmtree(self.folder)
        self.put(self.folder, files)


class TestSameVersionThere(ThereAlready):
    def test_adopted_after_hashing_without_a_single_write(self):
        release = self.installer.release()
        self.assertEqual((release["version"], release["verified"]), ("v0.12.0", False))
        status = self.installer.model_status("27b-16gb")
        self.assertEqual((status["state"], status["installed"]), ("foreign", False), "on disk is not yet installed")
        before = self.picture(self.folder)
        job = self.run_job("27b-16gb")
        self.assertEqual(job.state, "done", job.error)
        self.assertTrue(self.installer.release()["verified"])
        status = self.installer.model_status("27b-16gb")
        self.assertEqual((status["state"], status["verified"], status["deletable"]), ("ready", True, False))
        self.assertEqual(self.picture(self.folder), before, "verified by reading only")
        self.assertEqual(self.asked(TARBALL), [])
        self.assertEqual(self.asked(M27["artifact"]), [], "files already on disk are hashed, not fetched")
        self.assertEqual(job.snapshot()["bytes_total"], 0, "nothing was downloaded")
        self.assertGreater(job.step_total, 0, "it was hashed")
        # "delete" has nothing of the app's to delete here
        self.assertEqual(self.installer.remove_model("27b-16gb")["removed"], 0)
        self.assertEqual(self.picture(self.folder), before)

    def test_incomplete_folder_is_completed_not_rewritten(self):
        (self.folder / M27["artifact"]).unlink()  # an `hf download` that stopped before the big file
        before = self.picture(self.folder)
        job = self.run_job("27b-16gb")
        self.assertEqual(job.state, "done", job.error)
        after = self.picture(self.folder)
        self.assertEqual(
            {n: v for n, v in after.items() if n != M27["artifact"]}, before, "what was there is as it was"
        )
        self.assertEqual(after[M27["artifact"]][0], self.files[M27["artifact"]])
        self.assertEqual([n for n in self.files if self.asked(n) and n != "SHA256SUMS"], [M27["artifact"]])
        self.assertEqual(self.installer.model_status("27b-16gb")["state"], "ready")
        # deleting removes the one file the app created; the folder and the rest are not the app's
        self.assertEqual(self.installer.remove_model("27b-16gb")["removed"], 1)
        self.assertEqual(self.picture(self.folder), before)

    def test_changed_after_verification_is_no_longer_ready(self):
        self.run_job("27b-16gb")
        artifact = self.folder / M27["artifact"]
        artifact.write_bytes(b"something else")
        status = self.installer.model_status("27b-16gb")
        self.assertEqual(
            (status["state"], status["installed"], status["mismatch"]), ("different", False, [M27["artifact"]])
        )


class TestOtherVersionThere(ThereAlready):
    """An older revision of the repository, same file names, in the model's folder."""

    def setUp(self):
        super().setUp()
        self.replace_folder(self.older())
        self.before = self.picture(self.folder)

    def refused(self, job):
        self.assertEqual((job.state, job.error["code"]), ("failed", "different_version"), job.error)
        self.assertIn(str(self.folder), job.error["message"])
        self.assertEqual(self.picture(self.folder), self.before, "not one byte or time of it changed")
        self.assertEqual(sorted(p.name for p in self.root.joinpath("models").iterdir()), ["27b-16gb"])

    def test_never_deleted_never_overwritten(self):
        job = self.run_job("27b-16gb")
        self.refused(job)
        for name in self.files:
            if name != "SHA256SUMS":  # (the repository's own SHA256SUMS is read to know the digests)
                self.assertEqual(self.asked(name), [], name + " was not downloaded")
        status = self.installer.model_status("27b-16gb")
        self.assertEqual(
            (status["state"], status["installed"], status["verified"], status["deletable"]),
            ("different", False, False, False),
        )
        self.assertEqual(status["mismatch"], sorted([M27["artifact"], "README.md", "SHA256SUMS"]))
        self.refused(self.run_job("27b-16gb"))  # asking again changes nothing
        self.assertEqual(
            self.installer.remove_model("27b-16gb")["removed"], 0
        )  # nor does "delete": nothing is the app's
        self.assertEqual(self.picture(self.folder), self.before)

    def test_a_matching_size_is_not_a_matching_file(self):
        # Only the big file differs, and it has the right size: the other files are those of the repository.
        self.replace_folder(dict(self.files, **{M27["artifact"]: bytes(len(self.files[M27["artifact"]]))}))
        self.before = self.picture(self.folder)
        job = self.run_job("27b-16gb")
        self.refused(job)
        snap = job.snapshot()
        self.assertEqual((snap["bytes_done"], snap["bytes_total"]), (0, 0), "nothing counted as downloaded")
        self.assertGreaterEqual(job.step_total, len(self.files[M27["artifact"]]), "the big file was hashed")
        status = self.installer.model_status("27b-16gb")
        self.assertEqual((status["verified"], status["mismatch"]), (False, [M27["artifact"]]))
        record = json.loads((self.root / ".capy-app" / "models" / "27b-16gb.json").read_text())
        self.assertEqual(record["files"], {}, "all or nothing: the files that did match are not adopted either")

    def test_new_folder_next_to_it(self):
        self.refused(self.run_job("27b-16gb"))
        job = self.run_job("27b-16gb", new_folder=True)
        self.assertEqual(job.state, "done", job.error)
        self.assertEqual(self.picture(self.folder), self.before, "the other version is still as it was")
        for name, data in self.files.items():
            if name != ".gitattributes":
                self.assertEqual((self.new / name).read_bytes(), data)
        status = self.installer.model_status("27b-16gb")
        self.assertEqual((status["state"], status["folder"], status["deletable"]), ("ready", str(self.new), True))
        self.assertEqual(self.installer.artifact("27b-16gb"), self.new / M27["artifact"], "what the engine is given")
        # deleting the app's download brings back the folder that was there, untouched
        self.installer.remove_model("27b-16gb")
        self.assertFalse(self.new.exists())
        self.assertEqual(self.picture(self.folder), self.before)
        self.assertEqual(self.installer.model_status("27b-16gb")["folder"], str(self.folder))

    def test_used_as_it_is_only_if_the_user_says_so(self):
        self.refused(self.run_job("27b-16gb"))
        self.assertEqual(self.installer.installed_models(), [])
        self.installer.accept_as_is("27b-16gb")
        status = self.installer.model_status("27b-16gb")
        self.assertEqual(
            (status["state"], status["installed"], status["verified"], status["found"]),
            ("as_is", True, False, "different"),
        )
        self.assertEqual(self.installer.artifact("27b-16gb"), self.folder / M27["artifact"])
        self.assertEqual(self.picture(self.folder), self.before)


class TestOwnFiles(InstallCase):
    """What the app downloaded is the app's to replace or delete, as long as nobody changed it."""

    def setUp(self):
        super().setUp()
        self.assertEqual(self.run_job("27b-16gb").state, "done")
        self.folder = self.root / "models" / "27b-16gb"

    def test_changed_by_someone_else_is_not_overwritten(self):
        artifact = self.folder / M27["artifact"]
        artifact.write_bytes(b"my own build of this file")
        before = self.picture(self.folder)
        self.assertEqual(self.installer.model_status("27b-16gb")["state"], "different")
        job = self.run_job("27b-16gb")
        self.assertEqual((job.state, job.error["code"]), ("failed", "different_version"))
        self.assertEqual(self.picture(self.folder), before)
        # delete: the app's untouched files go, the changed one and so the folder stay
        self.installer.remove_model("27b-16gb")
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()), [M27["artifact"]])
        self.assertEqual(artifact.read_bytes(), b"my own build of this file")

    def test_delete_leaves_what_the_user_added(self):
        (self.folder / "notes.txt").write_text("mine")
        result = self.installer.remove_model("27b-16gb")
        self.assertEqual(result["left"], ["notes.txt"])
        self.assertEqual((self.folder / "notes.txt").read_text(), "mine")

    def test_new_revision_replaces_only_its_own_files(self):
        newer = model_files(M27["artifact"])
        self.sources.add_model(M27["hf_repo"], newer, revision="b" * 40)
        (self.folder / "notes.txt").write_text("mine")
        job = self.run_job("27b-16gb")
        self.assertEqual(job.state, "done", job.error)
        self.assertEqual((self.folder / M27["artifact"]).read_bytes(), newer[M27["artifact"]])
        self.assertEqual((self.folder / "notes.txt").read_text(), "mine")
        self.assertEqual(self.installer.model_status("27b-16gb")["state"], "ready")


class TestLinksOut(ThereAlready):
    """The engine root is the boundary: the app writes and deletes nothing that a link leads out of it."""

    def setUp(self):
        super().setUp()
        self.elsewhere = Path(self.tmp.name) / "elsewhere" / "flash"
        self.elsewhere.parent.mkdir()
        shutil.move(str(self.folder), str(self.elsewhere))
        os.symlink(str(self.elsewhere), self.folder)  # models/27b-16gb -> somewhere outside the root
        self.before = self.picture(self.elsewhere)

    def untouched(self):
        self.assertEqual(self.picture(self.elsewhere), self.before)
        self.assertEqual(sorted(p.name for p in self.elsewhere.parent.iterdir()), ["flash"])

    def test_a_linked_model_folder_is_never_written_or_deleted(self):
        status = self.installer.model_status("27b-16gb")
        self.assertEqual((status["state"], status["installed"], status["deletable"]), ("outside", False, False))
        job = self.run_job("27b-16gb")
        self.assertEqual((job.state, job.error["code"]), ("failed", "outside"), job.error)
        self.untouched()
        self.assertEqual(self.asked(M27["artifact"]), [])
        with self.assertRaises(DownloadError) as c:
            self.installer.remove_model("27b-16gb")
        self.assertEqual(c.exception.code, "outside")
        self.untouched()
        self.assertTrue(self.folder.is_symlink(), "the link itself is left too")

    def test_it_can_be_used_as_it_is_or_left_for_a_new_folder(self):
        self.installer.accept_as_is("27b-16gb")
        status = self.installer.model_status("27b-16gb")
        self.assertEqual((status["state"], status["installed"], status["verified"]), ("as_is", True, False))
        job = self.run_job("27b-16gb", new_folder=True)
        self.assertEqual(job.state, "done", job.error)
        self.assertTrue((self.new / M27["artifact"]).is_file() and not self.new.is_symlink())
        self.assertEqual(self.installer.model_status("27b-16gb")["state"], "ready")
        self.untouched()

    def test_other_version_behind_the_link(self):
        shutil.rmtree(self.elsewhere)
        self.put(self.elsewhere, self.older())
        self.before = self.picture(self.elsewhere)
        job = self.run_job("27b-16gb")
        self.assertEqual((job.state, job.error["code"]), ("failed", "outside"))
        self.untouched()

    def test_models_folder_itself_linked_out(self):
        os.unlink(self.folder)
        os.rmdir(self.root / "models")
        outside = Path(self.tmp.name) / "elsewhere" / "all-models"
        outside.mkdir()
        os.symlink(str(outside), self.root / "models")
        job = self.run_job("27b-16gb")
        self.assertEqual((job.state, job.error["code"]), ("failed", "outside"), job.error)
        self.assertEqual(list(outside.iterdir()), [], "nothing was created through the link")

    def test_a_file_that_is_a_link_out(self):
        os.unlink(self.folder)
        shutil.move(str(self.elsewhere), str(self.folder))  # a real folder again...
        secret = Path(self.tmp.name) / "elsewhere" / "secret.txt"
        secret.write_text("not the app's")
        os.unlink(self.folder / "NOTICE")
        os.symlink(str(secret), self.folder / "NOTICE")  # ...with one file that is a link out
        self.assertEqual(self.installer.model_status("27b-16gb")["state"], "outside")
        job = self.run_job("27b-16gb")
        self.assertEqual((job.state, job.error["code"]), ("failed", "outside"))
        self.assertEqual(secret.read_text(), "not the app's")


class TestReleaseThere(InstallCase):
    """The same rule for releases: what install.sh (or the user) put there is not the app's to repair."""

    def cli_release(self, version="v0.12.0", tamper=False):
        release = self.root / ("capy-flash-" + version)
        for name, data in release_files().items():
            path = release / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(0o755)
        if tamper:
            (release / "bin" / "ninfer-serve").write_bytes(b"#!/bin/sh\necho my own build\n")
        if not (self.root / "current").is_symlink():
            os.symlink(str(release), self.root / "current")
        return release

    def test_foreign_release_that_fails_its_sums_is_left_alone(self):
        release = self.cli_release(tamper=True)
        before = self.picture(release)
        job = self.run_job()
        self.assertEqual((job.state, job.error["code"]), ("failed", "foreign_release"), job.error)
        self.assertEqual(self.picture(release), before)
        self.assertEqual(self.asked(TARBALL), [], "and it is not downloaded over it")

    def test_the_apps_own_release_is_repaired(self):
        self.assertEqual(self.run_job().state, "done")
        (self.root / "capy-flash-v0.12.0" / "bin" / "ninfer-serve").write_bytes(b"damaged")
        self.sources.requests.clear()
        job = self.run_job(update=True)
        self.assertEqual(job.state, "done", job.error)
        self.assertEqual(len(self.asked(TARBALL)), 1)
        self.assertTrue(self.installer.release()["verified"])

    def test_a_tarball_left_by_curl_is_not_overwritten(self):
        self.root.mkdir(parents=True)
        partial = self.root / TARBALL  # install.sh downloads to this very name
        partial.write_bytes(b"half a tarball from curl -C -")
        self.assertEqual(self.run_job().state, "done")
        self.assertEqual(partial.read_bytes(), b"half a tarball from curl -C -")

    def test_current_pointing_to_someone_elses_build_is_not_replaced(self):
        build = Path(self.tmp.name) / "my-build"
        build.mkdir()
        self.root.mkdir(parents=True)
        os.symlink(str(build), self.root / "current")
        job = self.run_job()
        self.assertEqual((job.state, job.error["code"]), ("failed", "foreign_current"), job.error)
        self.assertEqual(os.readlink(self.root / "current"), str(build))

    def test_update_moves_current_between_releases_of_this_root(self):
        old = self.cli_release()
        before = self.picture(old)
        self.sources.version = "v0.12.1"
        self.sources.release = tarball("v0.12.1", release_files())
        self.sources.release_sha = sha256(self.sources.release)
        job = self.run_job(update=True)
        self.assertEqual(job.state, "done", job.error)
        self.assertEqual(self.installer.release()["version"], "v0.12.1")
        self.assertEqual(self.picture(old), before, "the release install.sh made is still there, as it was")


if __name__ == "__main__":
    unittest.main()
