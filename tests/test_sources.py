"""A kb's sources: add, list, remove, and what `test` sees."""

from __future__ import annotations

import os

import pytest

from smartrag.errors import Conflict, InvalidRequest, NotFound, Unsupported
from smartrag.kb import Home, add_source, list_sources, remove_source
from smartrag.kb import sources as src


@pytest.fixture
def home(tmp_path):
    home = Home(tmp_path / "home")
    home.new("legal")
    return home


@pytest.fixture
def docs(tmp_path):
    root = tmp_path / "docs"
    files = {
        "就業規則.txt": "a",
        "sub/report.PDF": "bb",
        "sub/deeper/slides.pptx": "ccc",
        "old.xls": "d",
        "~$lock.docx": "e",
        ".git/objects/o": "f",
        "node_modules/pkg/index.html": "g",
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


# --- globs -------------------------------------------------------------------------


@pytest.mark.parametrize(("pattern", "path", "matches"), [
    ("**/*.pdf", "a.pdf", True),
    ("**/*.pdf", "x/y/A.PDF", True),
    ("*.pdf", "x/a.pdf", False),
    ("sub/*", "sub/a.txt", True),
    ("sub/*", "sub/deeper/a.txt", False),
    ("sub/**", "sub/deeper/a.txt", True),
    ("**/.git/**", "a/.git/objects/o", True),
    ("**/~$*", "dir/~$lock.docx", True),
    ("report-?.txt", "report-1.txt", True),
    ("[!a]*.txt", "b.txt", True),
    ("[!a]*.txt", "a.txt", False),
])
def test_glob(pattern, path, matches):
    assert bool(src.compile_glob(pattern).match(path)) is matches


def test_glob_rejects_absolute_patterns():
    with pytest.raises(ValueError):
        src.compile_glob("/abs/*.pdf")


# --- add ---------------------------------------------------------------------------


def test_add_and_list_includes_the_sink(home, docs):
    added = add_source(home, "legal", "kitei", str(docs), include=["**/*.txt"])
    assert added.source.root_exists and added.source.include == ["**/*.txt"]
    listed = list_sources(home, "legal")
    assert [(s.id, s.kind) for s in listed.sources] == [("kitei", "source"), ("sink", "sink")]
    assert "**/.git/**" in listed.builtin_exclude


def test_add_refuses_conflicts_and_reserved_ids(home, docs):
    add_source(home, "legal", "kitei", str(docs))
    with pytest.raises(Conflict):
        add_source(home, "legal", "kitei", str(docs / "sub"))
    with pytest.raises(Conflict, match="already source"):
        add_source(home, "legal", "again", str(docs))
    with pytest.raises(InvalidRequest, match="reserved"):
        add_source(home, "legal", "sink", str(docs))
    with pytest.raises(InvalidRequest):
        add_source(home, "legal", "Bad Id", str(docs))


def test_add_warns_about_overlap(home, docs):
    add_source(home, "legal", "all", str(docs))
    added = add_source(home, "legal", "sub", str(docs / "sub"))
    assert any("overlaps source 'all'" in w for w in added.warnings)


def test_add_checks_the_location(home, docs, tmp_path):
    with pytest.raises(InvalidRequest, match="does not exist"):
        add_source(home, "legal", "nas", str(tmp_path / "unmounted"))
    added = add_source(home, "legal", "nas", str(tmp_path / "unmounted"), allow_missing=True)
    assert not added.source.root_exists
    with pytest.raises(InvalidRequest, match="inside the kb"):
        add_source(home, "legal", "self", str(home.root / "legal" / "sink"))
    with pytest.raises(Unsupported):
        add_source(home, "legal", "bucket", "s3://bucket/docs")
    with pytest.raises(InvalidRequest, match="--exclude"):
        add_source(home, "legal", "x", str(docs), exclude=["/abs"])


def test_remote_kbs_have_no_sources(home, docs):
    home.new("factory", remote="https://rag.example.jp", token_env="T")
    with pytest.raises(InvalidRequest, match="managed by its server"):
        add_source(home, "factory", "x", str(docs))
    with pytest.raises(InvalidRequest, match="managed by its server"):
        list_sources(home, "factory")


# --- remove ------------------------------------------------------------------------


def test_remove(home, docs):
    add_source(home, "legal", "kitei", str(docs))
    removed = remove_source(home, "legal", "kitei")
    assert removed.source == "kitei" and removed.documents_removed is None
    assert [s.id for s in list_sources(home, "legal").sources] == ["sink"]
    assert (docs / "就業規則.txt").exists()
    with pytest.raises(NotFound):
        remove_source(home, "legal", "kitei")
    with pytest.raises(InvalidRequest, match="sink cannot be removed"):
        remove_source(home, "legal", "sink")


# --- test --------------------------------------------------------------------------


def test_test_reports_what_a_sync_would_see(home, docs):
    add_source(home, "legal", "kitei", str(docs), exclude=["**/old.*"])
    report = src.test_source(home, "legal", "kitei", sample=2)
    assert report.ok and report.root_kind == "directory"
    assert report.files_matched == 3
    assert report.by_extension == {".txt": 1, ".pdf": 1, ".pptx": 1}
    assert report.supported_files == 3 and report.unsupported_files == 0
    assert report.excluded_builtin == 1  # ~$lock.docx; .git and node_modules are pruned
    assert report.excluded_by_pattern == 1  # old.xls
    assert report.bytes_matched == 1 + 2 + 3
    assert len(report.sample) == 2


def test_test_include_and_recursion(home, docs):
    add_source(home, "legal", "pdfs", str(docs), include=["**/*.pdf"])
    report = src.test_source(home, "legal", "pdfs")
    assert report.files_matched == 1 and report.not_included >= 2
    add_source(home, "legal", "flat", str(docs / "sub"), recursive=False)
    assert src.test_source(home, "legal", "flat").files_matched == 1


def test_test_flags_unsupported_types_and_skips_symlinks(home, docs):
    (docs / "archive.zip").write_text("x", encoding="utf-8")
    (docs / "legacy.doc").write_text("x", encoding="utf-8")
    os.symlink(docs / "就業規則.txt", docs / "link.txt")
    add_source(home, "legal", "kitei", str(docs))
    report = src.test_source(home, "legal", "kitei")
    assert report.unsupported_files == 2  # archive.zip, legacy.doc
    assert report.symlinks_skipped == 1
    assert any("cannot extract (.doc, .zip)" in p for p in report.problems)


def test_tabular_files_are_supported(home, tmp_path):
    root = tmp_path / "tables"
    root.mkdir()
    for name in ("見積.csv", "export.tsv", "売上.xlsx", "old.xls", "big.xlsb", "macro.xlsm"):
        (root / name).write_text("x", encoding="utf-8")
    add_source(home, "legal", "tables", str(root))
    report = src.test_source(home, "legal", "tables")
    assert report.files_matched == 6 and report.supported_files == 6


def test_test_missing_root_is_not_ok(home, tmp_path):
    add_source(home, "legal", "nas", str(tmp_path / "gone"), allow_missing=True)
    report = src.test_source(home, "legal", "nas")
    assert not report.ok and report.root_kind == "missing"


def test_test_a_single_file_and_the_sink(home, docs):
    add_source(home, "legal", "one", str(docs / "就業規則.txt"))
    report = src.test_source(home, "legal", "one")
    assert report.root_kind == "file" and report.files_matched == 1
    sink = src.test_source(home, "legal", "sink")
    assert sink.ok and sink.files_matched == 0
