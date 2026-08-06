# tests/ingest/test_olx_loader.py
from pathlib import Path
from realestate.ingest.olx_loader import load_olx_html_directory
from tests.ingest.test_olx_parser import SAMPLE_HTML

MALFORMED_HTML = "<html><body>no prerendered state here</body></html>"


def test_loads_valid_html_files(tmp_path: Path):
    (tmp_path / "304473136.html").write_text(SAMPLE_HTML, encoding="utf-8")

    listings, failures = load_olx_html_directory(tmp_path)

    assert len(listings) == 1
    assert listings[0].external_id == "304473136"
    assert failures == []


def test_quarantines_malformed_files_instead_of_raising(tmp_path: Path):
    (tmp_path / "304473136.html").write_text(SAMPLE_HTML, encoding="utf-8")
    (tmp_path / "999.html").write_text(MALFORMED_HTML, encoding="utf-8")

    listings, failures = load_olx_html_directory(tmp_path)

    assert len(listings) == 1
    assert len(failures) == 1
    assert failures[0][0].name == "999.html"
