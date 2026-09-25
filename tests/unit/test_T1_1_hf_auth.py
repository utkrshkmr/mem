"""T1.1: hf_auth resolution order, stripping, errors, mode warning, redaction on handlers."""
import logging
import os

import pytest

from kmatters import hf_auth
from kmatters.logging_utils import setup_logging

FAKE = "hf_FAKEtokenForTests0123456789"


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolated cwd and HOME, no HF_TOKEN_FILE, HF_TOKEN restored afterwards."""
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("HF_TOKEN_FILE", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(hf_auth, "_TOKEN", None)
    return home, cwd, tmp_path


def _write(p, text, mode=0o600):
    p.write_text(text)
    os.chmod(p, mode)
    return p


def test_T1_1_resolution_order(iso, monkeypatch, record):
    home, cwd, tmp = iso
    _write(home / "hf-tok", "hf_home\n")
    assert hf_auth.load_hf_token() == "hf_home"
    _write(cwd / "hf-tok", "hf_cwd\n")
    assert hf_auth.load_hf_token() == "hf_cwd"
    envf = _write(tmp / "custom-tok", "hf_env\n")
    monkeypatch.setenv("HF_TOKEN_FILE", str(envf))
    assert hf_auth.load_hf_token() == "hf_env"
    assert os.environ["HF_TOKEN"] == "hf_env"
    record("T1.1", resolution_order="env > ./hf-tok > ~/hf-tok", passed=True)


def test_T1_1_whitespace_stripped(iso):
    home, cwd, _ = iso
    _write(cwd / "hf-tok", f"  \n {FAKE} \n\n")
    assert hf_auth.load_hf_token() == FAKE


def test_T1_1_missing_raises_clear_error(iso):
    with pytest.raises(FileNotFoundError) as e:
        hf_auth.load_hf_token()
    assert str(e.value) == ("Hugging Face token file not found. Put your token in ./hf-tok "
                            "(one line, chmod 600) or set HF_TOKEN_FILE.")


def test_T1_1_empty_file_raises(iso):
    _, cwd, _ = iso
    _write(cwd / "hf-tok", "  \n")
    with pytest.raises(ValueError):
        hf_auth.load_hf_token()


def test_T1_1_mode_warning(iso, caplog):
    _, cwd, _ = iso
    _write(cwd / "hf-tok", FAKE, mode=0o644)
    with caplog.at_level(logging.WARNING):
        hf_auth.load_hf_token()
    assert any("readable by group/others" in r.getMessage() for r in caplog.records)
    caplog.clear()
    os.chmod(cwd / "hf-tok", 0o600)
    with caplog.at_level(logging.WARNING):
        hf_auth.load_hf_token()
    assert not any("readable by group/others" in r.getMessage() for r in caplog.records)


def test_T1_1_redaction_on_handlers(iso, caplog, capfd):
    _, cwd, tmp = iso
    _write(cwd / "hf-tok", FAKE)
    hf_auth.load_hf_token()
    log_file = tmp / "run.log"
    root = setup_logging(log_file)
    try:
        assert all(any(isinstance(f, hf_auth.RedactFilter) for f in h.filters)
                   for h in root.handlers if getattr(h, "_kmatters", False))
        caplog.handler.addFilter(hf_auth.RedactFilter())
        child = logging.getLogger("kmatters.some.child")      # logger filters would not apply here
        child.info("token is %s", FAKE)
        child.warning(f"inline {FAKE} again")
        for h in root.handlers:
            h.flush()
        file_text = log_file.read_text()
        err = capfd.readouterr().err
        assert FAKE not in file_text and "hf_***" in file_text
        assert FAKE not in err and "hf_***" in err
        assert FAKE not in caplog.text and "hf_***" in caplog.text
    finally:
        setup_logging(None)
