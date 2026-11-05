"""The technical report's number check (analysis/verify_report.py) passes and catches a changed number."""
from pathlib import Path

from analysis import verify_report as vr


def test_report_numbers_pass(capsys):
    assert vr.main([]) == 0
    assert ", 0 failed." in capsys.readouterr().out


def test_changed_number_is_caught(tmp_path, monkeypatch, capsys):
    tex = (vr.ROOT / vr.REPORT).read_text().replace("0.52~m/s for a 0.6~m/s command", "0.53~m/s for a 0.6~m/s command")
    fake = tmp_path / "report_en.tex"
    fake.write_text(tex)
    monkeypatch.setattr(vr, "REPORT", str(fake))
    assert vr.main([]) == 1
    assert "0.53: unmapped" in capsys.readouterr().out


def test_paths_and_code_are_not_claims():
    tex = r"\begin{document} see \code{results/walk_sweep_cpu} and \cmd{pip install -e .} at 0.52 m/s"
    numbers, tikz = vr.numbersIn(tex)
    assert numbers == ["0.52"] and tikz == []
    assert Path(vr.ROOT / vr.REPORT).exists()
