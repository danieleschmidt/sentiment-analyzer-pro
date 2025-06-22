import sys
import subprocess
import pytest


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "--help"], capture_output=True
    )
    assert result.returncode == 0
    assert b"Sentiment Analyzer CLI" in result.stdout


def test_cli_requires_subcommand():
    import src.cli as cli
    with pytest.raises(SystemExit):
        cli.main([])


def test_cli_verbose_flag(capsys):
    import src.cli as cli
    cli.main(["-v", "version"])
    out, _ = capsys.readouterr()
    assert "Verbosity level: 1" in out


def test_cli_train_creates_model(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    model_file = tmp_path / "model.joblib"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "train",
            "--csv",
            "data/sample_reviews.csv",
            "--model",
            str(model_file),
        ],
        check=True,
    )
    assert model_file.exists()


def test_cli_serve_calls_webapp(monkeypatch):
    pytest.importorskip("flask")
    called = {}

    def fake_main(args):
        called["args"] = args

    import types, sys, src

    fake = types.SimpleNamespace(main=fake_main)
    monkeypatch.setitem(sys.modules, "src.webapp", fake)
    monkeypatch.setattr(src, "webapp", fake, raising=False)
    from src import cli

    cli.main(["serve", "--model", "foo", "--host", "127.0.0.1", "--port", "1234"])
    assert called["args"] == ["--model", "foo", "--host", "127.0.0.1", "--port", "1234"]


def test_cli_analyze_prints_errors(monkeypatch, tmp_path, capsys):
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")

    import pandas as pd

    csv = tmp_path / "data.csv"
    pd.DataFrame({"text": ["good", "bad"], "label": ["pos", "pos"]}).to_csv(csv, index=False)

    called = {}

    def fake_build_model():
        class Dummy:
            def fit(self, X, y):
                return self

            def predict(self, X):
                return ["neg"] * len(X)

        return Dummy()

    def fake_analyze(texts, labels, preds):
        called["preds"] = list(preds)
        return pd.DataFrame({"text": ["bad"], "true": ["pos"], "predicted": ["neg"]})

    import importlib
    import src.models as models

    evaluate_module = importlib.import_module("src.evaluate")
    monkeypatch.setattr(models, "build_model", fake_build_model)
    monkeypatch.setattr(evaluate_module, "analyze_errors", fake_analyze)

    import src.cli as cli
    importlib.reload(cli)

    cli.main(["analyze", str(csv)])
    out, _ = capsys.readouterr()

    assert "bad" in out
    assert called["preds"] == ["neg", "neg"]


def test_cli_preprocess_writes_file(tmp_path):
    pytest.importorskip("pandas")

    import pandas as pd

    csv = tmp_path / "raw.csv"
    pd.DataFrame({"text": ["Hello WORLD!"]}).to_csv(csv, index=False)
    out = tmp_path / "clean.csv"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "preprocess",
            str(csv),
            "--out",
            str(out),
        ],
        check=True,
    )

    df = pd.read_csv(out)
    assert df.iloc[0]["text"] == "hello world"


def test_cli_split_creates_files(tmp_path):
    pytest.importorskip("pandas")

    import pandas as pd

    csv = tmp_path / "all.csv"
    pd.DataFrame(
        {"text": ["a", "b", "c", "d"], "label": ["p", "n", "p", "n"]}
    ).to_csv(csv, index=False)

    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "split",
            str(csv),
            "--train",
            str(train),
            "--test",
            str(test),
            "--ratio",
            "0.25",
        ],
        check=True,
    )

    assert train.exists() and test.exists()
    train_rows = pd.read_csv(train).shape[0]
    test_rows = pd.read_csv(test).shape[0]
    assert train_rows + test_rows == 4


def test_cli_summary_outputs_stats(tmp_path):
    pytest.importorskip("pandas")

    import pandas as pd

    csv = tmp_path / "data.csv"
    pd.DataFrame(
        {"text": ["good movie", "bad movie", "meh"], "label": ["pos", "neg", "neu"]}
    ).to_csv(csv, index=False)

    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "summary", str(csv), "--top", "2"],
        capture_output=True,
        text=True,
        check=True,
    )

    out = result.stdout
    assert "Rows: 3" in out
    assert "Avg words: 1.67" in out
    assert "pos: 1" in out
    assert "movie: 2" in out


def test_cli_summary_top_strips_punctuation(tmp_path):
    pytest.importorskip("pandas")

    import pandas as pd

    csv = tmp_path / "data.csv"
    pd.DataFrame({"text": ["good movie!", "bad, movie.", "meh??"]}).to_csv(
        csv, index=False
    )

    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "summary", str(csv), "--top", "1"],
        capture_output=True,
        text=True,
        check=True,
    )

    out = result.stdout
    assert "movie: 2" in out


def test_cli_summary_top_respects_limit(tmp_path):
    pytest.importorskip("pandas")

    import pandas as pd

    csv = tmp_path / "data.csv"
    pd.DataFrame({"text": ["apple banana", "banana", "cherry apple"]}).to_csv(
        csv, index=False
    )

    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "summary", str(csv), "--top", "2"],
        capture_output=True,
        text=True,
        check=True,
    )

    out = result.stdout
    assert "apple: 2" in out
    assert "banana: 2" in out
    assert "cherry" not in out


def test_cli_version(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, 'importlib.metadata', None)
    import src.cli as cli
    monkeypatch.setattr(cli.metadata, 'version', lambda name: '9.9.9', raising=False)
    monkeypatch.setattr(cli.metadata, 'PackageNotFoundError', Exception, raising=False)
    cli.main(['version'])
    out, _ = capsys.readouterr()
    assert out.strip() == '9.9.9'


def test_cli_global_version(monkeypatch, capsys):
    import src.cli as cli
    monkeypatch.setattr(cli.metadata, 'version', lambda name: '1.2.3', raising=False)
    monkeypatch.setattr(cli.metadata, 'PackageNotFoundError', Exception, raising=False)
    with pytest.raises(SystemExit):
        cli.main(['--version'])
    out, _ = capsys.readouterr()
    assert out.strip() == '1.2.3'
