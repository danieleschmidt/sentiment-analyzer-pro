import sys
import subprocess
import pytest


def test_cli_help():
    result = subprocess.run([sys.executable, '-m', 'src.cli', '--help'], capture_output=True)
    assert result.returncode == 0
    assert b'Sentiment Analyzer CLI' in result.stdout


def test_cli_train_creates_model(tmp_path):
    pytest.importorskip('sklearn')
    pytest.importorskip('joblib')
    model_file = tmp_path / 'model.joblib'
    subprocess.run([
        sys.executable,
        '-m',
        'src.cli',
        'train',
        '--csv',
        'data/sample_reviews.csv',
        '--model',
        str(model_file),
    ], check=True)
    assert model_file.exists()
