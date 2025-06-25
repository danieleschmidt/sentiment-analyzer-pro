import pathlib


def test_readme_documents_new_features():
    readme = pathlib.Path('README.md').read_text().lower()
    assert 'naive bayes' in readme
    assert '--lemmatize' in readme


def test_tests_cover_new_features():
    required = ['tests/test_nb_model.py', 'tests/test_preprocessing.py', 'tests/test_cross_validation.py']
    for path in required:
        assert pathlib.Path(path).is_file(), f"Missing {path}"
