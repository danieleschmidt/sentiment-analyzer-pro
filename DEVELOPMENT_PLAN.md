# Development Plan

The following checklist outlines the major tasks required to build the Sentiment Analyzer Pro project. Items are organized in approximate chronological order and grouped by milestone.

## Milestone 1: Project Setup
- [x] Initialize Git repository and configure remote
- [x] Create Python virtual environment
- [x] Add `requirements.txt` with base dependencies
- [x] Set up initial project structure as described in `README.md`

## Milestone 2: Data Collection and Preprocessing
- [x] Gather and clean initial dataset(s)
- [x] Implement text preprocessing module (`src/preprocessing.py`)
- [x] Create unit tests for preprocessing
- [x] Write documentation for data handling

## Milestone 3: Baseline Sentiment Classifier
- [x] Implement baseline model (e.g., Naive Bayes or Logistic Regression) in `src/models.py`
- [x] Implement training script (`src/train.py`)
- [x] Implement prediction script (`src/predict.py`)
- [x] Add evaluation module (`src/evaluate.py`)
- [x] Write unit tests for model training and prediction

## Milestone 4: Advanced Modeling
- [x] Experiment with RNN/LSTM models
- [x] Explore Transformer-based approaches
- [x] Compare models and record results
- [x] Update documentation with findings

## Milestone 5: Aspect-Based Sentiment Analysis
- [x] Design approach for aspect extraction
- [x] Implement aspect-based sentiment component
- [x] Add tests for aspect-based methods
- [x] Document usage and limitations

## Milestone 6: Robust Evaluation and Deployment
- [x] Create comprehensive evaluation suite (confusion matrices, error analysis)
- [x] Build simple CLI or web demo for predictions
- [x] Prepare model for deployment (packaging, Dockerfile, etc.)
- [x] Update Dockerfile to use optional extras for the web API
- [x] Ensure continuous integration and automated testing

## Milestone 7: Web API
- [x] Implement Flask-based prediction server
- [x] Add tests for web server
- [x] Document web server usage

## Milestone 8: Packaging and Release
- [x] Split heavy dependencies into optional extras
- [x] Publish package to PyPI
- [x] Provide installation instructions for extras

## Milestone 9: API Enhancements
- [x] Add health check endpoint to the web server
- [x] Document the root endpoint in the README

## Milestone 10: Version Reporting
- [x] Expose package version via `/version` endpoint
- [x] Document the endpoint in the README

## Milestone 11: CLI Enhancements
- [x] Add `version` command to the CLI
- [x] Document the new command in the README

## Milestone 12: CLI Polishing
- [x] Provide `--version` flag for quick access
- [x] Review CLI usability and options

## Milestone 13: Data Utilities
- [x] Add CLI command for preprocessing datasets
- [x] Document preprocessing usage in the README

---

## Milestone 14: Dataset Splitting
- [x] Implement CLI command for splitting datasets
- [x] Document dataset splitting usage in the README
- [x] Add tests for the new CLI command

## Milestone 15: Dataset Statistics
- [x] Add CLI command to summarize datasets
- [x] Document dataset summary usage in the README
- [x] Write tests for the summary command

## Milestone 16: Dataset Insights
 - [x] Extend the summary command to list the most frequent words
 - [x] Document the new option in the README
 - [x] Add tests covering word frequency output

---

Continue refining tasks as the project evolves. Check items off as they are completed to track progress.
