# Contributing

Thanks for your interest in improving this project. It's a small, single-pipeline
codebase, so the process is intentionally lightweight.

## Getting set up

```bash
git clone <repo-url>
cd churn-prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place a copy of `European_Bank.csv` in `data/` (not tracked in git — see
[README § Data](README.md#data)). Then run the pipeline once to generate the
model artifacts the app needs:

```bash
cd src
python3 pipeline.py
```

## Project layout

See [README § Project structure](README.md#project-structure) for what lives
where. In short: `src/` holds one module per pipeline stage (data prep →
feature engineering → training → thresholding → scoring → explainability →
scenario simulation), `pipeline.py` orchestrates them, and `app.py` is the
Streamlit dashboard on top.

## Making a change

1. **Open an issue first** for anything beyond a small fix — bugs, typos, and
   docs improvements can go straight to a PR.
2. **Branch from `main`**: `git checkout -b fix/short-description`.
3. **Keep changes scoped.** If you're touching `src/`, prefer editing the
   relevant module directly rather than adding logic to `pipeline.py` or
   `app.py` — both are meant to stay thin orchestration/UI layers.
4. **Re-run the pipeline** after changing anything in `src/data_prep.py`,
   `feature_engineering.py`, or `train.py`, since these affect the persisted
   model/scaler/column artifacts that `app.py` depends on. A model trained
   before your change is not compatible with code after it.
5. **Update the README's Results table** if your change affects model
   performance (new features, different hyperparameters, a different
   train/test split, etc.) — the numbers there are meant to reflect the
   current code, not a one-time run.

## Style

- Plain functions over classes; each `src/` module should stay importable and
  testable on its own (see how `pipeline.py` composes them).
- No hard-coded paths inside `src/*.py` — pass paths in via function
  arguments or `pipeline.py`'s CLI flags.
- Keep `app.py` free of modeling logic; it should only load artifacts, collect
  inputs, and call into `src/`.

## Reporting bugs / requesting features

Open a GitHub issue with:
- What you ran (command or dashboard tab)
- What you expected vs. what happened
- Python version and OS, if it looks environment-related

## Code of conduct

Be respectful and constructive. Disagreements about approach are fine and
expected — assume good faith.
