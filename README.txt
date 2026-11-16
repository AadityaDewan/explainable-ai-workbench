XAI WORKBENCH
=============
A web workbench that applies SHAP, LIME, permutation importance, partial dependence,
counterfactual explanations and Grad-CAM to trained models, quantifies agreement between
methods, tests stability under small input changes, finds cases where methods disagree,
and runs a small user study on explanation understandability and trust.

FEATURES
- Two trained models on the breast-cancer dataset (random forest, logistic regression).
- Local explanation per test case: SHAP, LIME, permutation importance (global),
  single-feature counterfactuals (plus a counterfactual sensitivity score).
- Agreement: Spearman rank correlation of absolute attributions and top-5 overlap for every method pair.
- Stability: Gaussian noise (fraction of feature SD), compared with a "seed-only" baseline so
  method randomness is separated from input sensitivity; prediction flips are counted.
- Disagreement: scans 20 test cases, ranks the lowest SHAP-vs-LIME agreement, and relates it to model confidence.
- Partial dependence plot for any feature.
- Grad-CAM on a small CNN trained on the scikit-learn digits images, with noise-stability check (needs PyTorch).
- User study: ratings (understandability 1-5, trust 1-5, comprehension check) stored in SQLite, with a summary table.
- Discussion tab on which explanations practitioners can trust.

TECHNOLOGIES: Python 3.10-3.12, Flask, scikit-learn, NumPy, SciPy, SQLite, plain HTML/JS. Optional: PyTorch.

FOLDER STRUCTURE
 app.py              backend, models, explanation methods, API, user-study DB
 templates/index.html  single-page UI
 requirements.txt, .env.example, .gitignore, LICENSE.txt, SETUP_COMMANDS.txt
 study.db            created automatically on first study submission

PREREQUISITES: Python 3.10+ (3.12 recommended), a modern browser, ~500 MB disk (2+ GB if you add PyTorch).
Installation per OS: see SETUP_COMMANDS.txt (Windows, macOS, Linux sections, copy-paste ready).

RUN (after installing; inside the project folder with the virtual environment active)
 python app.py        then open http://127.0.0.1:5000
STOP: press Ctrl+C in the terminal.
LOGIN: none required.

SUGGESTED USER-STUDY PROTOCOL
 Recruit 5-10 participants. Show each the same test case with one explanation type at a time
 (SHAP, LIME, counterfactual, ...), ask them to name the most influential feature, then
 submit ratings in the "User study" tab. Compare the summary table across methods.

ASSUMPTIONS
- SHAP and LIME are implemented from scratch (permutation-sampling Shapley estimates and a
  weighted-ridge local surrogate on Gaussian perturbations) rather than via the shap/lime packages,
  to keep installation light and results reproducible. Values are approximations and are noisy by design;
  the stability tab quantifies this.
- Tabular methods use the breast-cancer dataset (30 features). Grad-CAM needs an image model, so it uses a
  small CNN on 8x8 digit images.
- Agreement uses absolute attribution magnitudes; the permutation importance is global, so it is the same for every case.
- The study is a lightweight in-app questionnaire, not a controlled experiment; statistics are descriptive only.

TROUBLESHOOTING
- "python not found": install Python and reopen the terminal (Windows: "py" may work instead of "python").
- PowerShell refuses to activate venv: Set-ExecutionPolicy -Scope Process Bypass
- Port in use: set PORT=8000 (see SETUP_COMMANDS.txt).
- Grad-CAM tab shows "PyTorch not installed": pip install torch, restart the app.
- Stability and disagreement tabs take 10-40 seconds; this is normal.
- pip build errors on very new Python versions: use Python 3.12.
