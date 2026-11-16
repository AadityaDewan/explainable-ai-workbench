"""XAI Workbench: SHAP, LIME, permutation importance, PDP, counterfactuals, Grad-CAM,
agreement, stability, disagreement analysis and a small user study."""
import os, sqlite3, time
import numpy as np
from flask import Flask, jsonify, request, render_template
from scipy.stats import spearmanr
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

app = Flask(__name__)
DB = os.environ.get("DB_PATH", "study.db")

# ---------- data & models ----------
D = load_breast_cancer()
X, y, FN = D.data, D.target, list(D.feature_names)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
SD = Xtr.std(0); NF = X.shape[1]
MODELS = {
    "random_forest": RandomForestClassifier(60, random_state=0).fit(Xtr, ytr),
    "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr),
}
_perm = {}

def proba(m): return lambda Z: m.predict_proba(Z)[:, 1]

# ---------- explanation methods ----------
def shap_values(f, x, rng, n=15, bg=None):
    """Permutation-sampling Shapley estimate (baseline = random training rows)."""
    bg = Xtr if bg is None else bg
    phi = np.zeros(NF)
    for _ in range(n):
        perm = rng.permutation(NF); cur = bg[rng.integers(len(bg))].copy(); rows = [cur.copy()]
        for j in perm:
            cur = cur.copy(); cur[j] = x[j]; rows.append(cur)
        phi[perm] += np.diff(f(np.array(rows)))
    return phi / n

def lime_values(f, x, rng, n=500):
    """LIME-style: Gaussian perturbations, exponential kernel, weighted ridge surrogate."""
    Z = x + rng.normal(0, 1, (n, NF)) * SD
    U = (Z - x) / SD
    w = np.exp(-(np.linalg.norm(U, axis=1) ** 2) / (2 * (0.75 * np.sqrt(NF)) ** 2))
    return Ridge(alpha=1.0).fit(U, f(Z), sample_weight=w).coef_

def counterfactual(m, x):
    """Single-feature counterfactual search over training quantiles."""
    p0 = m.predict_proba(x[None])[0, 1]; want1 = p0 < .5
    imp = np.zeros(NF); found = []
    for j in range(NF):
        grid = np.quantile(Xtr[:, j], np.linspace(.01, .99, 25))
        Z = np.tile(x, (len(grid), 1)); Z[:, j] = grid
        ok = (m.predict_proba(Z)[:, 1] >= .5) == want1
        if ok.any():
            d = np.abs(grid - x[j]); k = np.argmin(np.where(ok, d, np.inf)); cost = d[k] / SD[j]
            imp[j] = 1 / (cost + .1)
            found.append({"feature": FN[j], "from": float(x[j]), "to": float(grid[k]), "cost_sd": float(cost)})
    return imp, sorted(found, key=lambda r: r["cost_sd"])[:5]

def perm_imp(name):
    if name not in _perm:
        r = permutation_importance(MODELS[name], Xte, yte, n_repeats=8, random_state=0)
        _perm[name] = r.importances_mean
    return _perm[name]

def topk(v, k=5): return set(np.argsort(-np.abs(v))[:k].tolist())
def rho(a, b):
    r = spearmanr(np.abs(a), np.abs(b))[0]
    return 0.0 if np.isnan(r) else float(r)

def explain(name, x, seed=0, ns=15, nl=500):
    m = MODELS[name]; f = proba(m); rng = np.random.default_rng(seed)
    cf, cfl = counterfactual(m, x)
    return dict(shap=shap_values(f, x, rng, ns), lime=lime_values(f, x, rng, nl), perm=perm_imp(name), cf=cf), cfl

def get_inst(idx):
    if not 0 <= idx < len(Xte): raise ValueError(f"idx must be 0..{len(Xte)-1}")
    return Xte[idx]

def err(e, code=400): return jsonify(error=str(e)), code

# ---------- routes ----------
@app.route("/")
def home(): return render_template("index.html")

@app.route("/api/meta")
def meta():
    return jsonify(models=list(MODELS), n_test=len(Xte), features=FN, has_torch=HAS_TORCH,
                   accuracy={k: float(m.score(Xte, yte)) for k, m in MODELS.items()})

def args():
    name = request.args.get("model", "random_forest")
    if name not in MODELS: raise ValueError("unknown model")
    return name, int(request.args.get("idx", 0))

@app.route("/api/explain")
def api_explain():
    try:
        name, idx = args(); x = get_inst(idx); E, cfl = explain(name, x)
        keys = ["shap", "lime", "perm", "cf"]; pairs = []
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = E[keys[i]], E[keys[j]]
                pairs.append(dict(a=keys[i], b=keys[j], spearman=rho(a, b), top5_overlap=len(topk(a) & topk(b)) / 5))
        return jsonify(prob=float(MODELS[name].predict_proba(x[None])[0, 1]), true=int(yte[idx]), features=FN,
                       values=x.tolist(), expl={k: v.tolist() for k, v in E.items()}, counterfactuals=cfl, agreement=pairs)
    except ValueError as e: return err(e)

@app.route("/api/stability")
def api_stability():
    try:
        name, idx = args(); x = get_inst(idx); eps = float(request.args.get("eps", .05)); n = int(request.args.get("n", 10))
        if not (0 < eps <= 1 and 1 <= n <= 30): raise ValueError("eps in (0,1], n in 1..30")
        m = MODELS[name]; base, _ = explain(name, x, 0, 10, 300); rng = np.random.default_rng(1)
        rep, _ = explain(name, x, 99, 10, 300)  # same input, new seed = method's own sampling noise
        out = {"eps_sd": eps, "n": n, "baseline_seed_only": {k: rho(base[k], rep[k]) for k in ("shap", "lime")}}
        res = {"shap": [], "lime": [], "shap_top5": [], "lime_top5": [], "flipped": 0}
        for i in range(n):
            x2 = x + rng.normal(0, eps, NF) * SD
            res["flipped"] += int((m.predict(x2[None])[0]) != m.predict(x[None])[0])
            E, _ = explain(name, x2, i + 5, 10, 300)
            for k in ("shap", "lime"):
                res[k].append(rho(base[k], E[k])); res[k + "_top5"].append(len(topk(base[k]) & topk(E[k])) / 5)
        out["perturbed"] = {k: {"mean": float(np.mean(v)), "min": float(np.min(v))} for k, v in res.items() if k != "flipped"}
        out["prediction_flips"] = res["flipped"]
        return jsonify(out)
    except ValueError as e: return err(e)

@app.route("/api/disagree")
def api_disagree():
    try:
        name = request.args.get("model", "random_forest"); n = min(int(request.args.get("n", 20)), len(Xte)); rows = []
        for i in range(n):
            E, _ = explain(name, Xte[i], i, 8, 300); p = float(MODELS[name].predict_proba(Xte[i][None])[0, 1])
            rows.append(dict(idx=i, prob=p, spearman=rho(E["shap"], E["lime"]), overlap=len(topk(E["shap"]) & topk(E["lime"])) / 5,
                             shap_top3=[FN[j] for j in np.argsort(-np.abs(E["shap"]))[:3]],
                             lime_top3=[FN[j] for j in np.argsort(-np.abs(E["lime"]))[:3]]))
        corr = spearmanr([abs(r["prob"] - .5) for r in rows], [r["spearman"] for r in rows])[0]
        return jsonify(worst=sorted(rows, key=lambda r: r["spearman"])[:5], mean_spearman=float(np.mean([r["spearman"] for r in rows])),
                       confidence_vs_agreement=None if np.isnan(corr) else float(corr), n=n)
    except ValueError as e: return err(e)

@app.route("/api/pdp")
def api_pdp():
    try:
        name = request.args.get("model", "random_forest"); fe = request.args.get("feature", FN[0])
        if fe not in FN: raise ValueError("unknown feature")
        j = FN.index(fe); grid = np.linspace(*np.quantile(Xtr[:, j], [.02, .98]), 20); B = Xtr[:100]; m = MODELS[name]; out = []
        for g in grid:
            Z = B.copy(); Z[:, j] = g; out.append(float(m.predict_proba(Z)[:, 1].mean()))
        return jsonify(grid=grid.tolist(), pdp=out, feature=fe)
    except ValueError as e: return err(e)

# ---------- Grad-CAM (optional, PyTorch) ----------
_cnn = {}
def cnn():
    if _cnn: return _cnn
    dg = load_digits(); Xi = torch.tensor(dg.images / 16., dtype=torch.float32)[:, None]; yi = torch.tensor(dg.target)
    torch.manual_seed(0)
    c1, c2, fc = nn.Conv2d(1, 8, 3, padding=1), nn.Conv2d(8, 16, 3, padding=1), nn.Linear(16, 10)
    opt = torch.optim.Adam([*c1.parameters(), *c2.parameters(), *fc.parameters()], 0.01); ntr = 1400
    for _ in range(40):
        A = torch.relu(c2(torch.relu(c1(Xi[:ntr])))); loss = nn.functional.cross_entropy(fc(A.mean((2, 3))), yi[:ntr])
        opt.zero_grad(); loss.backward(); opt.step()
    _cnn.update(c1=c1, c2=c2, fc=fc, X=Xi, y=yi, ntr=ntr); return _cnn

def gradcam(img, c):
    A = torch.relu(c["c2"](torch.relu(c["c1"](img)))); A.retain_grad()
    logits = c["fc"](A.mean((2, 3))); k = int(logits.argmax()); logits[0, k].backward()
    w = A.grad.mean((2, 3), keepdim=True); cam = torch.relu((w * A).sum(1))[0].detach().numpy()
    return k, cam / (cam.max() + 1e-9)

@app.route("/api/gradcam")
def api_gradcam():
    if not HAS_TORCH: return err("PyTorch not installed. Run: pip install torch (see README).", 501)
    try:
        idx = int(request.args.get("idx", 0)); c = cnn()
        if not 0 <= idx < len(c["X"]) - c["ntr"]: raise ValueError(f"idx must be 0..{len(c['X'])-c['ntr']-1}")
        img = c["X"][c["ntr"] + idx][None]; k, cam = gradcam(img.clone(), c)
        noisy = (img + 0.03 * torch.randn_like(img)).clamp(0, 1); k2, cam2 = gradcam(noisy, c)
        return jsonify(image=img[0, 0].tolist(), cam=cam.tolist(), cam_noisy=cam2.tolist(), pred=k, pred_noisy=k2,
                       true=int(c["y"][c["ntr"] + idx]), cam_spearman=float(np.nan_to_num(spearmanr(cam.ravel(), cam2.ravel())[0])))
    except ValueError as e: return err(e)

# ---------- user study ----------
def db():
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS study(id INTEGER PRIMARY KEY, ts REAL, participant TEXT, method TEXT, "
                "understand INT, trust INT, correct INT, comment TEXT)")
    return con

@app.route("/api/study", methods=["POST"])
def study_post():
    d = request.get_json(silent=True) or {}
    try:
        p = str(d.get("participant", "")).strip()[:40]; me = d.get("method")
        u, t = int(d.get("understand")), int(d.get("trust"))
        if not p: raise ValueError("Participant ID required")
        if me not in ("shap", "lime", "cf", "perm", "pdp", "gradcam"): raise ValueError("Invalid method")
        if not (1 <= u <= 5 and 1 <= t <= 5): raise ValueError("Ratings must be 1-5")
        with db() as con:
            con.execute("INSERT INTO study(ts,participant,method,understand,trust,correct,comment) VALUES(?,?,?,?,?,?,?)",
                        (time.time(), p, me, u, t, int(bool(d.get("correct"))), str(d.get("comment", ""))[:300]))
        return jsonify(ok=True)
    except (ValueError, TypeError) as e: return err(e)

@app.route("/api/study/summary")
def study_summary():
    with db() as con:
        r = con.execute("SELECT method,COUNT(*),AVG(understand),AVG(trust),AVG(correct) FROM study GROUP BY method").fetchall()
    return jsonify([dict(method=a, n=b, understand=c, trust=d, comprehension_correct=e) for a, b, c, d, e in r])

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)))
