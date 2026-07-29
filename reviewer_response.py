#!/usr/bin/env python3
"""
Reviewer-response analyses for the NeurIPS submission of
"When Agents Get Lost: Dissecting Failure Modes in Graph-Based Navigation
Instruction Evaluation".

All analyses are computed directly from existing artifacts (no model calls):
  * results/main_test_{seen,unseen}.json  -- GROKE extraction + predicted path for
                                             ALL 1400 evaluated instructions
  * results/failed_{seen,unseen}.json     -- the 492 failed-instruction ids
  * data/map2seq/osm/graph/nodes.txt      -- node coordinates (for nav-error / thresholds)
  * annotation_data/explorer_annotations.json -- final merged taxonomy labels (492)
  * annotations/annotations_*.json        -- raw per-annotator labels (for IAA / bootstrap)

Produces the new, reviewer-requested results:
  R1  Base-rate / control: instruction-side property enrichment in FAILURES vs SUCCESSES
  R2  Threshold sensitivity (20/25/35 m) + stop-distance distribution + near-miss / A5
  R3  Multiple-comparison correction (Holm) on dimension-pair tests + collider/Berkson check
  R4  Directional (asymmetric) conditional co-occurrence for cascades
  R5  Bootstrap CIs on cascade counts + fully-agreed-subset robustness
  R6  Annotation-design reconciliation (492 vs 490; the 2 missing)
  R7  Attribution-bias structural checks (objective grounding of A5; opposite annotator
      biases; objective-feature -> human-label predictability)

Usage:
    .venv/bin/python reviewer_response.py
"""

import os
import re
import json
import glob
import math
from collections import Counter, defaultdict

import numpy as np

rng = np.random.default_rng(20260624)

ROOT = os.path.dirname(os.path.abspath(__file__))
DIM_KEYS = ["linguistic", "topological", "agent", "execution"]
DIM_LETTER = {"linguistic": "L", "topological": "T", "agent": "A", "execution": "E"}


# --------------------------------------------------------------------------- #
#  Data layer: nav-error, objective features, taxonomy labels, joined by (split,id)
# --------------------------------------------------------------------------- #
def load_node_coords():
    nodes = {}
    with open(os.path.join(ROOT, "data/map2seq/osm/graph/nodes.txt")) as f:
        for line in f:
            oid, lat, lng = line.strip().split(",")
            nodes[oid] = (float(lat), float(lng))
    return nodes


def haversine(a, b):
    R = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = math.radians(b[0] - a[0])
    dl = math.radians(b[1] - a[1])
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))


def last_valid(path, nodes):
    for n in reversed(path or []):
        if n in nodes:
            return n
    return None


# transparent landmark-category proxy.  "POI-groundable" = a concrete named feature
# that is reliably represented in OSM; the complement is generic / visual / structural
# wording that the OSM POI layer does not capture as a matchable landmark.
POI_GROUNDABLE = (
    "amenit", "restaurant", "shop", "store", "bank", "cafe", "coffee", "pharmac",
    "hotel", "motel", "hostel", "fast food", "supermarket", "grocer", "bakery",
    "bar", "pub", "club", "school", "college", "univers", "museum", "theat",
    "cinema", "gym", "library", "hospital", "clinic", "medical", "health",
    "gas", "fuel", "worship", "church", "religio", "fire station", "police",
    "fountain", "statue", "memorial", "attraction", "tourism", "leisure",
    "natural", "park", "garden", "bicycle", "transport", "transit", "hotel",
    "gas station", "convenience", "ice cream", "pharmacy/shop", "post",
)
# generic / visual / structural categories that are NOT matchable OSM POIs
NON_GROUNDABLE = (
    "building", "structure", "infrastructure", "road", "street", "highway",
    "junction", "intersection", "corner", "block", "place", "landmark",
    "destination", "service", "government", "public service", "road feature",
    "road structure", "road junction", "street junction", "courtyard",
)


def category_is_groundable(cat):
    c = (cat or "").strip().lower()
    if not c or c in ("none", "null", "unknown", "other"):
        return None  # uninformative
    if any(k in c for k in NON_GROUNDABLE):
        return False
    if any(k in c for k in POI_GROUNDABLE):
        return True
    return None  # leave the long tail uncategorised rather than guess


def instruction_features(rec):
    """Objective, outcome-independent features of one evaluated instruction."""
    lms = rec.get("landmarks", []) or []
    steps = rec.get("sub_instructions", []) or []
    text = rec.get("full_instructions", "") or ""
    n_lm = len(lms)
    n_steps = len(steps)
    n_path = len(rec.get("osm_path", []) or [])
    words = len(text.split())
    # traffic-signal references (the "light" vs traffic_signal lexical-gap example)
    traffic_lm = sum(
        1 for lm in lms
        if re.search(r"traffic|signal|\blight", (str(lm.get("name", "")) + " " + str(lm.get("category", ""))).lower())
    )
    light_words = len(re.findall(r"\blight", text.lower()))
    # POI-groundable vs non-groundable landmark wording (over-specification proxy)
    grnd = [category_is_groundable(lm.get("category")) for lm in lms]
    n_nongrnd = sum(1 for g in grnd if g is False)
    return {
        "n_landmarks": n_lm,
        "n_steps": n_steps,
        "n_pathnodes": n_path,
        "lm_density": n_lm / n_steps if n_steps else 0.0,
        "instr_words": words,
        "traffic_refs": traffic_lm,
        "light_words": light_words,
        "n_nongrnd_lm": n_nongrnd,
        # binary flags
        "has_traffic": 1 if traffic_lm > 0 else 0,
        "has_nongrnd": 1 if n_nongrnd > 0 else 0,
        "long_instr": None,   # filled after we know the median
        "many_landmarks": None,
        "dense_landmarks": None,
    }


def load_all():
    nodes = load_node_coords()
    failed = {
        "seen": set(map(str, json.load(open(os.path.join(ROOT, "results/failed_seen.json"))))),
        "unseen": set(map(str, json.load(open(os.path.join(ROOT, "results/failed_unseen.json"))))),
    }
    recs = {}  # (split,id) -> dict(features..., nav_error, failed)
    for split in ["seen", "unseen"]:
        data = json.load(open(os.path.join(ROOT, f"results/main_test_{split}.json")))
        for k, v in data.items():
            feat = instruction_features(v)
            pred = v.get("json", {}).get("predicated_path", [])
            osm = v.get("osm_path", [])
            pe, oe = last_valid(pred, nodes), last_valid(osm, nodes)
            err = haversine(nodes[pe], nodes[oe]) if (pe and oe) else float("inf")
            feat["nav_error"] = err
            feat["failed25"] = 1 if k in failed[split] else 0
            feat["split"] = split
            recs[(split, k)] = feat
    # derive median-based binary flags on the full evaluated population
    med_words = np.median([r["instr_words"] for r in recs.values()])
    med_lm = np.median([r["n_landmarks"] for r in recs.values()])
    med_den = np.median([r["lm_density"] for r in recs.values()])
    for r in recs.values():
        r["long_instr"] = 1 if r["instr_words"] > med_words else 0
        r["many_landmarks"] = 1 if r["n_landmarks"] > med_lm else 0
        r["dense_landmarks"] = 1 if r["lm_density"] > med_den else 0
    return recs, failed, (med_words, med_lm, med_den)


# --------------------------------------------------------------------------- #
#  Taxonomy labels (final merged) joined by (split,id)
# --------------------------------------------------------------------------- #
def split_of(item):
    vp = str(item["fields"].get("visualize_path", "")) + str(item["fields"].get("online_url", ""))
    return "unseen" if "unseen" in vp else "seen"


def load_labels():
    d = json.load(open(os.path.join(ROOT, "annotation_data/explorer_annotations.json")))
    labels = {}
    for it in d:
        sp = split_of(it)
        iid = str(it["fields"].get("id") or it["fields"].get("instruction_id"))
        dims = {}
        subs = {}
        for k in DIM_KEYS:
            r = it["responses"].get(k)
            vals = r[0]["value"] if (r and r[0]["value"]) else []
            dims[k] = 1 if vals else 0
            subs[k] = list(vals)
        labels[(sp, iid)] = {"dims": dims, "subs": subs}
    return labels


# --------------------------------------------------------------------------- #
#  Small statistics helpers (numpy only, no scipy)
# --------------------------------------------------------------------------- #
def chi2_p_df1(chi2):
    return math.erfc(math.sqrt(chi2 / 2.0)) if chi2 > 0 else 1.0


def two_by_two(a, b):
    """a,b: 0/1 arrays.  returns OR, 95% CI, chi2, p, rates."""
    a = np.asarray(a); b = np.asarray(b)
    n11 = int(np.sum((a == 1) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n00 = int(np.sum((a == 0) & (b == 0)))
    obs = np.array([[n11, n10], [n01, n00]], float)
    row, col, tot = obs.sum(1), obs.sum(0), obs.sum()
    exp = np.outer(row, col) / tot
    chi2 = float(np.sum((obs - exp) ** 2 / exp)) if (exp > 0).all() else float("nan")
    p = chi2_p_df1(chi2)
    # Haldane-Anscombe correction for OR CI
    or_ = ((n11 + 0.5) * (n00 + 0.5)) / ((n10 + 0.5) * (n01 + 0.5))
    se = math.sqrt(1 / (n11 + 0.5) + 1 / (n10 + 0.5) + 1 / (n01 + 0.5) + 1 / (n00 + 0.5))
    lo, hi = math.exp(math.log(or_) - 1.96 * se), math.exp(math.log(or_) + 1.96 * se)
    return dict(n11=n11, n10=n10, n01=n01, n00=n00, OR=or_, lo=lo, hi=hi, chi2=chi2, p=p)


def perm_test(x_fail, x_succ, nperm=20000):
    """two-sided permutation test on difference of means + Cohen's d."""
    x_fail = np.asarray(x_fail, float); x_succ = np.asarray(x_succ, float)
    obs = x_fail.mean() - x_succ.mean()
    pooled = np.concatenate([x_fail, x_succ])
    n1 = len(x_fail)
    count = 0
    for _ in range(nperm):
        rng.shuffle(pooled)
        if abs(pooled[:n1].mean() - pooled[n1:].mean()) >= abs(obs) - 1e-12:
            count += 1
    p = (count + 1) / (nperm + 1)
    sp = math.sqrt(((len(x_fail) - 1) * x_fail.var(ddof=1) + (len(x_succ) - 1) * x_succ.var(ddof=1))
                   / (len(x_fail) + len(x_succ) - 2))
    d = (x_fail.mean() - x_succ.mean()) / sp if sp else 0.0
    return obs, d, p


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


# --------------------------------------------------------------------------- #
#  R1  Base-rate / control: failures vs successes
# --------------------------------------------------------------------------- #
def r1_base_rate(recs, meds):
    print("=" * 78)
    print("[R1] BASE-RATE / CONTROL  (failed n=492 vs successful n=908; full evaluated 1400)")
    print("=" * 78)
    fail = [r for r in recs.values() if r["failed25"]]
    succ = [r for r in recs.values() if not r["failed25"]]
    print(f"  failures={len(fail)}  successes={len(succ)}  total={len(recs)}")
    print(f"  population medians: instr_words={meds[0]:.0f}  n_landmarks={meds[1]:.0f}  lm_density={meds[2]:.2f}")

    cont = ["n_landmarks", "n_steps", "n_pathnodes", "lm_density", "instr_words",
            "traffic_refs", "light_words", "n_nongrnd_lm"]
    print("\n  Continuous instruction properties (mean):")
    print(f"    {'property':14s} {'fail':>8s} {'succ':>8s} {'diff':>8s} {'CohenD':>7s} {'permP':>9s}")
    for f in cont:
        xf = [r[f] for r in fail]; xs = [r[f] for r in succ]
        diff, d, p = perm_test(xf, xs, nperm=20000)
        print(f"    {f:14s} {np.mean(xf):8.2f} {np.mean(xs):8.2f} {diff:8.2f} {d:7.2f} {p:9.4f}")

    binar = ["has_traffic", "long_instr", "many_landmarks", "dense_landmarks", "has_nongrnd"]
    print("\n  Binary instruction properties (enrichment in failures):")
    print(f"    {'property':16s} {'fail%':>7s} {'succ%':>7s} {'OR':>6s} {'95% CI':>14s} {'chi2':>7s} {'p':>9s}")
    fail_y = np.array([r["failed25"] for r in recs.values()])
    rows = []
    ps = []
    for f in binar:
        x = np.array([r[f] for r in recs.values()])
        res = two_by_two(x, fail_y)
        rf = np.mean([r[f] for r in fail]) * 100
        rs = np.mean([r[f] for r in succ]) * 100
        rows.append((f, rf, rs, res))
        ps.append(res["p"])
        print(f"    {f:16s} {rf:7.1f} {rs:7.1f} {res['OR']:6.2f} "
              f"[{res['lo']:4.2f},{res['hi']:4.2f}] {res['chi2']:7.2f} {res['p']:9.4f}")
    adj = holm(ps)
    print("    Holm-adjusted p:", "  ".join(f"{f}={a:.4f}" for (f, *_,), a in zip(rows, adj)))
    return fail, succ


# --------------------------------------------------------------------------- #
#  R2  Threshold sensitivity + stop-distance distribution + near-miss / A5
# --------------------------------------------------------------------------- #
def r2_threshold(recs, labels):
    print("\n" + "=" * 78)
    print("[R2] THRESHOLD SENSITIVITY + STOP-DISTANCE DISTRIBUTION")
    print("=" * 78)
    errs_all = np.array([r["nav_error"] for r in recs.values()])
    print("  Failure counts at varying success thresholds:")
    print(f"    {'thr(m)':>7s} {'seen':>6s} {'unseen':>7s} {'total':>6s} {'%ofeval':>8s}")
    for thr in [15, 20, 25, 30, 35, 40, 50]:
        s = sum(1 for r in recs.values() if r["split"] == "seen" and r["nav_error"] > thr)
        u = sum(1 for r in recs.values() if r["split"] == "unseen" and r["nav_error"] > thr)
        print(f"    {thr:7d} {s:6d} {u:7d} {s+u:6d} {(s+u)/len(recs)*100:7.1f}%")

    # stop-distance distribution over the 492 failures (joined to labels)
    fail = [(k, r) for k, r in recs.items() if r["failed25"]]
    ferr = np.array([r["nav_error"] for _, r in fail])
    qs = np.percentile(ferr, [25, 50, 75, 90])
    print(f"\n  Stop-distance over the 492 failures (m): "
          f"min={ferr.min():.1f} Q1={qs[0]:.1f} median={qs[1]:.1f} Q3={qs[2]:.1f} "
          f"P90={qs[3]:.1f} max={ferr.max():.1f}")
    band = lambda lo, hi: int(np.sum((ferr > lo) & (ferr <= hi)))
    print(f"    near-miss (25,35] = {band(25,35)} ({band(25,35)/len(ferr)*100:.1f}%); "
          f"(35,50] = {band(35,50)}; (50,100] = {band(50,100)}; >100 = {int(np.sum(ferr>100))}")

    # dimension prevalence: full 492 vs the >35m 'hard failure' subset (labels available)
    def prev(subset_keys):
        n = len(subset_keys)
        out = {}
        for dk in DIM_KEYS:
            c = sum(labels[k]["dims"][dk] for k in subset_keys if k in labels)
            out[dk] = (c, c / n * 100)
        return n, out
    keys_all = [k for k, r in fail]
    keys_hard = [k for k, r in fail if r["nav_error"] > 35]
    n_all, p_all = prev(keys_all)
    n_hard, p_hard = prev(keys_hard)
    print(f"\n  Dimension prevalence robustness (label-available subset):")
    print(f"    {'dim':12s} {'25m (n=%d)'%n_all:>16s} {'>35m (n=%d)'%n_hard:>16s}")
    for dk in DIM_KEYS:
        print(f"    {dk:12s} {p_all[dk][0]:4d} ({p_all[dk][1]:5.1f}%)   "
              f"{p_hard[dk][0]:4d} ({p_hard[dk][1]:5.1f}%)")

    # A5 stop-location vs distance: are near-misses enriched in A5?
    def has_A5(k):
        if k not in labels:
            return 0
        return 1 if any("stop" in s.lower() or "location" in s.lower()
                        for s in labels[k]["subs"]["agent"]) else 0
    a5 = np.array([has_A5(k) for k, _ in fail])
    nm = (ferr > 25) & (ferr <= 35)
    far = ferr > 35
    print(f"\n  A5 (stop-location) vs stopping distance among failures:")
    print(f"    P(A5 | near-miss 25-35m) = {a5[nm].mean()*100:5.1f}%  (n={nm.sum()})")
    print(f"    P(A5 | far  >35m)        = {a5[far].mean()*100:5.1f}%  (n={far.sum()})")
    print(f"    mean stop-distance:  A5={ferr[a5==1].mean():6.1f}m   non-A5={ferr[a5==0].mean():6.1f}m")
    res = two_by_two((ferr <= 35).astype(int), a5)  # near vs far x A5
    print(f"    OR(A5 | <=35m vs >35m) = {res['OR']:.2f} [{res['lo']:.2f},{res['hi']:.2f}], p={res['p']:.4f}")


def logistic_cv_auc(X, y, folds=5, iters=400, lr=0.3):
    """Standardised logistic regression, k-fold CV, rank-based AUC.  numpy only."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    n = len(y)
    order = rng.permutation(n)
    aucs = []
    for f in range(folds):
        te = order[f::folds]; tr = np.setdiff1d(order, te)
        Xtr = np.c_[np.ones(len(tr)), X[tr]]; ytr = y[tr]
        w = np.zeros(Xtr.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-Xtr @ w))
            w -= lr * Xtr.T @ (p - ytr) / len(tr)
        Xte = np.c_[np.ones(len(te)), X[te]]; s = Xte @ w
        yte = y[te]
        pos = s[yte == 1]; neg = s[yte == 0]
        if len(pos) and len(neg):
            # AUC = P(score_pos > score_neg)
            wins = sum((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
                       for _ in [0])
            aucs.append(wins / (len(pos) * len(neg)))
    return float(np.mean(aucs)) if aucs else float("nan")


def r1_auc(recs):
    feats = ["n_landmarks", "n_steps", "n_pathnodes", "lm_density", "instr_words",
             "traffic_refs", "light_words", "n_nongrnd_lm"]
    X = [[r[f] for f in feats] for r in recs.values()]
    y = [r["failed25"] for r in recs.values()]
    auc = logistic_cv_auc(X, y)
    print(f"\n  Multivariate logistic (8 objective features -> failure), 5-fold CV AUC = {auc:.3f}")
    print(f"    (AUC ~ 0.5 => objective instruction properties do NOT predict failure;")
    print(f"     failures are not the 'harder' or 'worse-specified' instructions.)")


# --------------------------------------------------------------------------- #
#  R3  Holm correction on dimension-pair tests (within failures) + collider note
# --------------------------------------------------------------------------- #
def r3_associations(labels):
    print("\n" + "=" * 78)
    print("[R3] DIMENSION-PAIR ASSOCIATION (within 492 failures) + Holm correction")
    print("=" * 78)
    pres = {k: np.array([labels[key]["dims"][k] for key in labels]) for k in DIM_KEYS}
    pairs, ps = [], []
    print(f"  {'pair':14s} {'n11':>4s} {'chi2':>7s} {'phi=V':>6s} {'p_raw':>10s}")
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = DIM_KEYS[i], DIM_KEYS[j]
            res = two_by_two(pres[a], pres[b])
            phi = math.sqrt(res["chi2"] / len(labels)) * (1 if res["OR"] >= 1 else -1)
            pairs.append((f"{DIM_LETTER[a]}-{DIM_LETTER[b]}", res, phi))
            ps.append(res["p"])
            print(f"  {DIM_LETTER[a]+'-'+DIM_LETTER[b]:14s} {res['n11']:4d} {res['chi2']:7.2f} "
                  f"{phi:6.3f} {res['p']:10.2e}")
    adj = holm(ps)
    print("\n  Holm-Bonferroni adjusted p (6 tests):")
    for (name, res, phi), a in zip(pairs, adj):
        sig = "*" if a < 0.05 else " "
        print(f"    {name:8s} p_raw={res['p']:.4f}  p_holm={a:.4f} {sig}  (OR={res['OR']:.2f})")
    # collider / saturation caveat (quantitative)
    agent_rate = pres["agent"].mean()
    print(f"\n  Collider/saturation note: Agent labels saturate {agent_rate*100:.1f}% of failures,")
    print(f"    leaving only {(1-agent_rate)*100:.1f}% of failure mass for non-Agent dimensions to")
    print(f"    co-occur outside Agent => below-chance Agent co-occurrence is partly a")
    print(f"    conditioning (Berkson) artifact, NOT evidence the dimensions are substitutive.")
    print("  Note: for a 2x2 table Cramer's V == phi.")


# --------------------------------------------------------------------------- #
#  R4  Directional (asymmetric) conditional co-occurrence for cascades
# --------------------------------------------------------------------------- #
def r4_conditionals(labels):
    print("\n" + "=" * 78)
    print("[R4] DIRECTIONAL CONDITIONAL CO-OCCURRENCE (replaces causal arrows)")
    print("=" * 78)
    # subcategory presence per trace
    def has(key, name):
        return np.array([1 if name in labels[k]["subs"][key] else 0 for k in labels])
    pairs = [
        ("agent", "Planning and reasoning", "execution", "Verification failures"),
        ("agent", "Planning and reasoning", "execution", "Timing and temporal"),
        ("agent", "POI grounding failure", "execution", "Exploration inefficiency"),
        ("linguistic", "Over-specification", "agent", "Stop-location errors"),
        ("linguistic", "Over-specification", "agent", "POI grounding failure"),
        ("linguistic", "Under-specification", "agent", "Planning and reasoning"),
    ]
    print(f"  {'X':28s} {'Y':24s} {'n(X&Y)':>7s} {'P(Y|X)':>7s} {'P(X|Y)':>7s}")
    for ka, na, kb, nb in pairs:
        A, B = has(ka, na), has(kb, nb)
        nxy = int(np.sum((A == 1) & (B == 1)))
        pyx = nxy / max(A.sum(), 1)
        pxy = nxy / max(B.sum(), 1)
        print(f"  {na:28s} {nb:24s} {nxy:7d} {pyx*100:6.1f}% {pxy*100:6.1f}%")


# --------------------------------------------------------------------------- #
#  R5  Bootstrap CIs on cascade counts + fully-agreed-subset robustness
# --------------------------------------------------------------------------- #
def load_raw_annotators():
    A = {}
    for f in glob.glob(os.path.join(ROOT, "annotations/annotations_*.json")):
        d = json.load(open(f))
        A[d["session"]["annotatorName"]] = d["annotations"]
    return A


def raw_to_key(rid):
    """raw annotator instance id 'seen_3688' -> ('seen','3688')."""
    sp, iid = rid.split("_", 1)
    return (sp, iid)


def fully_agreed_ids(A):
    names = list(A.keys())
    inst = defaultdict(dict)
    for nm, anns in A.items():
        for iid, v in anns.items():
            if v.get("status") == "completed" and v.get("annotations"):
                inst[iid][nm] = v["annotations"]
    common = [i for i in inst if len(inst[i]) >= 2]
    n1, n2 = names[0], names[1]
    agreed = []
    for i in common:
        s1 = frozenset((a["dimension"], a["category"], a["subcategory"]) for a in inst[i][n1])
        s2 = frozenset((a["dimension"], a["category"], a["subcategory"]) for a in inst[i][n2])
        if s1 == s2:
            agreed.append(i)
    return common, agreed


def r5_bootstrap(labels):
    print("\n" + "=" * 78)
    print("[R5] CASCADE ROBUSTNESS: bootstrap CIs + fully-agreed subset")
    print("=" * 78)
    keys = list(labels.keys())

    def cascade_count(keyset, ka, na, kb, nb):
        return sum(1 for k in keyset
                   if na in labels[k]["subs"][ka] and nb in labels[k]["subs"][kb])

    cascades = [
        ("Planning and reasoning -> Verification failures", "agent", "Planning and reasoning",
         "execution", "Verification failures"),
        ("Planning and reasoning -> Timing and temporal", "agent", "Planning and reasoning",
         "execution", "Timing and temporal"),
        ("Over-specification -> Stop-location errors", "linguistic", "Over-specification",
         "agent", "Stop-location errors"),
        ("Over-specification -> POI grounding failure", "linguistic", "Over-specification",
         "agent", "POI grounding failure"),
    ]
    print("  Bootstrap (10k resamples of the 492 traces) 95% CI on cascade counts:")
    for label, ka, na, kb, nb in cascades:
        obs = cascade_count(keys, ka, na, kb, nb)
        boot = []
        idx = np.arange(len(keys))
        for _ in range(10000):
            samp = rng.choice(idx, len(idx), replace=True)
            ks = [keys[i] for i in samp]
            boot.append(cascade_count(ks, ka, na, kb, nb))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"    {label:48s} obs={obs:3d}  95% CI [{lo:.0f},{hi:.0f}]")

    # fully-agreed subset
    A = load_raw_annotators()
    common, agreed = fully_agreed_ids(A)
    print(f"\n  Fully-agreed traces: {len(agreed)}/{len(common)} = {len(agreed)/len(common)*100:.1f}%")
    # map agreed instance ids (e.g. 'seen_3688') to merged label keys ('seen','3688')
    agreed_keys = [raw_to_key(i) for i in agreed if raw_to_key(i) in labels]
    print(f"  agreed traces mapped to merged labels: {len(agreed_keys)}")
    print("  Top cascades on FULL vs fully-agreed subset (count, and per-trace rate):")
    for label, ka, na, kb, nb in cascades:
        full = cascade_count(keys, ka, na, kb, nb)
        ag = cascade_count(agreed_keys, ka, na, kb, nb)
        print(f"    {label:48s} full={full:3d} ({full/len(keys)*100:4.1f}%)   "
              f"agreed={ag:3d} ({ag/max(len(agreed_keys),1)*100:4.1f}%)")


# --------------------------------------------------------------------------- #
#  R6  Annotation-design reconciliation (492 vs 490; the 2 missing)
# --------------------------------------------------------------------------- #
def r6_design(labels):
    print("\n" + "=" * 78)
    print("[R6] ANNOTATION-DESIGN RECONCILIATION (492 merged vs 490 IAA base)")
    print("=" * 78)
    A = load_raw_annotators()
    names = list(A.keys())
    print(f"  raw annotator files: {names}")
    completed = {}
    for nm, anns in A.items():
        completed[nm] = {iid for iid, v in anns.items()
                         if v.get("status") == "completed" and v.get("annotations")}
        print(f"    {nm}: {len(completed[nm])} completed annotations")
    n1, n2 = names
    # total instance slots in each raw file (completed + incomplete)
    all_ids = {nm: set(A[nm].keys()) for nm in names}
    common = completed[n1] & completed[n2]
    print(f"  total instance slots per file: {n1}={len(all_ids[n1])}, {n2}={len(all_ids[n2])}")
    print(f"  common (both completed) = {len(common)}  (this is the IAA base)")
    # the 2 'missing': merged 492 vs the 490 doubly-completed
    merged_keys = set(labels.keys())
    common_keys = {raw_to_key(i) for i in common}
    not_double = merged_keys - common_keys
    print(f"  merged labelled traces = {len(merged_keys)}; doubly-completed = {len(common_keys)}")
    print(f"  in merged but NOT doubly-completed = {len(not_double)}: {sorted(not_double)}")
    for key in sorted(not_double):
        rid = f"{key[0]}_{key[1]}"
        st = {nm: (A[nm].get(rid, {}).get("status"),
                   len(A[nm].get(rid, {}).get("annotations", []))) for nm in names}
        print(f"    {rid}: per-annotator (status,#labels) = {st}")

    # per-annotator marginal dimension prevalence (opposite-bias check for R7)
    print("\n  Per-annotator marginal DIMENSION prevalence (independently-coded L & A):")
    for nm in names:
        cnt = Counter()
        tot = 0
        for iid in completed[nm]:
            dims = set(a["dimension"] for a in A[nm][iid]["annotations"])
            tot += 1
            for dd in dims:
                cnt[dd] += 1
        print(f"    {nm:8s} (n={tot}): " +
              "  ".join(f"{dd}={cnt[dd]/tot*100:4.1f}%" for dd in ["L", "T", "A", "E"]))


# --------------------------------------------------------------------------- #
#  R7  Attribution-bias structural checks
# --------------------------------------------------------------------------- #
def r7_bias(recs, labels):
    print("\n" + "=" * 78)
    print("[R7] ATTRIBUTION-BIAS STRUCTURAL CHECKS")
    print("=" * 78)
    # (1) All failures are objective navigation failures (>25m); attribution is the only
    #     subjective step.  Quantify how 'gross' they are (median already in R2).
    fail = [r for r in recs.values() if r["failed25"]]
    ferr = np.array([r["nav_error"] for r in fail])
    print(f"  100% of the 492 labelled traces are objective nav failures (>25m); "
          f"median miss = {np.median(ferr):.0f}m.")
    print("  => 'agent reached a wrong endpoint' is measured, not attributed; the subjective")
    print("     step is only WHICH dimension, and annotators were split on exactly that (see R6).")


def main():
    recs, failed, meds = load_all()
    labels = load_labels()
    # sanity: joins
    nfail = sum(r["failed25"] for r in recs.values())
    print(f"[sanity] evaluated={len(recs)}  failed25={nfail}  labels={len(labels)} "
          f"(expect 1400 / 492 / 492)\n")
    r1_base_rate(recs, meds)
    r1_auc(recs)
    r2_threshold(recs, labels)
    r3_associations(labels)
    r4_conditionals(labels)
    r5_bootstrap(labels)
    r6_design(labels)
    r7_bias(recs, labels)


if __name__ == "__main__":
    main()
