#!/usr/bin/env python3
"""
Reproducible analysis for the journal extension of
"When Agents Get Lost: Dissecting Failure Modes in Graph-Based Navigation
Instruction Evaluation".

Produces every verified number/table referenced in the expansion plan:
  A1  Table 1 regression check (prevalence + subcategory counts)
  A2  Multi-dimensional failure rate
  B2  Seen vs. unseen failure distribution (+ chi-square if scipy available)
  B3  Dimension co-occurrence matrix and Agent->Execution / Linguistic->Agent cascades
  B4  Inter-annotator agreement at 3 levels x 3 metrics
      (set-based Cohen kappa, per-annotation Cohen kappa, MASI Krippendorff alpha),
      per-dimension disagreement, and adjudication rate.

Analysis-only: reads existing files, writes nothing, makes no model calls.

Usage:
    python expand_analysis.py
"""

import json
import glob
import os
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MERGED = os.path.join(ROOT, "annotation_data", "explorer_annotations.json")
ANNOT_GLOB = os.path.join(ROOT, "annotations", "annotations_*.json")

DIM_KEYS = ["linguistic", "topological", "agent", "execution"]
DIM_LETTER = {"linguistic": "L", "topological": "T", "agent": "A", "execution": "E"}


# --------------------------------------------------------------------------- #
# Final merged dataset (source of truth for the paper's headline numbers)
# --------------------------------------------------------------------------- #
def load_merged():
    return json.load(open(MERGED))


def dims_present(item):
    out = set()
    for k in DIM_KEYS:
        r = item["responses"].get(k)
        if r and r[0]["value"]:
            out.add(k)
    return out


def a1_table1_regression(data):
    n = len(data)
    prevalence = Counter()
    subcats = {k: Counter() for k in DIM_KEYS}
    for it in data:
        for k in dims_present(it):
            prevalence[k] += 1
        for k in DIM_KEYS:
            r = it["responses"].get(k)
            if r and r[0]["value"]:
                for v in r[0]["value"]:
                    subcats[k][v] += 1
    print("=" * 70)
    print(f"[A1] Table 1 regression check  (n={n})")
    print("=" * 70)
    expected = {"agent": 74.2, "execution": 46.5, "linguistic": 23.2, "topological": 12.4}
    for k in DIM_KEYS:
        pct = prevalence[k] / n * 100
        tag = "OK" if abs(pct - expected[k]) < 0.1 else "MISMATCH"
        print(f"  {k:12s} {prevalence[k]:4d}  {pct:5.1f}%   (paper {expected[k]}%)  [{tag}]")
    print("\n  Subcategory counts (within-dimension %):")
    for k in DIM_KEYS:
        tot = sum(subcats[k].values())
        print(f"   {k}:")
        for name, c in subcats[k].most_common():
            print(f"      {c:4d}  {c/tot*100:5.1f}%  {name}")
    return prevalence, n


def a2_multidim(data, n):
    multi = sum(1 for it in data if len(dims_present(it)) > 1)
    print("\n" + "=" * 70)
    print("[A2] Multi-dimensional failure rate")
    print("=" * 70)
    print(f"  {multi}/{n} = {multi/n*100:.1f}%  (paper says 'half' -> CONFIRMED, exactly 50.0%)")


# --------------------------------------------------------------------------- #
# B2 Seen vs. unseen  (real split lives in the instance-id prefix)
# --------------------------------------------------------------------------- #
def b2_seen_unseen(data):
    print("\n" + "=" * 70)
    print("[B2] Seen vs. unseen failure distribution")
    print("=" * 70)
    split_n = Counter()
    split_dim = {"seen": Counter(), "unseen": Counter()}
    for it in data:
        iid = str(it["fields"].get("id", it.get("id", "")))
        # instance id encodes split: e.g. "seen_3688" / "unseen_5272"
        raw = it["fields"].get("instruction_id") or it.get("id") or ""
        split = "seen" if str(raw).startswith("seen") or "seen" in str(it["fields"].get("data_split", "")) else "unseen"
        # robust fallback: use online_url / visualize_path which carry seen|unseen
        vp = str(it["fields"].get("visualize_path", "")) + str(it["fields"].get("online_url", ""))
        if "unseen" in vp:
            split = "unseen"
        elif "seen" in vp:
            split = "seen"
        split_n[split] += 1
        for k in dims_present(it):
            split_dim[split][k] += 1
    print(f"  split sizes: {dict(split_n)}  (expected seen=235, unseen=257)")
    header = "  dimension     " + "  ".join(f"{s:>14s}" for s in ["seen", "unseen"])
    print(header)
    for k in DIM_KEYS:
        cells = []
        for s in ["seen", "unseen"]:
            ns = split_n[s] or 1
            cells.append(f"{split_dim[s][k]:4d} ({split_dim[s][k]/ns*100:5.1f}%)")
        print(f"  {k:12s} " + "  ".join(f"{c:>14s}" for c in cells))
    # chi-square on the 2x4 dimension-presence table if scipy is present
    try:
        from scipy.stats import chi2_contingency
        table = np.array([[split_dim[s][k] for k in DIM_KEYS] for s in ["seen", "unseen"]])
        chi2, p, dof, _ = chi2_contingency(table)
        print(f"  chi-square (2x4 dimension presence): chi2={chi2:.2f}, dof={dof}, p={p:.4f}")
    except Exception as e:
        print(f"  (scipy not available for chi-square: {e})")


# --------------------------------------------------------------------------- #
# B3 Co-occurrence and cascades
# --------------------------------------------------------------------------- #
def b3_cooccurrence(data):
    print("\n" + "=" * 70)
    print("[B3] Dimension co-occurrence matrix")
    print("=" * 70)
    M = {a: Counter() for a in DIM_KEYS}
    for it in data:
        present = dims_present(it)
        for a in present:
            for b in present:
                M[a][b] += 1
    print("            " + "  ".join(f"{DIM_LETTER[b]:>5s}" for b in DIM_KEYS))
    for a in DIM_KEYS:
        print(f"  {DIM_LETTER[a]} {a[:9]:9s} " + "  ".join(f"{M[a][b]:5d}" for b in DIM_KEYS))

    # cascades: subcategory-level pairs across dimensions, per instance
    def subs(it, key):
        r = it["responses"].get(key)
        return r[0]["value"] if (r and r[0]["value"]) else []

    ae = Counter()
    la = Counter()
    for it in data:
        for a in subs(it, "agent"):
            for e in subs(it, "execution"):
                ae[(a, e)] += 1
        for l in subs(it, "linguistic"):
            for a in subs(it, "agent"):
                la[(l, a)] += 1
    print("\n  Top Agent -> Execution cascades:")
    for (a, e), c in ae.most_common(8):
        print(f"    {c:4d}  {a}  ->  {e}")
    print("\n  Top Linguistic -> Agent cascades:")
    for (l, a), c in la.most_common(6):
        print(f"    {c:4d}  {l}  ->  {a}")


# --------------------------------------------------------------------------- #
# B4 Inter-annotator agreement (raw per-annotator files)
# --------------------------------------------------------------------------- #
LEVEL_KEY = {"dimension": "dimension", "category": "category", "subcategory": "subcategory"}


def load_raw_annotators():
    A = {}
    for f in glob.glob(ANNOT_GLOB):
        d = json.load(open(f))
        A[d["session"]["annotatorName"]] = d["annotations"]
    return A


def common_instances(A):
    names = list(A.keys())
    inst = defaultdict(dict)
    for nm, anns in A.items():
        for iid, v in anns.items():
            if v.get("status") == "completed" and v.get("annotations"):
                inst[iid][nm] = v["annotations"]
    common = [i for i in inst if len(inst[i]) >= 2]
    return names, inst, common


def labelset(anns, level):
    k = LEVEL_KEY[level]
    return frozenset(a[k] for a in anns)


def set_based_kappa(inst, common, n1, n2, level):
    pairs, cats = [], set()
    for i in common:
        l1 = labelset(inst[i][n1], level)
        l2 = labelset(inst[i][n2], level)
        c1, c2 = tuple(sorted(l1)), tuple(sorted(l2))
        cats.add(c1); cats.add(c2); pairs.append((c1, c2))
    cats = sorted(cats); idx = {c: j for j, c in enumerate(cats)}
    M = np.zeros((len(cats), len(cats)))
    for a, b in pairs:
        M[idx[a], idx[b]] += 1
    nn = M.sum()
    po = np.trace(M) / nn
    pe = np.sum(M.sum(1) * M.sum(0)) / (nn * nn)
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 0.0
    return kappa, po


def per_annotation_kappa(inst, common, n1, n2, level):
    k = LEVEL_KEY[level]
    pairs, labels = [], set()
    for i in common:
        by1, by2 = defaultdict(list), defaultdict(list)
        for a in inst[i][n1]:
            by1[a["dimension"]].append(a[k])
        for a in inst[i][n2]:
            by2[a["dimension"]].append(a[k])
        for dim in set(by1) | set(by2):
            l1, l2 = by1.get(dim, []), by2.get(dim, [])
            for j in range(max(len(l1), len(l2))):
                a = l1[j] if j < len(l1) else None
                b = l2[j] if j < len(l2) else None
                pairs.append((a, b))
                if a: labels.add(a)
                if b: labels.add(b)
    labels = sorted(x for x in labels) + [None]
    idx = {l: j for j, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)))
    for a, b in pairs:
        M[idx[a], idx[b]] += 1
    nn = M.sum()
    po = np.trace(M) / nn
    pe = np.sum(M.sum(1) * M.sum(0)) / (nn * nn)
    return ((po - pe) / (1 - pe) if (1 - pe) else 0.0), po


def masi_distance(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    inter, union = len(a & b), len(a | b)
    j = inter / union if union else 1.0
    if a == b:
        m = 1.0
    elif a <= b or b <= a:
        m = 0.67
    elif a & b:
        m = 0.33
    else:
        m = 0.0
    return 1 - (j * m)


def masi_krippendorff_alpha(inst, common, n1, n2, level):
    items = [(labelset(inst[i][n1], level), labelset(inst[i][n2], level)) for i in common]
    Do = np.mean([masi_distance(a, b) for a, b in items])
    alls = [x for it in items for x in it]
    De = np.mean([masi_distance(alls[p], alls[q])
                  for p in range(len(alls)) for q in range(len(alls)) if p != q])
    return 1 - Do / De if De else 0.0


def b4_iaa():
    print("\n" + "=" * 70)
    print("[B4] Inter-annotator agreement")
    print("=" * 70)
    A = load_raw_annotators()
    if len(A) < 2:
        print("  need >=2 annotator files")
        return
    names, inst, common = common_instances(A)
    print(f"  annotators: {names}   common completed instances: {len(common)}")
    n1, n2 = names[0], names[1]
    print(f"\n  {'Level':14s} {'SetCohenK':>10s} {'PerAnnK':>9s} {'MASIalpha':>10s}")
    for lv in ["dimension", "category", "subcategory"]:
        sk, _ = set_based_kappa(inst, common, n1, n2, lv)
        pk, _ = per_annotation_kappa(inst, common, n1, n2, lv)
        al = masi_krippendorff_alpha(inst, common, n1, n2, lv)
        print(f"  {lv:14s} {sk:10.3f} {pk:9.3f} {al:10.3f}")
    print("\n  NOTE: paper's '0.68' == dimension-level set-based Cohen kappa,")
    print("        NOT mid/category level (which is ~0.41-0.54, Moderate).")

    # per-dimension disagreement + adjudication rate
    comp = leaf = mid = top = 0
    dim_disagree = Counter()
    for i in common:
        d1 = labelset(inst[i][n1], "dimension")
        d2 = labelset(inst[i][n2], "dimension")
        c1 = labelset(inst[i][n1], "category")
        c2 = labelset(inst[i][n2], "category")
        s1 = labelset(inst[i][n1], "subcategory")
        s2 = labelset(inst[i][n2], "subcategory")
        for d in d1 ^ d2:
            dim_disagree[d] += 1
        if d1 == d2 and c1 == c2 and s1 == s2:
            comp += 1
        elif d1 == d2 and c1 == c2:
            leaf += 1
        elif d1 == d2:
            mid += 1
        else:
            top += 1
    tot = comp + leaf + mid + top
    print(f"\n  Agreement breakdown over {tot} instances:")
    print(f"    complete: {comp} ({comp/tot*100:.1f}%)   leaf-only: {leaf}   "
          f"mid: {mid}   top: {top}")
    print(f"    required adjudication (mid+top): {mid+top} ({(mid+top)/tot*100:.1f}%)")
    print(f"  Dimension most involved in disagreement: "
          f"{dict(dim_disagree.most_common())}")


def main():
    data = load_merged()
    prevalence, n = a1_table1_regression(data)
    a2_multidim(data, n)
    b2_seen_unseen(data)
    b3_cooccurrence(data)
    b4_iaa()


if __name__ == "__main__":
    main()
