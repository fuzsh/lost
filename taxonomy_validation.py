#!/usr/bin/env python3
"""
Taxonomy validation analysis for the reviewer response (exclusivity & completeness).
Analysis-only; reads existing files, writes nothing, no model calls.

Reproduces:
  - COMPLETENESS / coverage: fraction of traces categorized; the residual
    uncategorized traces (GROKE false positives) identified from annotator notes;
    dimension-cardinality distribution; rare/long-tail leaf categories.
  - EXCLUSIVITY / orthogonality: dimension co-occurrence vs. chance (see
    latex/make_figures.py for Cramer's V) and the inter-annotator dimension
    confusion. NOTE: only the Linguistic and Agent dimensions were independently
    double-coded; the Topological and Execution labels are identical across both
    annotator files (consensus-coded), so independent agreement is reported on L/A.

Usage:
    .venv/bin/python taxonomy_validation.py
"""
import json, glob
from collections import Counter, defaultdict

DIM = ['linguistic', 'topological', 'agent', 'execution']
LET = {'linguistic': 'L', 'topological': 'T', 'agent': 'A', 'execution': 'E'}


def vals(it, k):
    r = it['responses'].get(k)
    return r[0]['value'] if (r and r[0]['value']) else []


def dims(it):
    return [k for k in DIM if vals(it, k)]


def completeness():
    d = json.load(open('annotation_data/explorer_annotations.json'))
    n = len(d)
    uncovered = [it for it in d if not dims(it)]
    card = Counter(len(dims(it)) for it in d)
    print('=== COMPLETENESS / COVERAGE ===')
    print(f'traces={n}  categorized={n-len(uncovered)} ({(n-len(uncovered))/n*100:.1f}%)  '
          f'uncovered={len(uncovered)} ({len(uncovered)/n*100:.1f}%)')
    print('dimension-cardinality per trace:', dict(sorted(card.items())))
    multi = sum(v for k, v in card.items() if k >= 2)
    labeled = n - len(uncovered)
    print(f'multi-dimensional among labeled: {multi}/{labeled} = {multi/labeled*100:.1f}%')
    print('\nUncovered traces are GROKE false positives (agent succeeded; threshold/'
          'geometry artifact). Annotator notes:')
    for it in uncovered:
        note = it['responses'].get('note')
        txt = note[0]['value'] if note and note[0].get('value') else '(none)'
        print(f'  id={it["fields"].get("id")}: {str(txt)[:75]}')
    sub = Counter()
    for it in d:
        for k in DIM:
            for v in vals(it, k):
                sub[(LET[k], v)] += 1
    print('\nLong-tail leaf categories (<=3 occurrences):',
          [f'{a}:{b}={c}' for (a, b), c in sub.items() if c <= 3])


def exclusivity():
    A = {}
    for f in glob.glob('annotations/annotations_*.json'):
        x = json.load(open(f))
        A[x['session']['annotatorName']] = x['annotations']
    n1, n2 = list(A.keys())
    inst = defaultdict(dict)
    for nm, anns in A.items():
        for iid, v in anns.items():
            if v.get('status') == 'completed' and v.get('annotations'):
                inst[iid][nm] = set(a['dimension'] for a in v['annotations'])
    common = [i for i in inst if len(inst[i]) == 2]
    print('\n=== EXCLUSIVITY (independent inter-annotator dimension coding) ===')
    same, differ = Counter(), Counter()
    for i in common:
        s1, s2 = inst[i][n1], inst[i][n2]
        for dd in 'LTAE':
            (same if (dd in s1) == (dd in s2) else differ)[dd] += 1
    for dd in 'LTAE':
        tot = same[dd] + differ[dd]
        note = '  (consensus-coded, not independent)' if differ[dd] == 0 else ''
        print(f'  {dd}: identical on {same[dd]}/{tot} = {same[dd]/tot*100:.1f}% traces{note}')
    cross = Counter()
    for i in common:
        for x in inst[i][n1] - inst[i][n2]:
            for y in inst[i][n2] - inst[i][n1]:
                cross[tuple(sorted((x, y)))] += 1
    print('Dimension cross-substitution (where annotators disagree):', dict(cross))
    print('=> The only independently-coded boundary in dispute is Agent<->Linguistic,')
    print('   i.e. instruction-fault vs. agent-fault attribution (the central question).')


if __name__ == '__main__':
    completeness()
    exclusivity()
