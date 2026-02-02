# When Agents Get Lost: Dissecting Failure Modes in Graph-Based Navigation Instruction Evaluation

**Farzad Shami, Kimia Abedini, Seyed Hossein Hosseini, Henrikki Tenkanen**

Aalto University

[[Paper]](https://arxiv.org/abs/2601.10581) [[Project Page]](https://fuzsh.github.io/lost/) [[Data Explorer]](https://fuzsh.github.io/lost/explorer-tool/)

## Abstract

Vision-and-Language Navigation (VLN) requires agents to interpret natural language instructions for spatial reasoning, yet evaluating instruction quality remains challenging when agents fail. This gap highlights a critical need for a principled understanding of why navigation instructions fail. To address this, we first present a taxonomy of navigation instruction failures that clusters failure cases into four categories: (i) linguistic properties, (ii) topological constraints, (iii) agent limitations, and (iv) execution barriers. We then introduce a dataset of over 450 annotated failure navigation traces collected from GROKE, a vision-free evaluation framework that utilizes OpenStreetMap (OSM) data. Our analysis demonstrates that agent limitations (74.2%) constitute the dominant error category, with stop-location errors and planning failures as the most frequent subcategories. The dataset and taxonomy together provide actionable insights that enable instruction generation systems to identify and avoid under-specification patterns while allowing evaluation frameworks to systematically distinguish between instruction quality issues and agent-specific artifacts.

## Failure Taxonomy

We developed a hierarchical four-axis categorization framework:

| Dimension | Code | Prevalence | Top Subcategory |
|---|---|---|---|
| **Agent Limitations** | A | 74.2% | Stop-location errors (A5, 49.9%) |
| **Execution Failures** | E | 46.5% | Timing/temporal errors (E4, 49.8%) |
| **Linguistic Properties** | L | 23.2% | Over-specification (L2, 42.1%) |
| **Topological Constraints** | T | 12.4% | Junction complexity (T5, 73.8%) |

### Linguistic (L)
- **L1** Under-specification: Spatial under-specification, landmark under-specification, missing stopping criteria
- **L2** Over-specification: Visual-only features (e.g., "red building" not in OSM), transient objects
- **L3** Temporal ambiguity: Overlapping spatial regions, implicit sequencing, conditional dependencies
- **L4** Numerical inconsistency: Numerical mismatch, distance quantification errors
- **L5** Referential complexity: Deictic references, anaphora chains, implicit references
- **L6** Negation confusion: Action negation, landmark negation, double negation
- **L7** Directional ambiguity: Relative direction confusion, conflicting cues, cardinal-relative mismatch
- **L8** Scale/Distance vagueness: Temporal vagueness, perceptual scale issues
- **L9** Landmark co-reference: Synonym variation, partial references

### Topological (T)
- **T1** Path ambiguity: Parallel routes, loop alternatives
- **T2** Landmark displacement: Premature mention, post-turn reference, opposite-side displacement
- **T3** Scale mismatch: Distance underestimation, landmark density
- **T4** Stop-location errors: Stop-too-late (overshoot), stop-too-early (undershoot), missed stopping cue
- **T5** Junction complexity: Five-way+ intersections, offset intersections, roundabout navigation

### Agent Limitations (A)
- **A1** POI grounding failure: Similarity thresholds
- **A2** Heading initialization: Compass calibration, map orientation mismatch
- **A3** Sub-goal segmentation: Over-segmentation, under-segmentation, incorrect action primitives
- **A4** Context window saturation: Token overflow, information overload, visibility threshold miscalibration
- **A5** Perception failures: Spatial relation errors, object hallucination
- **A6** Planning and reasoning: Goal confusion, cascading failures, premature termination
- **A7** Memory and state tracking: Visited location amnesia, sub-goal state loss, instruction forgetting
- **A8** Multi-agent coordination: Incompatible formats, status misalignment

### Execution (E)
- **E1** Looping behavior: Immediate loops, cycle loops, area circling
- **E2** Exploration inefficiency: Random walk, boundary avoidance
- **E3** Action execution errors: Turn angle errors, distance errors
- **E4** Timing and temporal: Premature actions, delayed actions, simultaneous action conflicts
- **E5** Verification failures: False positive completion, false negative completion, no verification

## Dataset

The dataset contains **492 annotated navigation failure traces** collected from GROKE's evaluation of Map2Seq test sets, representing 35.14% of evaluated instructions. Each trace includes:

- Agent's step-by-step reasoning
- Identified sub-goals
- Extracted POIs
- Map with annotated path
- Graph network showing traversed path with marked POIs

### Annotation Quality

Three expert annotators participated in the annotation process:

| Level | Agreement | Cohen's kappa |
|---|---|---|
| Top Level | 95% | 0.95 |
| Mid-level | 82% | 0.78 |
| Leaf-level | 68% | 0.61 |

- 680 instances (68%) showed complete agreement
- 240 instances (24%) resolved via deepest common ancestor method
- 80 instances (8%) required manual adjudication

## Key Findings

- **Agent limitations (74.2%)** are the dominant error category, with stop-location errors and planning failures as the most frequent subcategories.
- **Execution failures (46.5%)** are the second most common, dominated by timing and temporal errors.
- **Half of all failures** exhibit multi-dimensional error patterns, indicating compounded issues rather than isolated problems.
- **Six design implications** are derived for improving vision-free navigation systems, including spatial representation improvements, ambiguous terminology handling, landmark detection enhancement, action timing refinement, junction complexity handling, and stop-location refinement.

## Project Structure

- `data/` - Navigation data
- `annotations/` - Annotated failure traces
- `annotation_data/` - Raw annotation data
- `annotation-tool/` - Web-based annotation tool
- `explorer-tool/` - Interactive data exploration tool
- `src/` - Source code
- `results/` - Evaluation results
- `evaluation_metrics.py` - Evaluation metrics
- `inter-annotator.py` - Inter-annotator agreement computation
- `index.html` - Project landing page (GitHub Pages)

## Citation

```bibtex
@misc{shami2025lost,
      title={When Agents Get Lost: Dissecting Failure Modes in Graph-Based Navigation Instruction Evaluation},
      author={Farzad Shami and Kimia Abedini and Seyed Hossein Hosseini and Henrikki Tenkanen},
      year={2025},
      eprint={2601.10581},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.10581},
}
```
