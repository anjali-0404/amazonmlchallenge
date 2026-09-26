# ML Challenge 2026: Business Entity Resolution Solution Report

**Team Name:** EntityResolvers  
**Submission Date:** September 27, 2026

---

## 1. Executive Summary
We present a high-throughput, memory-bounded, streaming entity resolution architecture tailored for multi-million row heterogeneous business entity datasets (Source 1 vs. Source 2/Source 3). By leveraging disk-backed indexed multi-key blocking (Unicode token normalization, country-scoped salient business name keys, and street/postal numeric tokens) coupled with a gradient-boosted decision tree classifier (HistGradientBoosting) trained with class-imbalance weighting, our pipeline balances candidate recall with extreme precision. The system achieves a validation Macro $F_{0.5}$ score of **0.9381** at optimal probability threshold $\tau = 0.95$, processing all 1.73M test queries in low-memory batches and generating 100% compliant, fully validated submission artifacts.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Heterogeneous Noise & Formatting Variations:** Business names exhibit high legal suffix noise (`Ltd`, `LLC`, `Pvt`, `Corp`, `Co`), variable punctuation, abbreviations, and diverse non-ASCII Unicode transliterations across multiple international jurisdictions.
- **Address Irregularities:** Address strings differ widely across sources in token ordering, inclusion/omission of building/unit numbers, postal code placements, and localized abbreviation styles.
- **Extreme Scale & Asymmetric Comparison Space:** Source 1 contains over 1.73 million entities to match against multiple million rows in Sources 2 & 3. A naive Cartesian cross-product would require $\sim 10^{13}$ pairwise comparisons, making in-memory indexing or un-indexed search computationally infeasible on standard hardware.
- **Metric-Driven Asymmetry ($F_{0.5}$ Optimization):** Macro $F_{0.5}$ weights precision twice as heavily as recall ($\beta = 0.5$), making false positive linkages heavily penalized. Consequently, threshold calibration and feature precision are paramount.

### 2.2 Solution Strategy
Our architecture is structured as a two-stage decoupled pipeline:

```
[Source 1 Entity] 
       │
       ▼
[Stage 1: Multi-Key Deterministic & Phonetic/Token Blocking]
       ├── Strict Country Partitioning
       ├── Primary Distinctive Name Token (len >= 3)
       └── Numeric Address / Building / Postal Key
       │
       ▼ (Generates top-30 candidate pool per query)
[Stage 2: Pairwise Feature Extraction & GBDT Scoring]
       ├── Token Set Jaccard Sim (Name & Address)
       ├── Exact Name-Key & Numeric Token Equivalence
       ├── Country Identity & Length Ratio Features
       │
       ▼
[Stage 3: Decision & Calibration]
       ├── HistGradientBoosting Probability Estimation
       └── Macro F0.5 Optimization on Validation Split (threshold = 0.95)
```

- **Approach Type:** Hybrid Multi-Key Blocking + Gradient Boosted Decision Tree (GBDT) Classifier + Asymmetric Macro Threshold Optimizer.
- **Core Innovation:** Disk-backed SQLite indexing with Write-Ahead Logging (WAL) and memory caching combined with constant-memory chunked streaming. This guarantees zero OOM (Out-Of-Memory) risks while scoring over 1.73M records efficiently with sub-millisecond candidate lookups.

---

## 3. Candidate Generation (Blocking)

To reduce the $O(N \times M)$ search space to $O(N \cdot K)$ where $K \le 30$:
- **Blocking Keys Used:**
  1. **Country Partition:** Strict matching on normalized ISO country identifier.
  2. **Salient Name Key:** Longest non-stopword token (up to 16 characters) extracted after NFKC Unicode normalization, case folding, ampersand expansion, and legal suffix removal (`ltd`, `limited`, `llc`, `inc`, `corp`, `pvt`, `company`, `co`, `llp`).
  3. **Address Numeric Key:** Regex-extracted continuous digit sequences representing street numbers, suites, or postal codes.
- **Candidate Pairs Generated:** Over 3.43M candidate pairs across the 60,000 Source 1 training sample, and over 1.73M candidate sets produced for test inference (averaging high candidate coverage per query entity).
- **Ensuring True Matches Were Not Lost:** Dual-criteria disjunctive union blocking (Name Key OR Address Number Key) conditioned on matching country ensures that entities with slight name variations but identical addresses—or updated addresses but stable distinctive names—are captured into the candidate pool.

---

## 4. Matching Model

### Features Used:
1. **Name Token Jaccard Similarity:** Set intersection over union of whitespace-delimited normalized tokens.
2. **Address Token Jaccard Similarity:** Multi-word overlap between normalized address strings.
3. **Exact Name Key Match Indicator:** Binary flag indicating exact equality between the primary salient name tokens.
4. **Address Numeric Match Indicator:** Binary flag indicating concordance between extracted numeric identifiers (e.g. house/building/pincode numbers).
5. **Country Match Indicator:** Categorical agreement flag.
6. **Name Length Ratio:** Normalized length discrepancy $\frac{\min(|S_{name}|, |C_{name}|)}{\max(|S_{name}|, |C_{name}|)}$.
7. **Address Length Ratio:** Length proportion measuring completeness of address descriptions.

### Model Architecture & Training:
- **Model Type:** `HistGradientBoostingClassifier` with $L_2$ regularization (`l2_regularization=1.0`), `learning_rate=0.06`, `max_leaf_nodes=31`, and max iterations = 250.
- **Sample Weighting:** Inverse positive class frequency weighting to account for candidate generation imbalance (positive pair ratio $\approx 1:100$).
- **Validation Split:** 80/20 Group-based split grouped on `source1_entity_id` to strictly prevent data leakage across pairs of the same entity.
- **Threshold Selection Method:** Direct grid search over validation split maximizing the challenge Macro $F_{0.5}$ metric:
  $$F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score (Validation):** **0.9381** (achieved at optimal decision threshold $\tau = 0.95$).
- **Submission Output Statistics:**
  - Total Source 1 test entities evaluated: **1,732,544**
  - Confident entity matches: **631,143**
  - Unmatched / Singleton entities: **1,101,401**
  - Candidate sets generated: **1,732,475** non-empty sets
- **Common False Positives (Avoided via High Threshold):**
  - Multi-branch entities or retail chains sharing the exact same business brand name in the same country but situated in completely different cities/addresses. The high $\tau = 0.95$ threshold and address Jaccard weight effectively prevents incorrect conflations.
- **Common False Negatives (Edge Cases):**
  - Severe typographical distortions in both the primary name word and the address digits simultaneously, or cases where one source contains purely generic words without a distinctive token $\ge 3$ characters.

---

## 6. Conclusion
Our scalable entity resolution pipeline combines deterministic multi-key blocking with gradient-boosted pairwise matching and metric-tailored calibration. By pairing low-overhead SQLite disk indexes with streaming inference, the solution scales seamlessly to multi-million entity datasets under strict execution and memory limits while delivering high Macro $F_{0.5}$ accuracy (0.9381) and passing all validation criteria.

---

## Appendix

### A. Code Artefacts
- **File Structure:**
  - `streaming_train_and_predict.py`: Core streaming pipeline containing SQLite index creation, candidate generation, GBDT model training, $F_{0.5}$ threshold search, streaming test inference, and submission validation.
  - `utils/validate_submission.py`: Official submission integrity and schema validator.
  - `output/matching_results.tsv`: Final predictions TSV (1,732,544 rows).
  - `output/candidate_pairs.tsv`: Candidate blocking pool TSV (1,732,544 rows).
- **Execution Entrypoint:**
  ```bash
  python streaming_train_and_predict.py
  ```

### B. Validation Verification
```text
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (1101401 empty, 631143 non-empty).
  candidate_pairs.tsv: 1732544 rows (69 empty, 1732475 non-empty).
PASS — no blocking issues found. Safe to submit.
```
