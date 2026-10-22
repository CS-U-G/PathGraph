# PathGraph Recommendation Quality Benchmark Report

**Benchmark Version:** 1.0.0  
**Execution Date:** 2026-10-22T14:48:13.693224+00:00  
**Environment:** Python 3.13.7 on Windows-11-10.0.26200-SP0  
**Graph Scale:** 96 occupations, 13,939 skills, 125 curated prerequisite edges  

---

## 1. Executive Summary & Aggregate Performance

| Evaluation Metric | Aggregate Mean | Target / Ideal | Status |
| :--- | :---: | :---: | :---: |
| **Precision@5** | `0.0000` | > 0.7000 | INSPECT |
| **Precision@10** | `0.0600` | > 0.6000 | INSPECT |
| **Recall@5** | `0.0000` | > 0.5000 | INSPECT |
| **Recall@10** | `0.0800` | > 0.7000 | INSPECT |
| **NDCG@5** | `0.0000` | > 0.7500 | INSPECT |
| **NDCG@10** | `0.0573` | > 0.7000 | INSPECT |
| **Mean Average Precision (MAP)** | `0.1219` | > 0.6000 | INSPECT |
| **Essential Curriculum Coverage** | `90.5%` | 100.0% | PASS |
| **Prerequisite Violation Rate** | `0.0%` | 0.0% | PASS |
| **Curriculum Bloat Ratio** | `11.80x` | 1.0x - 2.5x | BLOAT |
| **Negative Control Leakage Rate** | `0.0%` | 0.0% | PASS |

---

## 2. Per-Persona Evaluation Breakdown

| Persona ID | Target Occupation | P@5 | R@5 | NDCG@5 | MAP | Coverage | Prereq Violations | Bloat Ratio | Leakage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `persona_frontend_to_fullstack` | software developer | `0.00` | `0.00` | `0.00` | `0.10` | `86%` | `0/1` (`0.0%`) | `7.14x` | `0.0%` |
| `persona_manual_qa_to_sdet` | software tester | `0.00` | `0.00` | `0.00` | `0.32` | `100%` | `0/2` (`0.0%`) | `10.00x` | `0.0%` |
| `persona_data_analyst_to_ml` | data scientist | `0.00` | `0.00` | `0.00` | `0.02` | `67%` | `0/0` (`0.0%`) | `16.67x` | `0.0%` |
| `persona_sysadmin_to_devops` | cloud DevOps engineer | `0.00` | `0.00` | `0.00` | `0.08` | `100%` | `0/0` (`0.0%`) | `12.67x` | `0.0%` |
| `persona_helpdesk_to_security` | ICT security technician | `0.00` | `0.00` | `0.00` | `0.10` | `100%` | `0/2` (`0.0%`) | `12.50x` | `0.0%` |

---

## 3. Metric Definitions & Formulas

1. **Precision@k**: Proximity of top-k recommended skills to expert-verified target competencies:
   $$\text{Precision@k} = \frac{|\text{Unique Recommended Top-k} \cap \text{Ground Truth Relevant}|}{k}$$

2. **Recall@k**: Proportion of missing essential gap requirements retrieved in top-k:
   $$\text{Recall@k} = \frac{|\text{Unique Recommended Top-k} \cap \text{Ground Truth Missing Essential}|}{|\text{Ground Truth Missing Essential}|}$$

3. **NDCG@k (Normalized Discounted Cumulative Gain)**: Rank-weighted position discount of relevant skills:
   $$\text{DCG@k} = \sum_{i=1}^{k} \frac{\text{rel}_i}{\log_2(i + 1)}, \quad \text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

4. **Prerequisite Violation Rate**: Percentage of active instructional pairs where dependent precedes prerequisite:
   $$\text{Violation Rate} = \frac{|\{(u, v) \in E_{\text{prereq}} : \text{pos}(u) > \text{pos}(v)\}|}{|\{(u, v) \in E_{\text{prereq}} : u, v \in S\}|}$$

5. **Curriculum Bloat Ratio**: Ratio of total scheduled learning path steps to core verified essential requirements:
   $$\text{Bloat Ratio} = \frac{\text{Scheduled Path Length}}{\max(1, |\text{Verified Missing Essential}|)}$$

6. **Negative Control Leakage Rate**: Frequency of cross-domain irrelevant control skills surfaced:
   $$\text{Leakage Rate} = \frac{|S_{\text{recommended}} \cap S_{\text{irrelevant negative controls}}|}{|S_{\text{recommended}}|}$$

---

## 4. Observed Weaknesses & Architectural Next Actions

### Measured Weaknesses:
- Curriculum bloat ratio is elevated (mean 11.80x). Generated learning paths schedule an average of several times more skills than the core essential gap, diluting learner focus.
- Prerequisite violation rate is 0.0% across all evaluated instructional pairs, confirming DAG ordering integrity where prerequisite data exists.
- Zero cross-domain negative control leakage observed (0.0% leakage rate across all personas).

### Recommended Action Items:
1. Introduce curriculum budgeting / max_path_length constraints in PathEngine to bound optional skill expansion.

---
*Report generated automatically by PathGraph AI Benchmark Runner.*