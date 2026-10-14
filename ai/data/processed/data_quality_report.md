# ESCO v1.2.1 Data Quality & Ingestion Profiling Report

**Generated:** 2026-10-14T07:28:15.616710+00:00  
**ESCO Release Version:** v1.2.1  
**Filter Rule Version:** 1.0.0  
**Pipeline Duration:** 55.8s  

---

## 1. Executive Summary & Reconciliation of True Record Counts

All canonical CSV files were ingested and profiled using an **RFC 4180 compliant parser** (`csv.reader`) to properly preserve multiline quoted fields in descriptions and alternative labels.

### 1.1 Skills & Occupations Reconciliation
- **Raw Skill Rows in CSV:** **13,960** physical rows.
- **Unique Skill Concept URIs:** **13,939** unique skills (exactly 21 identical duplicate rows present in the raw ESCO release were deduplicated by canonical `conceptUri`). `skills.json` exports exactly these **13,939** unique skills.
- **Raw Occupation Rows in CSV:** **3,043** physical rows.
- **Unique Occupation Concept URIs:** **3,039** unique occupations (exactly 4 identical duplicate rows present in the raw ESCO release were deduplicated by canonical `conceptUri`).

| Entity Category | True Measured Count | Verification Note |
| :--- | :---: | :--- |
| **Raw Occupation Rows** | **3,043** | Physical lines in `occupations_en.csv` |
| **Unique Occupations** | **3,039** | Deduplicated by `conceptUri` |
| **Raw Skill Rows** | **13,960** | Physical lines in `skills_en.csv` |
| **Unique Skills (`skills.json`)** | **13,939** | Deduplicated by `conceptUri` |
| **Total Occupation-Skill Edges** | **126,051** | Exact rows in `occupationSkillRelations_en.csv` |
| **Essential Relations** | **67,600** | Core requirement edges |
| **Optional Relations** | **58,451** | Elective requirement edges |
| **Skill-to-Skill Edges** | **5,818** | Associative edges in `skillSkillRelations_en.csv` |
| **Broader Skill Pillar Edges** | **20,819** | Taxonomy edges in `broaderRelationsSkillPillar_en.csv` |
| **Total Alternative Labels** | **117,107** | Combined altLabels in skills and occupations |
| **Unique Skill Synonyms / Aliases** | **85,916** | Distinct alternative labels in skills |

---

## 2. Occupation Reconciliation & Structural ISCO Breakdown

Re-parsing `occupations_en.csv` directly with full RFC 4180 compliance confirms that the **true, exact count of core computing occupations is 96**.

| ISCO-08 Code | ISCO Group Title | True Measured Occupations | Decision Tier |
| :---: | :--- | :---: | :--- |
| `1330` | ICT service managers | **13** | Tier 1 (Whitelisted) |
| `2511` | Systems analysts | **21** | Tier 1 (Whitelisted) |
| `2512` | Software developers | **10** | Tier 1 (Whitelisted) |
| `2513` | Web and multimedia developers | **5** | Tier 1 (Whitelisted) |
| `2514` | Applications programmers | **7** | Tier 1 (Whitelisted) |
| `2519` | Software and applications developers NEC | **12** | Tier 1 (Whitelisted) |
| `2521` | Database designers and administrators | **5** | Tier 1 (Whitelisted) |
| `2522` | Systems administrators | **3** | Tier 1 (Whitelisted) |
| `2523` | Computer network professionals | **3** | Tier 1 (Whitelisted) |
| `2529` | Database and network professionals NEC / Cybersecurity | **9** | Tier 1 (Whitelisted) |
| `3511` | ICT operations technicians | **1** | Tier 1 (Whitelisted) |
| `3512` | ICT user support technicians | **4** | Tier 1 (Whitelisted) |
| `3513` | Computer network and systems technicians | **2** | Tier 1 (Whitelisted) |
| `3514` | Web technicians | **1** | Tier 1 (Whitelisted) |
| `3521` | Broadcasting and audiovisual technicians | **13** | Excluded (Outright False Positives) |
| `3522` | Telecommunications engineering technicians | **1** | Tier 2 (Candidate - Hardware/Network) |
| `2153` | Telecommunications engineers | **2** | Tier 2 (Candidate - Engineering) |
| `2166` | Graphic and multimedia designers | **16** | Tier 2 (Candidate - Design/UX) |
| `2131` | Biologists (includes Bioinformatics) | **19** | Tier 2 (Candidate - Life Sciences) |
| `1342` | Health services managers (includes Clinical Informatics) | **2** | Tier 2 (Candidate - Healthcare) |

**Core Whitelist Sum (1330 + 25xx + 351x):** **96 occupations**  
**Total in ISCO 133 + 25 + 35:** **110 occupations**  

---

## 3. Digital-Skill Ratio Formal Definition

The candidate qualification signal for Tier 2 occupations is formally defined as:

$$R_{\text{digital}}(occ) = \frac{|\text{EssentialSkills}(occ) \cap (\text{DigitalSkillsCollection} \cup \text{ISCED\_F\_06})|}{|\text{EssentialSkills}(occ)|}$$

- **Essential Skills Only:** Evaluated strictly on skills where `relationType == 'essential'`. Optional skills are excluded to avoid inflationary noise.
- **Digital Taxonomies Included:**
  1. All 1,286 curated skills in `digitalSkillsCollection_en.csv`.
  2. All skill concepts linked to ISCED-F broad field `06` ('Information and Communication Technologies') or narrow groups `0611`, `0612`, `0613`, `0619`, `0688` via `broaderRelationsSkillPillar_en.csv`.
- **Status:** As mandated, Tier 2 candidates remain **UNVALIDATED** until the 90-occupation validation sample in `data/interim/filter_validation_sample.csv` has been labelled by domain reviewers.

---

## 4. Skill Versatility Calculation

In compliance with Section 4, Item 4 of the Master Build Prompt, `skill_versatility` was computed across the **3,039 unique occupations** before filtering:

$$\text{skill\_versatility}(s) = \frac{|\{occ \in \text{UniqueOccupations} \mid (occ, s) \in \text{Edges}\}|}{\max_{s'} |\{occ \in \text{UniqueOccupations} \mid (occ, s') \in \text{Edges}\}|}$$

- Denominator universe: strictly the **3,039 unique occupations**.
- This metric directly populates the contract's `demand` field.
- High versatility represents broad cross-occupational transferability, not dynamic job posting trends.

---

## 5. Referential Integrity & Data Hygiene

Cross-file foreign key validation was executed across all raw tables:

- `occupationSkillRelations.occupationUri` $\rightarrow$ `occupations.conceptUri`: **0 broken references**
- `occupationSkillRelations.skillUri` $\rightarrow$ `skills.conceptUri`: **0 broken references**
- `skillSkillRelations.originalSkillUri` $\rightarrow$ `skills.conceptUri`: **0 broken references**
- `skillSkillRelations.relatedSkillUri` $\rightarrow$ `skills.conceptUri`: **0 broken references**
- Primary Key Uniqueness: **0 duplicate conceptUri primary keys** after URI deduplication.
- Preferred Label Uniqueness: **0 duplicate preferredLabel** values within occupations and skills.

---

## 6. Generated Artifacts & Integrity Checksums

| Artifact File | Description | SHA-256 Checksum |
| :--- | :--- | :--- |
| `occupations.json` | Processed / Interim artifact | `af38fc44aa72cea2...` |
| `skills.json` | Processed / Interim artifact | `d91ec75d8f6d34e5...` |
| `occupation_skill_edges.json` | Processed / Interim artifact | `fecdc04c68de7898...` |
| `skill_hierarchy.json` | Processed / Interim artifact | `11901c85b58a7e3c...` |
| `skill_associations.json` | Processed / Interim artifact | `1472b1b2dda23f18...` |
| `computing_occupation_filter_audit.json` | Processed / Interim artifact | `025de8cabb7f1ee2...` |
| `filter_validation_sample.csv` | Processed / Interim artifact | `1fc2f42b313d0b1d...` |
