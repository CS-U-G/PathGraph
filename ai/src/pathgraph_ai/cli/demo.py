#!/usr/bin/env python3
"""PathGraph AI - Local Interactive Terminal Demo.

Demonstrates the core capabilities of PathGraph using official ESCO data
and precomputed dense vector embeddings:
1. Search and select real ESCO occupations (3,039 occupations).
2. Input user skills and resolve them to stable ESCO skill URIs (13,939 skills).
3. Display recognized skills and unmatched inputs.
4. Perform skill-gap analysis (exact URI matches vs semantic candidate review).
5. Display matched essential/optional, missing essential/optional, and match percentages.
6. Display semantic candidate suggestions separately from confirmed skill matches
   with full pedagogical disclaimers.
7. Generate personalized curriculum learning paths using Kahn's topological sort.
8. Clearly disclose prerequisite assumptions and fallback orderings.
9. Run 100% locally with zero external AI APIs, API keys, or model re-downloads.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Reconfigure standard output for UTF-8 compatibility on Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathgraph_ai.domain.knowledge_graph import (
    ESCOKnowledgeGraph,
    OccupationNode,
    SkillNode,
)
from pathgraph_ai.embeddings.artifact import (
    EmbeddingArtifact,
    load_embeddings_artifact,
)
from pathgraph_ai.pipelines.data_loader import ESCODataLoader
from pathgraph_ai.recommenders.path_generator import (
    LearningPath,
    LearningPathGenerator,
    LearningStep,
    PrerequisiteGraph,
)
from pathgraph_ai.recommenders.skill_gap import (
    MatchedSkill,
    MissingSkill,
    SemanticCandidate,
    SkillGapAnalyzer,
    SkillGapReport,
)
from pathgraph_ai.settings import settings


# -----------------------------------------------------------------------------
# Color and Terminal Helpers
# -----------------------------------------------------------------------------

class Term:
    """Terminal styling and formatting helper."""
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    RESET = "\033[0m"

    @classmethod
    def strip(cls, text: str) -> str:
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)


def banner(title: str, width: int = 76) -> str:
    """Creates a framed section banner."""
    line = "=" * width
    return f"\n{Term.BOLD}{Term.CYAN}{line}\n  {title}\n{line}{Term.RESET}\n"


def subbanner(title: str, width: int = 76) -> str:
    """Creates a minor section divider."""
    line = "-" * width
    return f"\n{Term.BOLD}{line}\n  {title}\n{line}{Term.RESET}"


# -----------------------------------------------------------------------------
# Skill Resolution Engine
# -----------------------------------------------------------------------------

@dataclass
class ResolvedSkill:
    """Represents a skill successfully mapped from user input to an ESCO concept."""
    input_text: str
    skill_node: SkillNode
    match_source: str


class SkillResolver:
    """Resolves arbitrary user text inputs to verified ESCO skill concept URIs."""

    def __init__(self, kg: ESCOKnowledgeGraph) -> None:
        self.kg = kg
        self.label_to_skills: dict[str, list[SkillNode]] = defaultdict(list)
        self.alt_label_to_skills: dict[str, list[SkillNode]] = defaultdict(list)
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Indexes all ESCO skills for fast and deterministic resolution."""
        for skill in self.kg.skills.values():
            lbl_clean = skill.label.strip().lower()
            self.label_to_skills[lbl_clean].append(skill)
            if skill.alt_labels:
                for alt_line in skill.alt_labels.split("\n"):
                    clean_alt = alt_line.strip().lower()
                    if clean_alt and clean_alt != lbl_clean:
                        self.alt_label_to_skills[clean_alt].append(skill)

    def search_skills(self, query: str, limit: int = 10) -> list[tuple[SkillNode, str]]:
        """Searches ESCO skills for a query string, returning ranked candidate matches."""
        q = query.strip().lower()
        if not q:
            return []

        # 1. Exact URI lookup
        if q in self.kg.skills:
            return [(self.kg.skills[q], "exact_uri")]
        for uri, skill in self.kg.skills.items():
            if uri.lower() == q:
                return [(skill, "exact_uri")]

        results: list[tuple[SkillNode, str]] = []
        seen_uris: set[str] = set()

        # 2. Exact preferred label match
        if q in self.label_to_skills:
            for s in self.label_to_skills[q]:
                if s.uri not in seen_uris:
                    results.append((s, "exact_label"))
                    seen_uris.add(s.uri)

        # 3. Exact alternative label match
        if q in self.alt_label_to_skills:
            for s in self.alt_label_to_skills[q]:
                if s.uri not in seen_uris:
                    results.append((s, "exact_alt"))
                    seen_uris.add(s.uri)

        # 4. Preferred label starts with query
        for s in self.kg.skills.values():
            if s.uri in seen_uris:
                continue
            if s.label.lower().startswith(q):
                results.append((s, "prefix_label"))
                seen_uris.add(s.uri)

        # 5. Whole word match in preferred label
        pattern = r"\b" + re.escape(q) + r"\b"
        for s in self.kg.skills.values():
            if s.uri in seen_uris:
                continue
            if re.search(pattern, s.label.lower()):
                results.append((s, "word_match_label"))
                seen_uris.add(s.uri)

        # 6. Substring match in preferred label
        for s in self.kg.skills.values():
            if s.uri in seen_uris:
                continue
            if q in s.label.lower():
                results.append((s, "substring_label"))
                seen_uris.add(s.uri)

        # 7. Substring match in alternative labels
        for s in self.kg.skills.values():
            if s.uri in seen_uris:
                continue
            alt = (s.alt_labels or "").lower()
            if q in alt:
                results.append((s, "substring_alt"))
                seen_uris.add(s.uri)

        return results[:limit]

    def resolve_best(self, query: str) -> ResolvedSkill | None:
        """Resolves an input query to the single best ESCO skill match, or None."""
        matches = self.search_skills(query, limit=1)
        if matches:
            node, reason = matches[0]
            return ResolvedSkill(input_text=query.strip(), skill_node=node, match_source=reason)
        return None

    def resolve_batch(
        self, inputs: Sequence[str]
    ) -> tuple[list[ResolvedSkill], list[str]]:
        """Resolves a collection of user skill inputs into recognized vs unmatched lists."""
        recognized: list[ResolvedSkill] = []
        unmatched: list[str] = []
        seen_uris: set[str] = set()

        for raw_input in inputs:
            text = raw_input.strip()
            if not text:
                continue
            match = self.resolve_best(text)
            if match is not None:
                if match.skill_node.uri not in seen_uris:
                    recognized.append(match)
                    seen_uris.add(match.skill_node.uri)
            else:
                unmatched.append(text)

        return recognized, unmatched


# -----------------------------------------------------------------------------
# Pre-Configured Sample Profiles for Quick Demonstrations
# -----------------------------------------------------------------------------

SAMPLE_PROFILES: dict[str, dict[str, Any]] = {
    "synthetic": {
        "name": "Phase 3 Verified Synthetic Learner",
        "description": "Standard deterministic profile from test_end_to_end_integration.py.",
        "skills": [
            "computer programming",
            "integrated development environment software",
            "Java",
            "execute software tests",
            "operate agricultural machinery",
            "UNKNOWN_SYNTHETIC_SKILL_999",
        ],
    },
    "junior_dev": {
        "name": "Junior Python Developer",
        "description": "Learner with core Python, SQL, debugging, and git knowledge.",
        "skills": [
            "Python",
            "SQL",
            "debug software",
            "version control systems",
            "use object-oriented programming",
        ],
    },
    "full_match": {
        "name": "Senior Software Developer (Essential Coverage)",
        "description": "Simulates 100% coverage of essential software developer competencies.",
        "skills": [
            "computer programming",
            "debug software",
            "analyse software specifications",
            "define technical requirements",
            "develop software prototype",
            "use software libraries",
            "integrated development environment software",
            "utilise computer-aided software engineering tools",
            "software anomalies",
            "ICT debugging tools",
            "create flowchart diagram",
            "interpret technical requirements",
            "execute software tests",
        ],
    },
    "empty": {
        "name": "Complete Beginner / Empty Profile",
        "description": "Empty skill profile (0 skills) to verify 0% baseline and fallback path.",
        "skills": [],
    },
}


# -----------------------------------------------------------------------------
# Demo Application Controller
# -----------------------------------------------------------------------------

class PathGraphTerminalDemo:
    """Interactive command-line demonstration for PathGraph AI."""

    def __init__(self) -> None:
        self.kg: ESCOKnowledgeGraph | None = None
        self.analyzer: SkillGapAnalyzer | None = None
        self.resolver: SkillResolver | None = None
        self.path_generator: LearningPathGenerator = LearningPathGenerator()

        # State
        self.current_occupation: OccupationNode | None = None
        self.user_skills: dict[str, ResolvedSkill] = {}
        self.unmatched_inputs: list[str] = []
        self.last_gap_report: SkillGapReport | None = None
        self.last_learning_path: LearningPath | None = None

        # Settings
        self.top_k_candidates: int = 3
        self.similarity_threshold: float | None = 0.55
        self.candidate_pool: str = "user_skills"
        self.use_sample_dag: bool = False

    def initialize(self) -> bool:
        """Validates filesystem artifacts and loads ESCO knowledge graph & precomputed vectors."""
        print(banner("PATHGRAPH AI - LOCAL INTERACTIVE TERMINAL DEMO"))
        print(f"{Term.BOLD}Checking local environment and datasets...{Term.RESET}")

        loader = ESCODataLoader(raw_data_dir=settings.raw_data_dir)
        occ_csv = loader.get_file_path("occupations_en.csv")
        skill_csv = loader.get_file_path("skills_en.csv")
        rel_csv = loader.get_file_path("occupationSkillRelations_en.csv")

        missing_raw = []
        for name, p in [("occupations_en.csv", occ_csv), ("skills_en.csv", skill_csv), ("occupationSkillRelations_en.csv", rel_csv)]:
            if not p.exists():
                missing_raw.append(name)

        if missing_raw:
            print(f"{Term.RED}[ERROR] Required ESCO CSV files are missing in: {settings.raw_data_dir}{Term.RESET}")
            for m in missing_raw:
                print(f"        - {m}")
            return False

        if not settings.skills_embeddings_path.exists():
            print(f"{Term.RED}[ERROR] Precomputed skills embedding artifact not found at:{Term.RESET}")
            print(f"        {settings.skills_embeddings_path}")
            return False

        try:
            print(" -> Loading ESCO Knowledge Graph (3,039 occupations, 13,939 skills)...")
            self.kg = ESCOKnowledgeGraph.from_loader(loader=loader, load_hierarchy=True)
            self.resolver = SkillResolver(self.kg)

            print(" -> Initializing SkillGapAnalyzer with precomputed embeddings (.npz)...")
            self.analyzer = SkillGapAnalyzer(knowledge_graph=self.kg, artifact_path=settings.skills_embeddings_path)
            artifact = self.analyzer.skills_artifact

            print(f"{Term.GREEN}[OK] Initialized successfully:{Term.RESET}")
            print(f"     * Occupations loaded: {self.kg.num_occupations:,}")
            print(f"     * Skills loaded:      {self.kg.num_skills:,}")
            print(f"     * Graph relations:    {len(self.kg.occupation_skill_relations):,} occ-skill edges")
            print(f"     * Embedding matrix:   {artifact.count:,} skills x {artifact.dimension}D float32")
            print(f"     * Execution mode:     100% Local / Zero AI API keys / Zero model re-downloads\n")
            return True

        except Exception as e:
            print(f"\n{Term.RED}[FATAL ERROR] Initialization failed: {e}{Term.RESET}\n")
            return False

    def search_occupations(self, query: str, limit: int = 10) -> list[OccupationNode]:
        """Searches ESCO occupations by title, code, or alternative labels."""
        if not self.kg:
            return []
        q = query.strip().lower()
        if not q:
            return []

        if q in self.kg.occupations:
            return [self.kg.occupations[q]]
        for uri, occ in self.kg.occupations.items():
            if uri.lower() == q:
                return [occ]

        exact_matches: list[OccupationNode] = []
        prefix_matches: list[OccupationNode] = []
        word_matches: list[OccupationNode] = []
        substr_matches: list[OccupationNode] = []
        seen: set[str] = set()

        pat = r"\b" + re.escape(q) + r"\b"

        for occ in self.kg.occupations.values():
            lbl = occ.label.lower()
            code = (occ.code or "").lower()
            alt = (occ.alt_labels or "").lower()

            if lbl == q or code == q:
                exact_matches.append(occ)
                seen.add(occ.uri)
            elif lbl.startswith(q):
                if occ.uri not in seen:
                    prefix_matches.append(occ)
                    seen.add(occ.uri)
            elif re.search(pat, lbl):
                if occ.uri not in seen:
                    word_matches.append(occ)
                    seen.add(occ.uri)
            elif q in lbl or q in alt:
                if occ.uri not in seen:
                    substr_matches.append(occ)
                    seen.add(occ.uri)

        ordered = exact_matches + prefix_matches + word_matches + substr_matches
        return ordered[:limit]

    def select_occupation_interactive(self) -> None:
        """Interactive flow to search and select a target ESCO occupation."""
        print(subbanner("OCCUPATION SEARCH & SELECTION"))
        print("Search the official ESCO catalog (3,039 occupations).")
        print("Examples: 'software developer', 'data scientist', 'nurse', 'civil engineer'\n")

        while True:
            try:
                query = input(f"{Term.BOLD}Enter occupation keyword or ESCO URI (or 'c' to cancel): {Term.RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return

            if not query or query.lower() == "c":
                return

            matches = self.search_occupations(query, limit=10)
            if not matches:
                print(f"{Term.YELLOW}No ESCO occupations found matching '{query}'. Please try another term.{Term.RESET}\n")
                continue

            print(f"\nFound {len(matches)} matching occupation(s):")
            for idx, occ in enumerate(matches, start=1):
                skills = self.kg.get_skills_for_occupation(occ.uri) if self.kg else []
                ess_count = sum(1 for _, r in skills if r == "essential")
                opt_count = sum(1 for _, r in skills if r == "optional")
                code_str = f" [Code: {occ.code}]" if occ.code else ""
                print(f"  [{idx:2d}] {Term.BOLD}{occ.label}{Term.RESET}{code_str}")
                print(f"       Total skills: {len(skills)} ({ess_count} essential, {opt_count} optional)")
                print(f"       URI: {Term.DIM}{occ.uri}{Term.RESET}")

            print("  [ 0] Search again / Cancel")

            while True:
                choice = input(f"\n{Term.BOLD}Select occupation number (0-{len(matches)}): {Term.RESET}").strip()
                if choice == "0":
                    break
                if choice.isdigit() and 1 <= int(choice) <= len(matches):
                    selected = matches[int(choice) - 1]
                    self.set_target_occupation(selected)
                    return
                print(f"{Term.RED}Invalid selection. Please enter a number between 0 and {len(matches)}.{Term.RESET}")

    def set_target_occupation(self, occ: OccupationNode) -> None:
        """Sets the active target occupation and invalidates previous gap reports."""
        self.current_occupation = occ
        self.last_gap_report = None
        self.last_learning_path = None
        skills = self.kg.get_skills_for_occupation(occ.uri) if self.kg else []
        ess = sum(1 for _, r in skills if r == "essential")
        opt = sum(1 for _, r in skills if r == "optional")

        print(f"\n{Term.GREEN}[✓] Active Target Occupation Set:{Term.RESET}")
        print(f"    Title:       {Term.BOLD}{occ.label}{Term.RESET}")
        print(f"    ESCO URI:    {occ.uri}")
        print(f"    ISCO Code:   {occ.code or occ.isco_group or 'N/A'}")
        print(f"    Total Skills Requirement: {len(skills)} ({ess} essential, {opt} optional)\n")

    def manage_skills_interactive(self) -> None:
        """Interactive menu for inputting, inspecting, and managing user skills."""
        while True:
            print(subbanner("USER SKILLS MANAGEMENT"))
            print(f"Current Profile: {Term.BOLD}{len(self.user_skills)}{Term.RESET} recognized skill(s)")
            if self.unmatched_inputs:
                print(f"Unmatched inputs on file: {Term.YELLOW}{len(self.unmatched_inputs)}{Term.RESET}")

            print("\nOptions:")
            print("  [1] Enter skills in bulk (comma-separated names or URIs)")
            print("  [2] Search and add skills interactively (one-by-one)")
            print("  [3] Load a pre-configured sample profile")
            print("  [4] View currently recognized skills and unmatched inputs")
            print("  [5] Clear / Reset current skill profile")
            print("  [0] Return to main menu")

            choice = input(f"\n{Term.BOLD}Enter choice (0-5): {Term.RESET}").strip()

            if choice == "1":
                self.add_skills_bulk_interactive()
            elif choice == "2":
                self.add_skill_interactive()
            elif choice == "3":
                self.load_sample_profile_interactive()
            elif choice == "4":
                self.display_skill_profile_breakdown()
            elif choice == "5":
                self.user_skills.clear()
                self.unmatched_inputs.clear()
                self.last_gap_report = None
                self.last_learning_path = None
                print(f"{Term.GREEN}[✓] Skill profile cleared.{Term.RESET}")
            elif choice == "0":
                break
            else:
                print(f"{Term.RED}Invalid option. Please enter 0-5.{Term.RESET}")

    def add_skills_bulk_interactive(self) -> None:
        """Prompts for comma-separated or multiline skill strings and resolves them."""
        print("\nEnter skill names or ESCO URIs (comma-separated or type/paste):")
        print("Example: Python, Java, debug software, Docker, SQL, invalid_xyz\n")
        try:
            raw = input(f"{Term.BOLD}Skills input: {Term.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return

        if not raw:
            print(f"{Term.YELLOW}No input provided.{Term.RESET}")
            return

        tokens = [t.strip() for t in re.split(r"[,;\n]+", raw) if t.strip()]
        self.process_raw_skills(tokens)

    def process_raw_skills(self, tokens: Sequence[str]) -> None:
        """Resolves raw skill tokens, updates state, and prints resolution summary."""
        if not self.resolver:
            print(f"{Term.RED}Skill resolver not initialized.{Term.RESET}")
            return

        print(f"\nResolving {len(tokens)} input(s) against 13,939 ESCO concepts...")
        recognized, unmatched = self.resolver.resolve_batch(tokens)

        for rec in recognized:
            self.user_skills[rec.skill_node.uri] = rec

        self.unmatched_inputs.extend([u for u in unmatched if u not in self.unmatched_inputs])
        self.last_gap_report = None
        self.last_learning_path = None

        self.display_skill_resolution_report(recognized, unmatched)

    def add_skill_interactive(self) -> None:
        """Allows searching skills with full disambiguation."""
        if not self.resolver:
            return
        while True:
            try:
                query = input(f"\n{Term.BOLD}Enter skill keyword to search (or 'c' to finish): {Term.RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return

            if not query or query.lower() == "c":
                break

            matches = self.resolver.search_skills(query, limit=8)
            if not matches:
                print(f"{Term.YELLOW}No ESCO skills found matching '{query}'. Recorded as unmatched.{Term.RESET}")
                if query not in self.unmatched_inputs:
                    self.unmatched_inputs.append(query)
                continue

            print(f"\nMatches for '{query}':")
            for idx, (s, reason) in enumerate(matches, start=1):
                status = " (ALREADY IN PROFILE)" if s.uri in self.user_skills else ""
                print(f"  [{idx}] {Term.BOLD}{s.label}{Term.RESET}{status}")
                print(f"      Type: {s.skill_type or 'N/A'} | Reuse: {s.reuse_level or 'N/A'} | Match: {reason}")
                print(f"      URI:  {Term.DIM}{s.uri}{Term.RESET}")

            print("  [0] Skip / search again")

            choice = input(f"Select skill to add (0-{len(matches)}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(matches):
                chosen_node, match_src = matches[int(choice) - 1]
                rec = ResolvedSkill(input_text=query, skill_node=chosen_node, match_source=match_src)
                self.user_skills[chosen_node.uri] = rec
                print(f"{Term.GREEN}[✓] Added '{chosen_node.label}' to profile.{Term.RESET}")
                self.last_gap_report = None
                self.last_learning_path = None

    def load_sample_profile_interactive(self) -> None:
        """Allows loading pre-configured sample learner profiles."""
        print("\nSelect a pre-configured sample profile:")
        keys = list(SAMPLE_PROFILES.keys())
        for idx, k in enumerate(keys, start=1):
            prof = SAMPLE_PROFILES[k]
            print(f"  [{idx}] {Term.BOLD}{prof['name']}{Term.RESET}")
            print(f"      {prof['description']}")
            print(f"      Skills ({len(prof['skills'])}): {', '.join(prof['skills'][:4])}...")

        print("  [0] Cancel")
        choice = input(f"\nEnter choice (0-{len(keys)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(keys):
            selected_key = keys[int(choice) - 1]
            prof = SAMPLE_PROFILES[selected_key]
            self.user_skills.clear()
            self.unmatched_inputs.clear()
            print(f"\nLoading profile '{prof['name']}'...")
            self.process_raw_skills(prof["skills"])

    def display_skill_resolution_report(
        self, recognized: list[ResolvedSkill], unmatched: list[str]
    ) -> None:
        """Displays clear resolution breakdown of recognized skills vs unmatched inputs."""
        print(banner("SKILL RESOLUTION BREAKDOWN"))

        print(f"{Term.BOLD}{Term.GREEN}[✓] RECOGNIZED ESCO SKILLS ({len(recognized)}):{Term.RESET}")
        if recognized:
            for idx, r in enumerate(recognized, start=1):
                s = r.skill_node
                print(f"  {idx:2d}. {Term.BOLD}{s.label}{Term.RESET} (from: \"{r.input_text}\")")
                print(f"      URI:    {Term.DIM}{s.uri}{Term.RESET}")
                print(f"      Type:   {s.skill_type or 'skill/competence'} | Method: {r.match_source}")
        else:
            print("  (None recognized in this batch)")

        print(f"\n{Term.BOLD}{Term.YELLOW}[✗] UNMATCHED INPUTS ({len(unmatched)}):{Term.RESET}")
        if unmatched:
            print("  The following terms could not be mapped to any ESCO skill concepts:")
            for u in unmatched:
                print(f"  - \"{u}\" (No ESCO preferredLabel or altLabel match)")
        else:
            print("  (None — all inputs successfully resolved)")
        print()

    def display_skill_profile_breakdown(self) -> None:
        """Shows currently active user profile skills and unmatched terms."""
        print(banner("ACTIVE USER SKILL PROFILE"))
        print(f"Total Confirmed ESCO Skills: {Term.BOLD}{len(self.user_skills)}{Term.RESET}")
        if self.user_skills:
            for idx, (uri, r) in enumerate(self.user_skills.items(), start=1):
                s = r.skill_node
                print(f"  {idx:2d}. {Term.BOLD}{s.label}{Term.RESET}")
                print(f"      URI:  {Term.DIM}{uri}{Term.RESET}")
                print(f"      Type: {s.skill_type or 'N/A'} | Source: {r.match_source}")
        else:
            print("  (Profile is currently empty. Add skills to test matching.)")

        if self.unmatched_inputs:
            print(f"\nUnmatched inputs recorded ({len(self.unmatched_inputs)}):")
            for u in self.unmatched_inputs:
                print(f"  - \"{u}\"")
        print()

    def run_skill_gap_analysis(self) -> SkillGapReport | None:
        """Executes exact and semantic skill-gap analysis over target occupation."""
        if not self.current_occupation:
            print(f"\n{Term.YELLOW}[!] No target occupation selected. Please select an occupation first.{Term.RESET}")
            self.select_occupation_interactive()
            if not self.current_occupation:
                return None

        if not self.analyzer:
            print(f"{Term.RED}[ERROR] SkillGapAnalyzer not available.{Term.RESET}")
            return None

        occ = self.current_occupation
        user_uris = set(self.user_skills.keys())

        print(banner(f"SKILL GAP ANALYSIS: {occ.label.upper()}"))
        print(f"Target Occupation: {Term.BOLD}{occ.label}{Term.RESET} ({occ.uri})")
        print(f"Learner Profile:   {len(user_uris)} confirmed ESCO skill URIs")
        print("Evaluation Mode:   Exact URI Matching + Precomputed Semantic Candidate Review")
        print(f"Candidate Pool:    {self.candidate_pool} | Threshold: {self.similarity_threshold or 'None'}")
        print("Running analysis...\n")

        report = self.analyzer.analyze_gap_semantic(
            occupation_uri=occ.uri,
            user_skill_uris=user_uris,
            top_k_candidates=self.top_k_candidates,
            similarity_threshold=self.similarity_threshold,
            candidate_pool=self.candidate_pool,
        )
        self.last_gap_report = report

        print(subbanner("SUMMARY METRICS & MATCH PERCENTAGES"))
        print(f"Total Requirements:         {Term.BOLD}{report.total_required_skills}{Term.RESET}")
        print(f"  - Essential Requirements: {len(report.matched_essential) + len(report.missing_essential)}")
        print(f"  - Optional Requirements:  {len(report.matched_optional) + len(report.missing_optional)}")
        print()
        print(f"Verified Essential Match:   {Term.BOLD}{Term.GREEN}{report.essential_match_percentage:.1f}%{Term.RESET} "
              f"({len(report.matched_essential)} of {len(report.matched_essential) + len(report.missing_essential)} satisfied)")
        print(f"Verified Overall Match:     {Term.BOLD}{Term.CYAN}{report.overall_match_percentage:.1f}%{Term.RESET} "
              f"({len(report.matched_essential)} + {len(report.matched_optional)} of {report.total_required_skills} satisfied)")
        print(f"{Term.DIM}(Note: Percentages are strictly calculated from verified exact URI matches.){Term.RESET}\n")

        print(f"{Term.BOLD}{Term.GREEN}[✓] MATCHED ESSENTIAL SKILLS ({len(report.matched_essential)}):{Term.RESET}")
        if report.matched_essential:
            for s in report.matched_essential:
                print(f"  + {Term.BOLD}{s.skill_label}{Term.RESET} (Source: {s.match_source}, Score: {s.similarity_score:.1f})")
                print(f"    URI: {Term.DIM}{s.skill_uri}{Term.RESET}")
        else:
            print("  (None satisfied yet)")

        print(f"\n{Term.BOLD}{Term.CYAN}[✓] MATCHED OPTIONAL SKILLS ({len(report.matched_optional)}):{Term.RESET}")
        if report.matched_optional:
            for s in report.matched_optional:
                print(f"  + {Term.BOLD}{s.skill_label}{Term.RESET} (Source: {s.match_source}, Score: {s.similarity_score:.1f})")
                print(f"    URI: {Term.DIM}{s.skill_uri}{Term.RESET}")
        else:
            print("  (None satisfied yet)")

        print(f"\n{Term.BOLD}{Term.RED}[!] MISSING ESSENTIAL SKILLS ({len(report.missing_essential)}):{Term.RESET}")
        if report.missing_essential:
            for idx, s in enumerate(report.missing_essential[:15], start=1):
                type_str = f" [{s.skill_type}]" if s.skill_type else ""
                print(f"  {idx:2d}. {s.skill_label}{type_str}")
            if len(report.missing_essential) > 15:
                print(f"  ... and {len(report.missing_essential) - 15} more essential skill(s).")
        else:
            print("  (All essential skills satisfied!)")

        print(f"\n{Term.BOLD}{Term.YELLOW}[!] MISSING OPTIONAL SKILLS ({len(report.missing_optional)}):{Term.RESET}")
        if report.missing_optional:
            for idx, s in enumerate(report.missing_optional[:10], start=1):
                type_str = f" [{s.skill_type}]" if s.skill_type else ""
                print(f"  {idx:2d}. {s.skill_label}{type_str}")
            if len(report.missing_optional) > 10:
                print(f"  ... and {len(report.missing_optional) - 10} more optional skill(s).")
        else:
            print("  (All optional skills satisfied!)")

        self.display_semantic_candidates_section(report)
        return report

    def display_semantic_candidates_section(self, report: SkillGapReport) -> None:
        """Displays semantic candidate suggestions strictly separated from verified matches."""
        print(subbanner("SEMANTIC CANDIDATE SUGGESTIONS FOR REVIEW"))

        print(f"{Term.BOLD}{Term.MAGENTA}[CRITICAL PEDAGOGICAL / COMPETENCE NOTICE]{Term.RESET}")
        print("  * Semantic similarity is computed via offline precomputed embeddings.")
        print("  * Proximity indicates TOPICAL OR CONCEPTUAL RELATEDNESS, NOT proficiency.")
        print("  * Semantic candidates do NOT automatically certify competence, do NOT imply")
        print("    prerequisites, and do NOT remove required skills from the verified gap.")
        print("  * Verified match percentages above remain strictly intact.\n")

        if not report.semantic_candidates:
            if not self.user_skills and self.candidate_pool == "user_skills":
                print("  (No semantic candidates found: user skill profile is empty.)")
            else:
                thresh_str = f" >= {self.similarity_threshold}" if self.similarity_threshold else ""
                print(f"  (No semantic candidates met the similarity threshold{thresh_str}.)")
            return

        print(f"Surfaced {len(report.semantic_candidates)} candidate relation(s) for curriculum review:")
        by_req = report.candidates_by_required_skill
        count = 0
        for req_uri, candidates in by_req.items():
            first = candidates[0]
            rel_label = f"[{first.relation_type.upper()}]"
            print(f"\n  Required {rel_label}: {Term.BOLD}{first.required_skill_label}{Term.RESET}")
            print(f"  Status: {Term.RED}MISSING FROM VERIFIED GAP{Term.RESET}")
            for c in candidates:
                score_bar = "■" * int(c.similarity_score * 10)
                print(f"    ↳ Suggested Candidate: {Term.CYAN}{c.candidate_skill_label}{Term.RESET}")
                print(f"      Similarity Score:   {Term.BOLD}{c.similarity_score:.4f}{Term.RESET} |{score_bar}| ({c.matching_method})")
                print(f"      Candidate URI:      {Term.DIM}{c.candidate_skill_uri}{Term.RESET}")
            count += 1
            if count >= 8 and len(by_req) > 8:
                print(f"\n  ... and {len(by_req) - 8} more requirement(s) with semantic candidates.")
                break
        print()

    def generate_learning_path(self) -> LearningPath | None:
        """Generates structured learning curriculum sequence for missing skills."""
        if not self.last_gap_report:
            print(f"\n{Term.YELLOW}[!] No recent skill gap report. Running gap analysis first...{Term.RESET}")
            rep = self.run_skill_gap_analysis()
            if not rep:
                return None
        else:
            rep = self.last_gap_report

        missing_skills = [
            {"skillUri": m.skill_uri, "skillLabel": m.skill_label, "relationType": m.relation_type}
            for m in rep.missing_essential + rep.missing_optional
        ]

        if not missing_skills:
            print(banner("LEARNING PATH GENERATION"))
            print(f"{Term.GREEN}[✓] 100% Match Achieved! No missing skills to sequence.{Term.RESET}")
            return None

        dag = PrerequisiteGraph()
        prereq_configured = False

        if self.use_sample_dag:
            sample_edges = [
                ("http://data.europa.eu/esco/skill/f28617ad-afdd-4041-814c-216153a38998", "http://data.europa.eu/esco/skill/d9e5349e-8791-49c2-8ba4-839fdd1606c2"),
                ("http://data.europa.eu/esco/skill/d9e5349e-8791-49c2-8ba4-839fdd1606c2", "http://data.europa.eu/esco/skill/797a4416-0b8d-44f3-8fcd-eb10bd0be6b4"),
                ("http://data.europa.eu/esco/skill/797a4416-0b8d-44f3-8fcd-eb10bd0be6b4", "http://data.europa.eu/esco/skill/2522a6ce-3202-4ac8-9f5b-b9cb5a3a83a1"),
            ]
            for p_uri, t_uri in sample_edges:
                dag.add_prerequisite(p_uri, t_uri)
            prereq_configured = True
            generator = LearningPathGenerator(prerequisite_graph=dag)
        else:
            generator = self.path_generator

        path = generator.generate_path(
            target_occupation_uri=rep.occupation_uri,
            target_occupation_label=rep.occupation_label,
            missing_skills=missing_skills,
        )
        self.last_learning_path = path

        print(banner("PERSONALIZED LEARNING PATH CURRICULUM"))
        print(f"Target Occupation: {Term.BOLD}{path.target_occupation_label}{Term.RESET}")
        print(f"Total Sequenced Steps: {Term.BOLD}{path.total_steps}{Term.RESET}")
        print(f"Prerequisite Cycle Detected: {Term.RED if path.has_prerequisite_cycles else Term.GREEN}{path.has_prerequisite_cycles}{Term.RESET}")

        print(subbanner("PREREQUISITE RELATIONSHIP & ORDERING DISCLOSURE"))
        if not prereq_configured:
            print(f"{Term.BOLD}{Term.YELLOW}[!] IMPORTANT TAXONOMY vs PREREQUISITE DISCLOSURE:{Term.RESET}")
            print("    * No external pedagogical prerequisite relationships were configured for this run.")
            print("    * In ESCO, 'skillsHierarchy' is an occupational taxonomy (Levels 0-3), and")
            print("      'skillSkillRelations' represents associative co-occurrence.")
            print("    * Neither dataset represents verified educational learning prerequisites.")
            print("    * Consequently, the curriculum sequence below is a DETERMINISTIC FALLBACK ORDERING:")
            print("      1. Foundation / Essential competencies are scheduled first (sorted alphabetically).")
            print("      2. Specialization / Optional competencies follow second (sorted alphabetically).")
        else:
            print(f"{Term.BOLD}{Term.GREEN}[✓] Verified Pedagogical Prerequisite DAG Active:{Term.RESET}")
            print("    * Explicit curriculum dependencies applied.")
            print("    * Sequenced via Kahn's topological sort algorithm.")
        print()

        print(f"{Term.BOLD}Curriculum Roadmap ({path.total_steps} Steps):{Term.RESET}")
        essential_steps = [s for s in path.steps if s.relation_type == "essential"]
        optional_steps = [s for s in path.steps if s.relation_type == "optional"]

        print(f"\n-- Phase 1: Essential Competencies ({len(essential_steps)} steps) --")
        for s in essential_steps[:15]:
            prereq_str = f" [Prereqs: {len(s.prerequisites)} dependency]" if s.prerequisites else ""
            print(f"  Step {s.step_number:2d}: {Term.BOLD}{s.skill_label}{Term.RESET}{prereq_str}")
            print(f"          URI: {Term.DIM}{s.skill_uri}{Term.RESET}")
        if len(essential_steps) > 15:
            print(f"  ... and {len(essential_steps) - 15} additional essential steps.")

        if optional_steps:
            print(f"\n-- Phase 2: Optional & Specialization Competencies ({len(optional_steps)} steps) --")
            for s in optional_steps[:10]:
                prereq_str = f" [Prereqs: {len(s.prerequisites)} dependency]" if s.prerequisites else ""
                print(f"  Step {s.step_number:2d}: {s.skill_label}{prereq_str}")
            if len(optional_steps) > 10:
                print(f"  ... and {len(optional_steps) - 10} additional optional steps.")

        print()
        return path

    def configure_settings_interactive(self) -> None:
        """Configures demo parameters like thresholds, candidate pools, and sample DAG."""
        while True:
            print(subbanner("DEMO SETTINGS & PARAMETERS"))
            print(f"  [1] Similarity Cutoff Threshold: {Term.BOLD}{self.similarity_threshold or 'None (disabled)'}{Term.RESET}")
            print(f"  [2] Top-K Semantic Candidates:   {Term.BOLD}{self.top_k_candidates}{Term.RESET}")
            print(f"  [3] Semantic Candidate Pool:     {Term.BOLD}{self.candidate_pool}{Term.RESET}")
            print(f"  [4] Toggle Sample Prereq DAG:    {Term.BOLD}{'ENABLED' if self.use_sample_dag else 'DISABLED'}{Term.RESET}")
            print("  [0] Return to main menu")

            choice = input(f"\n{Term.BOLD}Enter choice (0-4): {Term.RESET}").strip()
            if choice == "1":
                raw = input("Enter new similarity threshold (0.0 to 1.0, or 'none' to disable): ").strip()
                if raw.lower() == "none":
                    self.similarity_threshold = None
                    print(f"{Term.GREEN}[✓] Threshold disabled.{Term.RESET}")
                else:
                    try:
                        val = float(raw)
                        if 0.0 <= val <= 1.0:
                            self.similarity_threshold = val
                            print(f"{Term.GREEN}[✓] Threshold updated to {val:.2f}{Term.RESET}")
                        else:
                            print(f"{Term.RED}Threshold must be between 0.0 and 1.0{Term.RESET}")
                    except ValueError:
                        print(f"{Term.RED}Invalid float.{Term.RESET}")
            elif choice == "2":
                raw = input("Enter top-k candidates per skill (integer >= 1): ").strip()
                if raw.isdigit() and int(raw) >= 1:
                    self.top_k_candidates = int(raw)
                    print(f"{Term.GREEN}[✓] Top-K updated to {self.top_k_candidates}{Term.RESET}")
                else:
                    print(f"{Term.RED}Invalid positive integer.{Term.RESET}")
            elif choice == "3":
                print("1. 'user_skills': Compare user's possessed skills to missing skills.")
                print("2. 'catalog':     Recommend related skills from the entire 13,939 ESCO catalog.")
                sub_c = input("Select pool (1 or 2): ").strip()
                if sub_c == "1":
                    self.candidate_pool = "user_skills"
                elif sub_c == "2":
                    self.candidate_pool = "catalog"
                print(f"{Term.GREEN}[✓] Candidate pool set to '{self.candidate_pool}'{Term.RESET}")
            elif choice == "4":
                self.use_sample_dag = not self.use_sample_dag
                status = "ENABLED (Sample DAG)" if self.use_sample_dag else "DISABLED (Standard Fallback)"
                print(f"{Term.GREEN}[✓] Prerequisite DAG mode: {status}{Term.RESET}")
            elif choice == "0":
                break
            else:
                print(f"{Term.RED}Invalid option.{Term.RESET}")

    def run_full_workflow(self) -> None:
        """Runs the entire pipeline end-to-end in a guided manner."""
        print(banner("FULL GUIDED WORKFLOW"))
        if not self.current_occupation:
            self.select_occupation_interactive()
            if not self.current_occupation:
                print(f"{Term.YELLOW}Workflow cancelled.{Term.RESET}")
                return

        if not self.user_skills:
            print("\nYour skill profile is currently empty.")
            print("Would you like to load a sample profile or enter skills?")
            print("  [1] Load 'Phase 3 Verified Synthetic Learner' sample profile")
            print("  [2] Load 'Junior Python Developer' sample profile")
            print("  [3] Enter skills manually")
            print("  [4] Continue with empty profile (test 0% baseline)")
            c = input("Choice (1-4, default=1): ").strip() or "1"
            if c == "1":
                self.process_raw_skills(SAMPLE_PROFILES["synthetic"]["skills"])
            elif c == "2":
                self.process_raw_skills(SAMPLE_PROFILES["junior_dev"]["skills"])
            elif c == "3":
                self.add_skills_bulk_interactive()
            elif c == "4":
                pass

        self.run_skill_gap_analysis()
        self.generate_learning_path()

    def run_interactive_loop(self) -> None:
        """Main terminal event loop."""
        while True:
            occ_label = self.current_occupation.label if self.current_occupation else "None (Not Selected)"
            skills_count = len(self.user_skills)

            print(banner("PATHGRAPH AI - MAIN CONTROL PANEL"))
            print(f"Target Occupation: {Term.BOLD}{occ_label}{Term.RESET}")
            print(f"Active Skills:     {Term.BOLD}{skills_count}{Term.RESET} recognized ESCO skill(s)")
            print(f"Prereq Mode:       {Term.BOLD}{'Sample DAG Active' if self.use_sample_dag else 'Standard Fallback'}{Term.RESET}")
            print()
            print("Actions:")
            print("  [1] Select Target Occupation (Search ESCO catalog)")
            print("  [2] Enter & Manage Skills (Input names, view matches)")
            print("  [3] Run Skill Gap Analysis (Exact & Semantic Separation)")
            print("  [4] Generate Learning Path (Sequenced Curriculum)")
            print("  [5] Run Full Workflow (Guided Steps 1 -> 4)")
            print("  [6] Demo Settings & Parameters")
            print("  [0] Exit Demo")

            try:
                choice = input(f"\n{Term.BOLD}Select an action (0-6): {Term.RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{Term.CYAN}Exiting PathGraph AI demo. Goodbye!{Term.RESET}\n")
                break

            if choice == "1":
                self.select_occupation_interactive()
            elif choice == "2":
                self.manage_skills_interactive()
            elif choice == "3":
                self.run_skill_gap_analysis()
            elif choice == "4":
                self.generate_learning_path()
            elif choice == "5":
                self.run_full_workflow()
            elif choice == "6":
                self.configure_settings_interactive()
            elif choice == "0":
                print(f"\n{Term.CYAN}Exiting PathGraph AI demo. Goodbye!{Term.RESET}\n")
                break
            else:
                print(f"{Term.RED}Invalid selection. Please choose an option from 0 to 6.{Term.RESET}")


def parse_cli_args() -> argparse.ArgumentParser:
    """Builds CLI argument parser for non-interactive scripting or automated verification."""
    parser = argparse.ArgumentParser(
        description="PathGraph AI - Local Interactive Terminal Demo & Verification CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--occupation",
        type=str,
        default=None,
        help="Pre-select an occupation by preferred label, keyword, or ESCO URI.",
    )
    parser.add_argument(
        "--skills",
        type=str,
        default=None,
        help="Comma-separated list of skill names or URIs to resolve.",
    )
    parser.add_argument(
        "--sample-profile",
        choices=["synthetic", "junior_dev", "full_match", "empty"],
        default=None,
        help="Load a pre-configured sample profile.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Similarity threshold for semantic candidate review.",
    )
    parser.add_argument(
        "--candidate-pool",
        choices=["user_skills", "catalog"],
        default="user_skills",
        help="Candidate pool for semantic matching.",
    )
    parser.add_argument(
        "--use-sample-dag",
        action="store_true",
        help="Enable sample pedagogical prerequisite DAG.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Execute workflow headlessly with provided flags and exit immediately.",
    )
    return parser


def main() -> int:
    """Main CLI and interactive entrypoint."""
    parser = parse_cli_args()
    args = parser.parse_args()

    demo = PathGraphTerminalDemo()
    if not demo.initialize():
        return 1

    demo.similarity_threshold = args.threshold
    demo.candidate_pool = args.candidate_pool
    demo.use_sample_dag = args.use_sample_dag

    if args.occupation:
        matches = demo.search_occupations(args.occupation, limit=1)
        if matches:
            demo.set_target_occupation(matches[0])
        else:
            print(f"{Term.YELLOW}Warning: No occupation found matching '{args.occupation}'{Term.RESET}")

    if args.sample_profile:
        prof = SAMPLE_PROFILES[args.sample_profile]
        print(f"Loading CLI sample profile: {prof['name']}")
        demo.process_raw_skills(prof["skills"])
    elif args.skills:
        tokens = [s.strip() for s in args.skills.split(",") if s.strip()]
        demo.process_raw_skills(tokens)

    if args.non_interactive or (args.occupation and (args.skills or args.sample_profile)):
        print("\n>>> Running automated end-to-end demo execution...\n")
        demo.run_skill_gap_analysis()
        demo.generate_learning_path()
        print(f"\n{Term.GREEN}[✓] Automated demo execution completed successfully.{Term.RESET}\n")
        return 0

    demo.run_interactive_loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
