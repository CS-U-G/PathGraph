"""Benchmark Personas for Computing Career-Transition Quality Evaluation.

Defines 5 realistic computing personas:
1. Junior Frontend Web Developer -> Full-Stack Software Developer
2. Manual QA Tester -> Software Development Engineer in Test (SDET)
3. Junior Data Analyst -> Machine Learning Engineer / Data Scientist
4. Systems Administrator -> Cloud DevOps Engineer
5. Helpdesk Support Technician -> Junior Cybersecurity Technician

Each persona provides:
- Current role and target role
- Current skills with prior proficiencies
- Verified target essential skills
- Expected missing essential skills (ground-truth gap)
- Target optional specialization skills
- Explicit instructional prerequisites (pedagogical dependencies)
- Known irrelevant skills (negative controls for recommendation leakage)
- Curricular provenance and documented limitations
"""

from __future__ import annotations

from pathgraph_ai.benchmarks.schemas import BenchmarkPersona, BenchmarkSkill


PERSONA_FRONTEND_TO_FULLSTACK = BenchmarkPersona(
    persona_id="persona_frontend_to_fullstack",
    title="Junior Frontend Developer to Full-Stack Software Developer",
    current_role="Junior Frontend Web Developer",
    target_role="Full-Stack Software Developer",
    target_occupation_uri="http://data.europa.eu/esco/occupation/f2b15a0e-e65a-438a-affb-29b9d50b77d1",
    target_occupation_label="software developer",
    current_skills=[
        BenchmarkSkill(
            skill_uri="http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97",
            label="computer programming",
            proficiency=0.85,
        ),
        BenchmarkSkill(
            skill_uri="http://data.europa.eu/esco/skill/3cd569a2-4f88-4c1e-9995-8dce8c5e51a7",
            label="JavaScript",
            proficiency=0.90,
        ),
        BenchmarkSkill(
            skill_uri="http://data.europa.eu/esco/skill/925463a7-d51f-4d5b-9f79-4d28cf30acde",
            label="integrated development environment software",
            proficiency=0.80,
        ),
    ],
    expected_missing_essential=[
        "http://data.europa.eu/esco/skill/209a5498-3449-4689-8ed9-bd08cab4fd78",  # engineering principles
        "http://data.europa.eu/esco/skill/6014921e-1d25-4039-adc2-04852d61880e",  # ICT debugging tools
        "http://data.europa.eu/esco/skill/7111b95d-0ce3-441a-9d92-4c75d05c4388",  # project management
        "http://data.europa.eu/esco/skill/72a74f69-5cf1-43c5-99b9-62a444578919",  # engineering processes
        "http://data.europa.eu/esco/skill/f84a433f-34f1-4083-b0a3-24802623509c",  # web services
        "http://data.europa.eu/esco/skill/2522a6ce-3202-4ac8-9f5b-b9cb5a3a83a1",  # debug software
        "http://data.europa.eu/esco/skill/286f8742-16a7-47b8-8097-f584e03b4194",  # use software design patterns
    ],
    expected_optional_skills=[
        "http://data.europa.eu/esco/skill/19a8293b-8e95-4de3-983f-77484079c389",  # Java
        "http://data.europa.eu/esco/skill/8fca240e-fb91-443e-a14a-1df812ca447b",  # Apache Maven
        "http://data.europa.eu/esco/skill/44f80879-1ea2-47e0-94f3-b3c990b7a66b",  # Python
    ],
    expert_relevant_target_skills=[
        "http://data.europa.eu/esco/skill/209a5498-3449-4689-8ed9-bd08cab4fd78",
        "http://data.europa.eu/esco/skill/6014921e-1d25-4039-adc2-04852d61880e",
        "http://data.europa.eu/esco/skill/7111b95d-0ce3-441a-9d92-4c75d05c4388",
        "http://data.europa.eu/esco/skill/72a74f69-5cf1-43c5-99b9-62a444578919",
        "http://data.europa.eu/esco/skill/f84a433f-34f1-4083-b0a3-24802623509c",
        "http://data.europa.eu/esco/skill/2522a6ce-3202-4ac8-9f5b-b9cb5a3a83a1",
        "http://data.europa.eu/esco/skill/286f8742-16a7-47b8-8097-f584e03b4194",
        "http://data.europa.eu/esco/skill/19a8293b-8e95-4de3-983f-77484079c389",
        "http://data.europa.eu/esco/skill/8fca240e-fb91-443e-a14a-1df812ca447b",
        "http://data.europa.eu/esco/skill/44f80879-1ea2-47e0-94f3-b3c990b7a66b",
    ],
    known_irrelevant_skills=[
        "http://data.europa.eu/esco/skill/00dd3271-077f-4a12-a958-e297fdd724ce",  # operate agricultural machinery
        "http://data.europa.eu/esco/skill/c4e7ec9e-f0b4-4b55-a0bc-4342207b539c",  # nursing care
        "http://data.europa.eu/esco/skill/50346fb6-cb65-4f7f-856f-2396e95c1064",  # livestock reproduction techniques
    ],
    explicit_prerequisites=[
        (
            "http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97",  # computer programming
            "http://data.europa.eu/esco/skill/2522a6ce-3202-4ac8-9f5b-b9cb5a3a83a1",  # debug software
        ),
        (
            "http://data.europa.eu/esco/skill/209a5498-3449-4689-8ed9-bd08cab4fd78",  # engineering principles
            "http://data.europa.eu/esco/skill/72a74f69-5cf1-43c5-99b9-62a444578919",  # engineering processes
        ),
    ],
    provenance_and_limitations=(
        "Curated from ACM/IEEE-CS Software Engineering Body of Knowledge (SWEBOK v3.0) and "
        "ESCO v1.2.1 software developer requirements. Demonstrates typical web frontend to "
        "backend/architecture expansion. Limitations: Assumes web frontend foundations do not "
        "subsume distributed system concurrency or systems architecture."
    ),
)


PERSONA_MANUAL_QA_TO_SDET = BenchmarkPersona(
    persona_id="persona_manual_qa_to_sdet",
    title="Manual QA Tester to Software Test Automation Engineer",
    current_role="Manual QA Tester",
    target_role="Software Development Engineer in Test (SDET)",
    target_occupation_uri="http://data.europa.eu/esco/occupation/106f79e4-6264-45f1-9e7a-297435cd684b",
    target_occupation_label="software tester",
    current_skills=[
        BenchmarkSkill(
            skill_uri="http://data.europa.eu/esco/skill/913e7e83-b8f8-4574-b1ca-1b38f3fd974a",
            label="execute software tests",
            proficiency=0.90,
        ),
        BenchmarkSkill(
            skill_uri="http://data.europa.eu/esco/skill/561c2525-5bd8-4d86-8f45-820f4be7db93",
            label="report test findings",
            proficiency=0.85,
        ),
    ],
    expected_missing_essential=[
        "http://data.europa.eu/esco/skill/2450c3b3-e78e-435b-b84d-e05d984e71dc",  # software architecture models
        "http://data.europa.eu/esco/skill/438ff259-d086-4004-acda-1ec3e1631bc6",  # software anomalies
        "http://data.europa.eu/esco/skill/85f46538-ae70-498a-bfbc-b8ddafe96c7d",  # levels of software testing
        "http://data.europa.eu/esco/skill/fb7aee4d-f6c6-4a87-8e9d-8e542b3771f8",  # software metrics
        "http://data.europa.eu/esco/skill/78a74fd1-efc6-49dc-adb1-1b0fbe35c22c",  # provide software testing documentation
    ],
    expected_optional_skills=[
        "http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97",  # computer programming
        "http://data.europa.eu/esco/skill/0b0335f3-0aa1-491e-895e-81fc8774a300",  # develop automated migration methods
    ],
    expert_relevant_target_skills=[
        "http://data.europa.eu/esco/skill/2450c3b3-e78e-435b-b84d-e05d984e71dc",
        "http://data.europa.eu/esco/skill/438ff259-d086-4004-acda-1ec3e1631bc6",
        "http://data.europa.eu/esco/skill/85f46538-ae70-498a-bfbc-b8ddafe96c7d",
        "http://data.europa.eu/esco/skill/fb7aee4d-f6c6-4a87-8e9d-8e542b3771f8",
        "http://data.europa.eu/esco/skill/78a74fd1-efc6-49dc-adb1-1b0fbe35c22c",
        "http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97",
    ],
    known_irrelevant_skills=[
        "http://data.europa.eu/esco/skill/8b211116-43b9-445a-939e-d52f97463f2d",  # bricklaying
        "http://data.europa.eu/esco/skill/092cfbe1-0e1f-4966-9e12-424a66a12b4e",  # veterinary surgery
        "http://data.europa.eu/esco/skill/20268593-6b32-4416-8367-759084c7ef3b",  # pastry baking
    ],
    explicit_prerequisites=[
        (
            "http://data.europa.eu/esco/skill/85f46538-ae70-498a-bfbc-b8ddafe96c7d",  # levels of software testing
            "http://data.europa.eu/esco/skill/fb7aee4d-f6c6-4a87-8e9d-8e542b3771f8",  # software metrics
        ),
        (
            "http://data.europa.eu/esco/skill/438ff259-d086-4004-acda-1ec3e1631bc6",  # software anomalies
            "http://data.europa.eu/esco/skill/78a74fd1-efc6-49dc-adb1-1b0fbe35c22c",  # provide software testing documentation
        ),
    ],
    provenance_and_limitations=(
        "Curated based on ISTQB Foundation & Advanced Test Automation curricula mapped to ESCO "
        "software tester occupation. Represents moving from black-box manual execution to formal "
        "test architecture and automation. Limitations: Does not evaluate programming language specificity."
    ),
)


PERSONA_DATA_ANALYST_TO_ML = BenchmarkPersona(
    persona_id="persona_data_analyst_to_ml",
    title="Data Analyst to Machine Learning Specialist / Data Scientist",
    current_role="Junior Data Analyst",
    target_role="Data Scientist / Machine Learning Engineer",
    target_occupation_uri="http://data.europa.eu/esco/occupation/258e46f9-0075-4a2e-adae-1ff0477e0f30",
    target_occupation_label="data scientist",
    current_skills=[
        BenchmarkSkill(
            skill_uri="http://data.europa.eu/esco/skill/348b74cd-49ce-4844-8bdf-ec188b497213",
            label="visual presentation techniques",
            proficiency=0.85,
        ),
        BenchmarkSkill(
            skill_uri="http://data.europa.eu/esco/skill/6360a934-cc87-4954-9656-b32db20592e3",
            label="quantitative analysis",
            proficiency=0.80,
        ),
        BenchmarkSkill(
            skill_uri="http://data.europa.eu/esco/skill/65e58886-bd1e-4c5b-8ca5-8d9b353c8aa1",
            label="data visualisation software",
            proficiency=0.90,
        ),
    ],
    expected_missing_essential=[
        "http://data.europa.eu/esco/skill/0760bfac-9a03-427a-ad84-ccd33a2f2ae8",  # mathematical modelling
        "http://data.europa.eu/esco/skill/25f0ea33-b4a2-4f31-b7b4-7d20e827b180",  # data mining
        "http://data.europa.eu/esco/skill/48db96bf-3314-45c6-bad8-fdb6e20e5639",  # data engineering
    ],
    expected_optional_skills=[
        "http://data.europa.eu/esco/skill/3a2d5b45-56e4-4f5a-a55a-4a4a65afdc43",  # machine learning
        "http://data.europa.eu/esco/skill/44f80879-1ea2-47e0-94f3-b3c990b7a66b",  # Python
    ],
    expert_relevant_target_skills=[
        "http://data.europa.eu/esco/skill/0760bfac-9a03-427a-ad84-ccd33a2f2ae8",
        "http://data.europa.eu/esco/skill/25f0ea33-b4a2-4f31-b7b4-7d20e827b180",
        "http://data.europa.eu/esco/skill/48db96bf-3314-45c6-bad8-fdb6e20e5639",
        "http://data.europa.eu/esco/skill/3a2d5b45-56e4-4f5a-a55a-4a4a65afdc43",
    ],
    known_irrelevant_skills=[
        "http://data.europa.eu/esco/skill/2ba203dc-5645-420a-8bf8-028448ec6d5a",  # culinary menu planning
        "http://data.europa.eu/esco/skill/372551ec-0520-410a-b286-9a295c5553e1",  # dental hygiene
        "http://data.europa.eu/esco/skill/5176a9ff-521b-4395-950c-0728c773950b",  # automotive welding
    ],
    explicit_prerequisites=[
        (
            "http://data.europa.eu/esco/skill/6360a934-cc87-4954-9656-b32db20592e3",  # quantitative analysis
            "http://data.europa.eu/esco/skill/0760bfac-9a03-427a-ad84-ccd33a2f2ae8",  # mathematical modelling
        ),
    ],
    provenance_and_limitations=(
        "Derived from standard Data Science industry career pathways (EDISON Data Science "
        "Framework). Targets transition from descriptive dashboard analytics to predictive and "
        "inferential modeling. Limitations: ESCO data scientist occupation has 63 essential skills, "
        "reflecting an encyclopedic breadth that over-represents generic research tasks."
    ),
)


PERSONA_SYSADMIN_TO_DEVOPS = BenchmarkPersona(
    persona_id="persona_sysadmin_to_devops",
    title="Systems Administrator to Cloud DevOps Engineer",
    current_role="ICT System Administrator",
    target_role="Cloud DevOps Engineer",
    target_occupation_uri="http://data.europa.eu/esco/occupation/cc867bee-ab5c-427f-9244-f7a204d9574b",
    target_occupation_label="cloud DevOps engineer",
    current_skills=[
        BenchmarkSkill(
            skill_uri="http://data.europa.eu/esco/skill/47ca0da1-cae5-4395-ae6b-fd97b9ff48d3",
            label="system backup best practice",
            proficiency=0.90,
        ),
        BenchmarkSkill(
            skill_uri="http://data.europa.eu/esco/skill/89f6560b-2194-45c9-9ece-d33049a73eef",
            label="computer engineering",
            proficiency=0.80,
        ),
    ],
    expected_missing_essential=[
        "http://data.europa.eu/esco/skill/1c2978b8-bb0d-4249-9c23-877571a4dffa",  # cloud security and compliance
        "http://data.europa.eu/esco/skill/24c200e5-be20-4370-a137-ab53797f3a17",  # software frameworks
        "http://data.europa.eu/esco/skill/8efa6a7a-6556-4cb8-908d-59d3b5c58d2f",  # cloud monitoring and reporting
    ],
    expected_optional_skills=[
        "http://data.europa.eu/esco/skill/3cd569a2-4f88-4c1e-9995-8dce8c5e51a7",  # JavaScript
    ],
    expert_relevant_target_skills=[
        "http://data.europa.eu/esco/skill/1c2978b8-bb0d-4249-9c23-877571a4dffa",
        "http://data.europa.eu/esco/skill/24c200e5-be20-4370-a137-ab53797f3a17",
        "http://data.europa.eu/esco/skill/8efa6a7a-6556-4cb8-908d-59d3b5c58d2f",
    ],
    known_irrelevant_skills=[
        "http://data.europa.eu/esco/skill/46128711-2e21-4f76-9c71-332e92c2122d",  # wildlife conservation
        "http://data.europa.eu/esco/skill/fbb94315-7762-4217-91a5-829d665a3d02",  # heavy vehicle operation
        "http://data.europa.eu/esco/skill/981a53b2-6564-42b7-a37a-973059695d73",  # commercial fishing techniques
    ],
    explicit_prerequisites=[
        (
            "http://data.europa.eu/esco/skill/89f6560b-2194-45c9-9ece-d33049a73eef",  # computer engineering
            "http://data.europa.eu/esco/skill/8efa6a7a-6556-4cb8-908d-59d3b5c58d2f",  # cloud monitoring and reporting
        ),
    ],
    provenance_and_limitations=(
        "Curated from Cloud Native Computing Foundation (CNCF) certifications (CKA/CKAD) and "
        "ESCO cloud DevOps engineer requirements. Focuses on containerized infrastructure, "
        "cloud compliance, and automated telemetry. Limitations: ESCO lacks granular modern tool "
        "concepts for Terraform or Kubernetes."
    ),
)


PERSONA_HELPDESK_TO_SECURITY = BenchmarkPersona(
    persona_id="persona_helpdesk_to_security",
    title="IT Support Specialist to Cybersecurity Technician",
    current_role="IT Helpdesk Specialist",
    target_role="Junior Cybersecurity Technician",
    target_occupation_uri="http://data.europa.eu/esco/occupation/a44a1dc5-be08-4840-8bd5-770c4ac1ca6d",
    target_occupation_label="ICT security technician",
    current_skills=[
        BenchmarkSkill(
            skill_uri="http://data.europa.eu/esco/skill/32428b21-514b-42c2-81b5-244a7804ea74",
            label="procurement of ICT network equipment",
            proficiency=0.75,
        ),
        BenchmarkSkill(
            skill_uri="http://data.europa.eu/esco/skill/6a536b36-c5c7-4d10-bf67-a4fb0f8549b2",
            label="audit techniques",
            proficiency=0.70,
        ),
    ],
    expected_missing_essential=[
        "http://data.europa.eu/esco/skill/09f2f811-a3fb-4de3-a70f-6420a6935575",  # systems development life-cycle
        "http://data.europa.eu/esco/skill/3ff589b7-68df-4ea5-ae41-b395bdb2378f",  # ICT security standards
        "http://data.europa.eu/esco/skill/501537a7-9d63-491d-ae02-09f04767ff1e",  # attack vectors
        "http://data.europa.eu/esco/skill/5898d99a-62a4-4e10-a2e3-0d815ce44248",  # cyber attack counter-measures
    ],
    expected_optional_skills=[
        "http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97",  # computer programming
    ],
    expert_relevant_target_skills=[
        "http://data.europa.eu/esco/skill/09f2f811-a3fb-4de3-a70f-6420a6935575",
        "http://data.europa.eu/esco/skill/3ff589b7-68df-4ea5-ae41-b395bdb2378f",
        "http://data.europa.eu/esco/skill/501537a7-9d63-491d-ae02-09f04767ff1e",
        "http://data.europa.eu/esco/skill/5898d99a-62a4-4e10-a2e3-0d815ce44248",
    ],
    known_irrelevant_skills=[
        "http://data.europa.eu/esco/skill/6c30f4a7-8979-450a-9d2c-806859546b2b",  # floral arrangement
        "http://data.europa.eu/esco/skill/731057ec-0248-4395-9204-685b8827fa65",  # childhood education pedagogy
        "http://data.europa.eu/esco/skill/81a2939b-7347-495b-b91c-3088b6348873",  # tractor maintenance
    ],
    explicit_prerequisites=[
        (
            "http://data.europa.eu/esco/skill/3ff589b7-68df-4ea5-ae41-b395bdb2378f",  # ICT security standards
            "http://data.europa.eu/esco/skill/501537a7-9d63-491d-ae02-09f04767ff1e",  # attack vectors
        ),
        (
            "http://data.europa.eu/esco/skill/501537a7-9d63-491d-ae02-09f04767ff1e",  # attack vectors
            "http://data.europa.eu/esco/skill/5898d99a-62a4-4e10-a2e3-0d815ce44248",  # cyber attack counter-measures
        ),
    ],
    provenance_and_limitations=(
        "Aligned with CompTIA Security+ and NIST NICE Framework (Operate and Maintain to Protect "
        "and Defend). Validates acquisition of core offensive vector understanding prior to defensive "
        "hardening. Limitations: Does not test real-time SIEM log analysis or forensic reverse-engineering."
    ),
)


ALL_BENCHMARK_PERSONAS: list[BenchmarkPersona] = [
    PERSONA_FRONTEND_TO_FULLSTACK,
    PERSONA_MANUAL_QA_TO_SDET,
    PERSONA_DATA_ANALYST_TO_ML,
    PERSONA_SYSADMIN_TO_DEVOPS,
    PERSONA_HELPDESK_TO_SECURITY,
]
