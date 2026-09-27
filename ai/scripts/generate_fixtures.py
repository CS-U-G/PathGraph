import json
from pathlib import Path

fixtures_dir = Path(__file__).resolve().parent.parent.parent / "contracts" / "fixtures"
fixtures_dir.mkdir(parents=True, exist_ok=True)

valid_request_full = {
    "learner": {
        "skills": [
            {
                "skill_id": "http://data.europa.eu/esco/skill/0007833a-6ff5-4089-a292-6284693b4dd6",
                "label": "Python (programming language)",
                "proficiency": 0.8
            },
            {
                "skill_id": None,
                "label": "fastapi web development",
                "proficiency": 0.4
            },
            {
                "label": "docker containerization"
            }
        ],
        "target": {
            "occupation_id": "http://data.europa.eu/esco/occupation/2512.1",
            "target_type": "occupation",
            "custom_goal": None
        }
    }
}

valid_request_minimal = {
    "learner": {
        "skills": [
            {
                "label": "Python programming"
            }
        ],
        "target": {
            "occupation_id": "http://data.europa.eu/esco/occupation/2512.1"
        }
    }
}

valid_response = {
    "learning_path": [
        {
            "skill_id": "http://data.europa.eu/esco/skill/0007833a-6ff5-4089-a292-6284693b4dd6",
            "label": "Python (programming language)",
            "priority": 1,
            "gap": 0.2,
            "prerequisite_satisfied": True,
            "relevance": 0.96,
            "demand": 0.91,
            "estimated_cost": 25.0,
            "reason": "Foundational requirement with 0.20 mastery gap; all prerequisite competencies met.",
            "reason_codes": [
                "REQUIRED_BY_TARGET",
                "MASTERY_GAP",
                "PREREQ_SATISFIED",
                "HIGH_RELEVANCE"
            ]
        },
        {
            "skill_id": "http://data.europa.eu/esco/skill/a2c67645-1234-4567-89ab-cdef01234567",
            "label": "RESTful web services",
            "priority": 2,
            "gap": 0.5,
            "prerequisite_satisfied": True,
            "relevance": 0.89,
            "demand": 0.87,
            "estimated_cost": 18.0,
            "reason": "Stepping stone architecture pattern required for target software developer role.",
            "reason_codes": [
                "REQUIRED_BY_TARGET",
                "STEPPING_STONE",
                "PREREQ_SATISFIED"
            ]
        }
    ],
    "meta": {
        "target_occupation_id": "http://data.europa.eu/esco/occupation/2512.1",
        "unresolved_skills": [
            "quantum cooking"
        ],
        "generated_at": "2026-09-27T08:30:00Z"
    }
}

error_response_400 = {
    "code": "INVALID_TARGET_OCCUPATION",
    "message": "Target occupation 'invalid_occ_9999' was not found in the ESCO taxonomy.",
    "details": None,
    "timestamp": "2026-09-27T08:30:00Z"
}

error_response_422 = {
    "code": "VALIDATION_ERROR",
    "message": "Validation failed for one or more request fields.",
    "details": [
        {
            "field": "learner.skills[0].proficiency",
            "issue": "Input should be less than or equal to 1.0"
        }
    ],
    "timestamp": "2026-09-27T08:30:00Z"
}

invalid_request_bad_proficiency = {
    "learner": {
        "skills": [
            {
                "label": "Python",
                "proficiency": 1.5
            }
        ],
        "target": {
            "occupation_id": "http://data.europa.eu/esco/occupation/2512.1"
        }
    }
}

invalid_request_missing_target = {
    "learner": {
        "skills": [
            {
                "label": "Python"
            }
        ]
    }
}

files = {
    "valid_request_full.json": valid_request_full,
    "valid_request_minimal.json": valid_request_minimal,
    "valid_response.json": valid_response,
    "error_response_400.json": error_response_400,
    "error_response_422.json": error_response_422,
    "invalid_request_bad_proficiency.json": invalid_request_bad_proficiency,
    "invalid_request_missing_target.json": invalid_request_missing_target,
}

for filename, data in files.items():
    p = fixtures_dir / filename
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote fixture: {p.name}")
