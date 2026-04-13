"""Tool schemas for career-ops job search skill."""

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "career_evaluate_offer",
            "description": (
                "Evaluate a job offer against the user's CV and profile. "
                "Generates a scored report (Blocks A-G) with match analysis, "
                "compensation research, interview prep, and legitimacy assessment. "
                "Saves the report to reports/ and adds a tracker entry."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "jd_text": {
                        "type": "string",
                        "description": "The full job description text to evaluate.",
                    },
                    "url": {
                        "type": "string",
                        "description": "URL of the job posting (optional, for reference).",
                    },
                },
                "required": ["jd_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "career_generate_cv",
            "description": (
                "Generate a tailored ATS-optimized CV/PDF for a specific job offer. "
                "Reads the user's cv.md, adapts content to match the JD keywords, "
                "and generates a PDF via Playwright."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "jd_text": {
                        "type": "string",
                        "description": "The full job description text to tailor the CV for.",
                    },
                    "company": {
                        "type": "string",
                        "description": "Company name (used in PDF filename).",
                    },
                    "url": {
                        "type": "string",
                        "description": "URL of the job posting (optional).",
                    },
                },
                "required": ["jd_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "career_scan_portals",
            "description": (
                "Scan configured job portals (Greenhouse, Ashby, Lever, BambooHR, etc.) "
                "for new job offers matching the user's target roles. "
                "Reads portals.yml for configuration and filters by title keywords."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "career_get_tracker",
            "description": (
                "Get the current application tracker showing all evaluated/applied offers "
                "with their scores, statuses, and report links."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "career_merge_tracker",
            "description": (
                "Merge pending tracker additions into the main applications.md file. "
                "Run this after evaluating multiple offers to consolidate the tracker."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "career_prepare_application",
            "description": (
                "Prepare application materials for a job offer: tailored CV/PDF + "
                "draft answers for application form fields. Does NOT submit the application — "
                "returns materials for the user to review first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "jd_text": {
                        "type": "string",
                        "description": "The full job description text.",
                    },
                    "company": {
                        "type": "string",
                        "description": "Company name.",
                    },
                    "url": {
                        "type": "string",
                        "description": "URL of the job posting.",
                    },
                    "form_fields": {
                        "type": "string",
                        "description": (
                            "Application form fields to draft answers for. "
                            "List each field on a new line."
                        ),
                    },
                },
                "required": ["jd_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "career_get_report",
            "description": (
                "Read a specific evaluation report by its number or filename. "
                "Example: '001' or '001-google-2026-04-13.md'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "report_id": {
                        "type": "string",
                        "description": "Report number (e.g. '001') or full filename.",
                    },
                },
                "required": ["report_id"],
            },
        },
    },
]
