"""
Career-ops job search skill for Open-Sable.

Integrates with career-ops (https://github.com/santifer/career-ops) to provide:
- Job offer evaluation via Ollama (local LLM)
- Tailored CV/PDF generation
- Portal scanning for new offers
- Application tracking
- Email-based job applications

Requires: career-ops repo cloned at a configured path, Ollama running locally.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default career-ops path
_DEFAULT_CAREER_OPS_PATH = os.path.expanduser("~/career-ops")


class CareerOpsSkill:
    """Job search automation powered by career-ops + Ollama."""

    def __init__(self, config):
        self.config = config
        self._initialized = False

        # career-ops repo path (configurable via CAREER_OPS_PATH env or config)
        self.career_ops_path = Path(
            getattr(config, "career_ops_path", None)
            or os.environ.get("CAREER_OPS_PATH", _DEFAULT_CAREER_OPS_PATH)
        )

        # Ollama settings
        self.ollama_base_url = getattr(
            config, "ollama_base_url", None
        ) or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = getattr(
            config, "career_ops_model", None
        ) or os.environ.get("CAREER_OPS_MODEL", "gemma4:31b")

    async def initialize(self) -> None:
        """Validate career-ops directory and Ollama availability."""
        if self._initialized:
            return

        # Check career-ops directory
        if not self.career_ops_path.exists():
            logger.warning(
                f"career-ops path not found: {self.career_ops_path}. "
                "Set CAREER_OPS_PATH env var or career_ops_path in config."
            )
            return

        required_files = ["cv.md", "config/profile.yml"]
        for f in required_files:
            if not (self.career_ops_path / f).exists():
                logger.warning(f"career-ops missing required file: {f}")
                return

        # Check Ollama is reachable
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.ollama_base_url}/api/tags")
                if resp.status_code == 200:
                    models = [m["name"] for m in resp.json().get("models", [])]
                    if not any(self.model in m for m in models):
                        logger.warning(
                            f"Model {self.model} not found in Ollama. "
                            f"Available: {models}"
                        )
                        return
        except Exception as e:
            logger.warning(f"Ollama not reachable at {self.ollama_base_url}: {e}")
            return

        self._initialized = True
        logger.info(
            f"✅ CareerOps skill ready (path={self.career_ops_path}, model={self.model})"
        )

    def is_ready(self) -> bool:
        return self._initialized

    # ── Internal helpers ─────────────────────────────────────────────────

    def _read_file(self, relative_path: str) -> str:
        """Read a file from career-ops directory."""
        path = self.career_ops_path / relative_path
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _write_file(self, relative_path: str, content: str) -> None:
        """Write a file in career-ops directory."""
        path = self.career_ops_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    async def _ollama_generate(self, prompt: str, system: str = "") -> str:
        """Call Ollama's generate API."""
        import httpx

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 4096},
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")

    def _get_next_report_number(self) -> int:
        """Get next sequential report number."""
        reports_dir = self.career_ops_path / "reports"
        reports_dir.mkdir(exist_ok=True)
        existing = list(reports_dir.glob("*.md"))
        if not existing:
            return 1
        numbers = []
        for f in existing:
            match = re.match(r"^(\d+)-", f.name)
            if match:
                numbers.append(int(match.group(1)))
        return max(numbers, default=0) + 1

    def _build_system_prompt(self) -> str:
        """Build system prompt from career-ops mode files."""
        parts = []

        shared = self._read_file("modes/_shared.md")
        if shared:
            parts.append(shared)

        profile = self._read_file("modes/_profile.md")
        if profile:
            parts.append(profile)

        return "\n\n---\n\n".join(parts)

    def _build_cv_context(self) -> str:
        """Build CV context string."""
        cv = self._read_file("cv.md")
        profile_yml = self._read_file("config/profile.yml")
        digest = self._read_file("article-digest.md")

        parts = [f"## CV\n\n{cv}"]
        if profile_yml:
            parts.append(f"## Profile Config\n\n```yaml\n{profile_yml}\n```")
        if digest:
            parts.append(f"## Article Digest (proof points)\n\n{digest}")

        return "\n\n".join(parts)

    # ── Public API ───────────────────────────────────────────────────────

    async def evaluate_offer(self, jd_text: str, url: str = "") -> Dict[str, Any]:
        """
        Evaluate a job offer using Ollama.
        Returns evaluation report with scoring blocks A-G.
        """
        system_prompt = self._build_system_prompt()
        cv_context = self._build_cv_context()
        oferta_mode = self._read_file("modes/oferta.md")

        prompt = f"""{oferta_mode}

---

{cv_context}

---

## Job Description to Evaluate

URL: {url or 'Not provided'}

{jd_text}

---

Generate the complete evaluation (Blocks A through F). For Block D (Comp), use your knowledge of market rates. For Block G (Legitimacy), analyze the JD text for signals.

Output the full evaluation in markdown format."""

        report = await self._ollama_generate(prompt, system=system_prompt)

        # Save report
        num = self._get_next_report_number()
        date = datetime.now().strftime("%Y-%m-%d")

        # Extract company name from report or JD
        company_slug = "unknown"
        company_match = re.search(r"(?:Company|Empresa|company)[:\s]+([^\n|]+)", jd_text)
        if company_match:
            company_slug = re.sub(r"[^a-z0-9]+", "-", company_match.group(1).lower().strip()).strip("-")
        if not company_slug or company_slug == "unknown":
            # Try first line of JD
            first_line = jd_text.strip().split("\n")[0][:50]
            company_slug = re.sub(r"[^a-z0-9]+", "-", first_line.lower()).strip("-")[:30]

        report_filename = f"{num:03d}-{company_slug}-{date}.md"
        report_header = f"**URL:** {url or 'N/A'}\n\n"
        full_report = report_header + report

        self._write_file(f"reports/{report_filename}", full_report)

        # Extract score from report
        score = "N/A"
        score_match = re.search(r"(\d\.\d)/5", report)
        if score_match:
            score = score_match.group(0)

        # Write tracker TSV
        tsv_line = (
            f"{num}\t{date}\t{company_slug}\t"
            f"(see report)\tEvaluated\t{score}\t❌\t"
            f"[{num:03d}](reports/{report_filename})\tOllama evaluation"
        )
        self._write_file(
            f"batch/tracker-additions/{num:03d}-{company_slug}.tsv",
            tsv_line,
        )

        return {
            "report": report,
            "report_file": f"reports/{report_filename}",
            "score": score,
            "company": company_slug,
            "number": num,
        }

    async def generate_cv(
        self, jd_text: str, company: str = "", url: str = ""
    ) -> Dict[str, Any]:
        """
        Generate a tailored CV PDF for a specific job offer.
        Uses Ollama to adapt CV content, then Playwright for PDF.
        """
        system_prompt = self._build_system_prompt()
        cv_context = self._build_cv_context()
        pdf_mode = self._read_file("modes/pdf.md")
        template_html = self._read_file("templates/cv-template.html")
        profile_yml = self._read_file("config/profile.yml")

        # Extract candidate name for filename
        name_match = re.search(r"full_name:\s*\"?([^\"]+)\"?", profile_yml)
        candidate = "candidate"
        if name_match:
            candidate = re.sub(r"[^a-z0-9]+", "-", name_match.group(1).lower()).strip("-")

        company_slug = re.sub(r"[^a-z0-9]+", "-", company.lower()).strip("-") if company else "general"
        date = datetime.now().strftime("%Y-%m-%d")

        prompt = f"""{pdf_mode}

---

{cv_context}

---

## HTML Template

```html
{template_html}
```

---

## Job Description

Company: {company}
URL: {url or 'N/A'}

{jd_text}

---

Generate ONLY the complete HTML document (no explanation, no markdown wrapping). Replace all {{{{placeholders}}}} in the template with the tailored content. Adapt the CV to match this specific job offer. Output raw HTML only."""

        html_content = await self._ollama_generate(prompt, system=system_prompt)

        # Clean any markdown code fences
        html_content = re.sub(r"^```html?\s*\n?", "", html_content)
        html_content = re.sub(r"\n?```\s*$", "", html_content)

        # Write HTML
        html_path = f"/tmp/cv-{candidate}-{company_slug}.html"
        Path(html_path).write_text(html_content, encoding="utf-8")

        # Generate PDF using career-ops script
        pdf_filename = f"cv-{candidate}-{company_slug}-{date}.pdf"
        pdf_path = str(self.career_ops_path / "output" / pdf_filename)

        try:
            result = await asyncio.create_subprocess_exec(
                "node",
                str(self.career_ops_path / "generate-pdf.mjs"),
                html_path,
                pdf_path,
                "--format=a4",
                cwd=str(self.career_ops_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                error = stderr.decode() if stderr else "Unknown error"
                return {
                    "success": False,
                    "error": f"PDF generation failed: {error}",
                    "html_path": html_path,
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"PDF generation failed: {e}",
                "html_path": html_path,
            }

        return {
            "success": True,
            "pdf_path": pdf_path,
            "html_path": html_path,
            "company": company_slug,
        }

    async def scan_portals(self) -> Dict[str, Any]:
        """
        Scan job portals using career-ops ATS API scanner.
        Reads portals.yml, hits Greenhouse/Ashby/Lever/etc APIs, filters results.
        """
        import yaml

        portals_file = self.career_ops_path / "portals.yml"
        if not portals_file.exists():
            return {"error": "portals.yml not found", "offers": []}

        portals = yaml.safe_load(portals_file.read_text())
        title_filter = portals.get("title_filter", {})
        positive_kw = [k.lower() for k in title_filter.get("positive", [])]
        negative_kw = [k.lower() for k in title_filter.get("negative", [])]

        # Read scan history for dedup
        history_file = self.career_ops_path / "data" / "scan-history.tsv"
        seen_urls = set()
        if history_file.exists():
            for line in history_file.read_text().splitlines():
                parts = line.split("\t")
                if len(parts) >= 2:
                    seen_urls.add(parts[1])

        import httpx

        all_offers = []
        companies = portals.get("tracked_companies", [])

        for company in companies:
            if not company.get("enabled", True):
                continue

            api_url = company.get("api")
            if not api_url:
                continue

            provider = company.get("api_provider", "")
            name = company.get("name", "unknown")

            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    if "ashbyhq.com" in api_url or provider == "ashby":
                        resp = await client.post(
                            api_url,
                            json={
                                "operationName": "ApiJobBoardWithTeams",
                                "variables": {
                                    "organizationHostedJobsPageName": company.get("slug", name.lower())
                                },
                                "query": (
                                    "query ApiJobBoardWithTeams($organizationHostedJobsPageName: String!) {"
                                    " jobBoard: jobBoardWithTeams(organizationHostedJobsPageName: $organizationHostedJobsPageName) {"
                                    " jobPostings { id title locationName employmentType } } }"
                                ),
                            },
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            postings = (
                                data.get("data", {})
                                .get("jobBoard", {})
                                .get("jobPostings", [])
                            )
                            for p in postings:
                                job_url = f"https://jobs.ashbyhq.com/{company.get('slug', name.lower())}/{p['id']}"
                                all_offers.append({
                                    "title": p.get("title", ""),
                                    "url": job_url,
                                    "company": name,
                                    "location": p.get("locationName", ""),
                                })

                    elif "greenhouse.io" in api_url or provider == "greenhouse":
                        resp = await client.get(api_url)
                        if resp.status_code == 200:
                            jobs = resp.json().get("jobs", [])
                            for j in jobs:
                                all_offers.append({
                                    "title": j.get("title", ""),
                                    "url": j.get("absolute_url", ""),
                                    "company": name,
                                    "location": (
                                        j.get("location", {}).get("name", "")
                                        if isinstance(j.get("location"), dict)
                                        else ""
                                    ),
                                })

                    elif "lever.co" in api_url or provider == "lever":
                        resp = await client.get(api_url)
                        if resp.status_code == 200:
                            for j in resp.json():
                                all_offers.append({
                                    "title": j.get("text", ""),
                                    "url": j.get("hostedUrl", j.get("applyUrl", "")),
                                    "company": name,
                                    "location": (
                                        j.get("categories", {}).get("location", "")
                                        if isinstance(j.get("categories"), dict)
                                        else ""
                                    ),
                                })

                    elif "bamboohr.com" in api_url or provider == "bamboohr":
                        resp = await client.get(api_url)
                        if resp.status_code == 200:
                            results = resp.json().get("result", [])
                            slug = company.get("slug", name.lower())
                            for r in results:
                                job_id = r.get("id", "")
                                all_offers.append({
                                    "title": r.get("jobOpeningName", ""),
                                    "url": f"https://{slug}.bamboohr.com/careers/{job_id}/detail",
                                    "company": name,
                                    "location": r.get("location", {}).get("city", ""),
                                })

                    elif "myworkdayjobs.com" in api_url or provider == "workday":
                        resp = await client.post(
                            api_url,
                            json={"appliedFacets": {}, "limit": 20, "offset": 0, "searchText": ""},
                        )
                        if resp.status_code == 200:
                            postings = resp.json().get("jobPostings", [])
                            for p in postings:
                                all_offers.append({
                                    "title": p.get("title", ""),
                                    "url": p.get("externalPath", ""),
                                    "company": name,
                                    "location": "",
                                })

                    elif ".teamtailor.com" in api_url or provider == "teamtailor":
                        resp = await client.get(api_url)
                        if resp.status_code == 200:
                            # Simple RSS XML parsing
                            import xml.etree.ElementTree as ET
                            root = ET.fromstring(resp.text)
                            for item in root.iter("item"):
                                title_el = item.find("title")
                                link_el = item.find("link")
                                if title_el is not None and link_el is not None:
                                    all_offers.append({
                                        "title": title_el.text or "",
                                        "url": link_el.text or "",
                                        "company": name,
                                        "location": "",
                                    })

                    else:
                        # Generic JSON endpoint
                        resp = await client.get(api_url)
                        if resp.status_code == 200:
                            data = resp.json()
                            if isinstance(data, list):
                                for j in data:
                                    all_offers.append({
                                        "title": j.get("title", j.get("text", "")),
                                        "url": j.get("url", j.get("absolute_url", j.get("hostedUrl", ""))),
                                        "company": name,
                                        "location": "",
                                    })

            except Exception as e:
                logger.warning(f"Scan failed for {name}: {e}")
                continue

        # Filter by title keywords
        filtered = []
        for offer in all_offers:
            title_lower = offer["title"].lower()

            # Skip if matches negative keywords
            if any(neg in title_lower for neg in negative_kw):
                continue

            # Skip if already seen
            if offer["url"] in seen_urls:
                continue

            # Include if matches positive keywords (or include all if no positives defined)
            if not positive_kw or any(pos in title_lower for pos in positive_kw):
                filtered.append(offer)

        # Update scan history
        if filtered:
            date = datetime.now().strftime("%Y-%m-%d")
            history_lines = []
            for o in filtered:
                history_lines.append(f"{date}\t{o['url']}\t{o['company']}\t{o['title']}")
            history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(history_file, "a", encoding="utf-8") as f:
                f.write("\n".join(history_lines) + "\n")

        return {
            "total_scanned": len(all_offers),
            "filtered_new": len(filtered),
            "offers": filtered[:50],  # Limit response size
        }

    async def get_tracker(self) -> str:
        """Get current application tracker status."""
        tracker = self._read_file("data/applications.md")
        if not tracker.strip():
            return "📋 No applications tracked yet."
        return tracker

    async def merge_tracker(self) -> str:
        """Run merge-tracker.mjs to consolidate tracker additions."""
        try:
            result = await asyncio.create_subprocess_exec(
                "node",
                str(self.career_ops_path / "merge-tracker.mjs"),
                cwd=str(self.career_ops_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()
            output = stdout.decode() if stdout else ""
            if result.returncode != 0:
                error = stderr.decode() if stderr else ""
                return f"❌ Merge failed: {error}"
            return f"✅ Tracker merged. {output}"
        except Exception as e:
            return f"❌ Merge failed: {e}"

    async def prepare_application(
        self, jd_text: str, company: str = "", url: str = "", form_fields: str = ""
    ) -> Dict[str, Any]:
        """
        Prepare application materials: tailored CV + draft answers for form fields.
        Does NOT submit — returns materials for user review.
        """
        # Generate CV
        cv_result = await self.generate_cv(jd_text, company=company, url=url)

        # If there are form fields, generate draft answers
        draft_answers = ""
        if form_fields:
            system_prompt = self._build_system_prompt()
            cv_context = self._build_cv_context()

            prompt = f"""{cv_context}

---

## Job Description
Company: {company}
{jd_text}

---

## Application Form Fields
{form_fields}

---

For each form field, draft a concise answer based on the CV and job description.
Be specific, use proof points from the CV. No corporate-speak.
Format: one section per field with the field name as header."""

            draft_answers = await self._ollama_generate(prompt, system=system_prompt)

        return {
            "cv": cv_result,
            "draft_answers": draft_answers,
            "status": "ready_for_review",
            "message": (
                "⚠️ Application materials ready for review. "
                "Please review before submitting. The agent will NOT submit automatically."
            ),
        }

    async def get_report(self, report_id: str) -> str:
        """Read a specific evaluation report."""
        reports_dir = self.career_ops_path / "reports"
        # Try exact match first
        exact = reports_dir / report_id
        if exact.exists():
            return exact.read_text(encoding="utf-8")

        # Try matching by number prefix
        for f in reports_dir.glob(f"{report_id}*"):
            return f.read_text(encoding="utf-8")

        return f"❌ Report not found: {report_id}"
