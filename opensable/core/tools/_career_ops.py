"""
Career-ops tool mixin for ToolRegistry.
Provides job search, evaluation, CV generation, and application tracking tools.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

_NOT_READY = (
    "❌ Career-ops skill not initialized. Ensure:\n"
    "1. career-ops repo is at ~/career-ops (or set CAREER_OPS_PATH)\n"
    "2. cv.md and config/profile.yml exist\n"
    "3. Ollama is running with the configured model"
)


class CareerOpsToolsMixin:
    """Mixin providing career-ops tool implementations for ToolRegistry."""

    def _career_ops_ready(self) -> bool:
        return self.career_ops_skill is not None and self.career_ops_skill.is_ready()

    # ══════════════════════════════════════════════════════════════════════
    # EVALUATE OFFER
    # ══════════════════════════════════════════════════════════════════════

    async def _career_evaluate_offer_tool(self, params: Dict[str, Any]) -> str:
        if not self._career_ops_ready():
            return _NOT_READY
        try:
            result = await self.career_ops_skill.evaluate_offer(
                jd_text=params.get("jd_text", ""),
                url=params.get("url", ""),
            )
            lines = [
                f"📊 **Evaluation Complete** — Score: {result['score']}",
                f"📄 Report: {result['report_file']}",
                f"🏢 Company: {result['company']}",
                "",
                result["report"][:3000],  # Truncate for tool response
            ]
            if len(result["report"]) > 3000:
                lines.append(f"\n... (full report in {result['report_file']})")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Career-ops evaluate failed: {e}")
            return f"❌ Evaluation failed: {e}"

    # ══════════════════════════════════════════════════════════════════════
    # GENERATE CV
    # ══════════════════════════════════════════════════════════════════════

    async def _career_generate_cv_tool(self, params: Dict[str, Any]) -> str:
        if not self._career_ops_ready():
            return _NOT_READY
        try:
            result = await self.career_ops_skill.generate_cv(
                jd_text=params.get("jd_text", ""),
                company=params.get("company", ""),
                url=params.get("url", ""),
            )
            if result.get("success"):
                return (
                    f"✅ CV generated!\n"
                    f"📄 PDF: {result['pdf_path']}\n"
                    f"🔧 HTML: {result['html_path']}"
                )
            return f"❌ CV generation failed: {result.get('error', 'Unknown error')}"
        except Exception as e:
            logger.error(f"Career-ops CV gen failed: {e}")
            return f"❌ CV generation failed: {e}"

    # ══════════════════════════════════════════════════════════════════════
    # SCAN PORTALS
    # ══════════════════════════════════════════════════════════════════════

    async def _career_scan_portals_tool(self, params: Dict[str, Any]) -> str:
        if not self._career_ops_ready():
            return _NOT_READY
        try:
            result = await self.career_ops_skill.scan_portals()
            if "error" in result:
                return f"❌ Scan error: {result['error']}"

            offers = result.get("offers", [])
            lines = [
                f"🔍 **Portal Scan Complete**",
                f"Total scanned: {result['total_scanned']}",
                f"New matches: {result['filtered_new']}",
                "",
            ]
            if offers:
                lines.append("**New Offers:**")
                for i, o in enumerate(offers[:20], 1):
                    lines.append(
                        f"{i}. **{o['title']}** @ {o['company']}"
                        + (f" ({o['location']})" if o.get("location") else "")
                        + f"\n   {o['url']}"
                    )
                if len(offers) > 20:
                    lines.append(f"\n... and {len(offers) - 20} more")
            else:
                lines.append("No new matching offers found.")

            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Career-ops scan failed: {e}")
            return f"❌ Scan failed: {e}"

    # ══════════════════════════════════════════════════════════════════════
    # TRACKER
    # ══════════════════════════════════════════════════════════════════════

    async def _career_get_tracker_tool(self, params: Dict[str, Any]) -> str:
        if not self._career_ops_ready():
            return _NOT_READY
        try:
            return await self.career_ops_skill.get_tracker()
        except Exception as e:
            return f"❌ Tracker read failed: {e}"

    async def _career_merge_tracker_tool(self, params: Dict[str, Any]) -> str:
        if not self._career_ops_ready():
            return _NOT_READY
        try:
            return await self.career_ops_skill.merge_tracker()
        except Exception as e:
            return f"❌ Merge failed: {e}"

    # ══════════════════════════════════════════════════════════════════════
    # PREPARE APPLICATION
    # ══════════════════════════════════════════════════════════════════════

    async def _career_prepare_application_tool(self, params: Dict[str, Any]) -> str:
        if not self._career_ops_ready():
            return _NOT_READY
        try:
            result = await self.career_ops_skill.prepare_application(
                jd_text=params.get("jd_text", ""),
                company=params.get("company", ""),
                url=params.get("url", ""),
                form_fields=params.get("form_fields", ""),
            )
            lines = [result["message"], ""]
            cv = result.get("cv", {})
            if cv.get("success"):
                lines.append(f"📄 CV PDF: {cv['pdf_path']}")
            elif cv.get("error"):
                lines.append(f"⚠️ CV: {cv['error']}")

            if result.get("draft_answers"):
                lines.append("\n**Draft Answers:**\n")
                lines.append(result["draft_answers"][:2000])
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Career-ops application prep failed: {e}")
            return f"❌ Application prep failed: {e}"

    # ══════════════════════════════════════════════════════════════════════
    # READ REPORT
    # ══════════════════════════════════════════════════════════════════════

    async def _career_get_report_tool(self, params: Dict[str, Any]) -> str:
        if not self._career_ops_ready():
            return _NOT_READY
        try:
            return await self.career_ops_skill.get_report(
                report_id=params.get("report_id", ""),
            )
        except Exception as e:
            return f"❌ Report read failed: {e}"
