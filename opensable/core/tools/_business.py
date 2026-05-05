"""
Business automation tools,  CRM, pipeline, email templates, follow-ups.

Provides tool handlers that delegate to CRMSkill, PipelineSkill,
EmailTemplatesSkill, and FollowUpSkill, following the mixin pattern.
"""

import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)

_NOT_READY = "❌ Business tools not available (pip install aiosqlite)"


class BusinessToolsMixin:
    """Mixin providing business automation tool implementations for ToolRegistry."""

    def _crm_ready(self) -> bool:
        return self.crm_skill is not None and self.crm_skill.is_ready()

    def _pipeline_ready(self) -> bool:
        return self.pipeline_skill is not None and self.pipeline_skill.is_ready()

    def _templates_ready(self) -> bool:
        return self.templates_skill is not None and self.templates_skill.is_ready()

    def _followup_ready(self) -> bool:
        return self.followup_skill is not None and self.followup_skill.is_ready()

    # ══════════════════════════════════════════════════════════════════════
    # CRM CONTACTS
    # ══════════════════════════════════════════════════════════════════════

    async def _crm_add_contact_tool(self, params: Dict) -> str:
        if not self._crm_ready():
            return _NOT_READY
        try:
            result = await self.crm_skill.add_contact(
                name=params.get("name", ""),
                email=params.get("email", ""),
                phone=params.get("phone", ""),
                company=params.get("company", ""),
                country=params.get("country", ""),
                city=params.get("city", ""),
                role=params.get("role", "prospect"),
                industry=params.get("industry", ""),
                products=params.get("products", ""),
                source=params.get("source", "manual"),
                tags=params.get("tags", ""),
                notes=params.get("notes", ""),
                website=params.get("website", ""),
                score=params.get("score", 0),
            )
            return f"✅ Contact added: {result['name']} (ID: {result['id']}, status: {result['status']})"
        except Exception as e:
            return f"❌ Error adding contact: {e}"

    async def _crm_update_contact_tool(self, params: Dict) -> str:
        if not self._crm_ready():
            return _NOT_READY
        try:
            contact_id = params.pop("contact_id", "")
            if not contact_id:
                return "❌ contact_id is required"
            result = await self.crm_skill.update_contact(contact_id, **params)
            if "error" in result:
                return f"❌ {result['error']}"
            return f"✅ Updated contact {result['updated']}: {', '.join(result['fields'])}"
        except Exception as e:
            return f"❌ Error updating contact: {e}"

    async def _crm_get_contact_tool(self, params: Dict) -> str:
        if not self._crm_ready():
            return _NOT_READY
        try:
            contact = await self.crm_skill.get_contact(params.get("contact_id", ""))
            if not contact:
                return "❌ Contact not found"
            lines = [f"👤 {contact['name']} ({contact.get('role', '?')})"]
            if contact.get("email"):
                lines.append(f"  📧 {contact['email']}")
            if contact.get("company"):
                lines.append(f"  🏢 {contact['company']}")
            if contact.get("country"):
                lines.append(f"  🌍 {contact.get('city', '')}, {contact['country']}")
            if contact.get("products"):
                lines.append(f"  📦 Products: {contact['products']}")
            lines.append(f"  Status: {contact['status']} | Score: {contact.get('score', 0)}")
            if contact.get("notes"):
                lines.append(f"  📝 {contact['notes'][:200]}")
            if contact.get("last_contact"):
                lines.append(f"  Last contact: {contact['last_contact']}")
            if contact.get("next_followup"):
                lines.append(f"  Next follow-up: {contact['next_followup']}")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error: {e}"

    async def _crm_search_contacts_tool(self, params: Dict) -> str:
        if not self._crm_ready():
            return _NOT_READY
        try:
            contacts = await self.crm_skill.search_contacts(
                query=params.get("query", ""),
                role=params.get("role", ""),
                status=params.get("status", ""),
                country=params.get("country", ""),
                tags=params.get("tags", ""),
                limit=params.get("limit", 20),
            )
            if not contacts:
                return "📭 No contacts found matching your criteria."
            lines = [f"📋 {len(contacts)} contacts found:"]
            for c in contacts:
                line = f"  • [{c['id']}] {c['name']}"
                if c.get("company"):
                    line += f" @ {c['company']}"
                if c.get("country"):
                    line += f" ({c['country']})"
                line += f",  {c['role']}/{c['status']}"
                lines.append(line)
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error: {e}"

    async def _crm_delete_contact_tool(self, params: Dict) -> str:
        if not self._crm_ready():
            return _NOT_READY
        try:
            result = await self.crm_skill.delete_contact(params.get("contact_id", ""))
            return f"🗑️ Deleted contact {result['deleted']} ({result['rows']} rows removed)"
        except Exception as e:
            return f"❌ Error: {e}"

    async def _crm_log_activity_tool(self, params: Dict) -> str:
        if not self._crm_ready():
            return _NOT_READY
        try:
            metadata = params.get("metadata")
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {"raw": metadata}
            result = await self.crm_skill.log_activity(
                contact_id=params.get("contact_id", ""),
                activity_type=params.get("activity_type", "note"),
                subject=params.get("subject", ""),
                body=params.get("body", ""),
                metadata=metadata,
            )
            return f"📝 Activity logged: {result['type']} for contact {result['contact_id']}"
        except Exception as e:
            return f"❌ Error: {e}"

    async def _crm_get_activities_tool(self, params: Dict) -> str:
        if not self._crm_ready():
            return _NOT_READY
        try:
            activities = await self.crm_skill.get_activities(
                contact_id=params.get("contact_id", ""),
                limit=params.get("limit", 20),
            )
            if not activities:
                return "📭 No activities found for this contact."
            lines = [f"📋 {len(activities)} activities:"]
            for a in activities:
                lines.append(f"  • [{a['created_at'][:10]}] {a['type']}: {a.get('subject', '')[:80]}")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error: {e}"

    async def _crm_stats_tool(self, params: Dict) -> str:
        if not self._crm_ready():
            return _NOT_READY
        try:
            stats = await self.crm_skill.get_stats()
            lines = [f"📊 CRM Statistics:"]
            lines.append(f"  Total contacts: {stats['total_contacts']}")
            if stats.get("by_role"):
                lines.append(f"  By role: {json.dumps(stats['by_role'])}")
            if stats.get("by_status"):
                lines.append(f"  By status: {json.dumps(stats['by_status'])}")
            if stats.get("top_countries"):
                lines.append(f"  Top countries: {json.dumps(stats['top_countries'])}")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error: {e}"

    # ══════════════════════════════════════════════════════════════════════
    # PIPELINE / DEALS
    # ══════════════════════════════════════════════════════════════════════

    async def _pipeline_create_deal_tool(self, params: Dict) -> str:
        if not self._pipeline_ready():
            return _NOT_READY
        try:
            result = await self.pipeline_skill.create_deal(
                title=params.get("title", ""),
                buyer_id=params.get("buyer_id", ""),
                manufacturer_id=params.get("manufacturer_id", ""),
                product=params.get("product", ""),
                quantity=params.get("quantity", ""),
                unit=params.get("unit", "meters"),
                quality_requirements=params.get("quality_requirements", ""),
                delivery_date=params.get("delivery_date", ""),
                price_range=params.get("price_range", ""),
                currency=params.get("currency", "EUR"),
                priority=params.get("priority", "medium"),
                notes=params.get("notes", ""),
                value_estimate=params.get("value_estimate", 0),
            )
            return f"✅ Deal created: '{result['title']}' (ID: {result['id']}, stage: {result['stage']})"
        except Exception as e:
            return f"❌ Error: {e}"

    async def _pipeline_advance_deal_tool(self, params: Dict) -> str:
        if not self._pipeline_ready():
            return _NOT_READY
        try:
            result = await self.pipeline_skill.advance_deal(
                deal_id=params.get("deal_id", ""),
                new_stage=params.get("new_stage", ""),
                reason=params.get("reason", ""),
            )
            if "error" in result:
                return f"❌ {result['error']}"
            return f"✅ Deal moved: {result['from']} → {result['to']}"
        except Exception as e:
            return f"❌ Error: {e}"

    async def _pipeline_get_deal_tool(self, params: Dict) -> str:
        if not self._pipeline_ready():
            return _NOT_READY
        try:
            deal = await self.pipeline_skill.get_deal(params.get("deal_id", ""))
            if not deal:
                return "❌ Deal not found"
            lines = [f"💼 {deal['title']} [{deal['stage'].upper()}]"]
            if deal.get("product"):
                lines.append(f"  📦 {deal['product']},  {deal.get('quantity', '?')} {deal.get('unit', '')}")
            if deal.get("quality_requirements"):
                lines.append(f"  ✅ Quality: {deal['quality_requirements']}")
            if deal.get("delivery_date"):
                lines.append(f"  📅 Delivery: {deal['delivery_date']}")
            if deal.get("price_range"):
                lines.append(f"  💰 Price: {deal['price_range']} {deal.get('currency', '')}")
            lines.append(f"  Priority: {deal.get('priority', '?')} | Value: €{deal.get('value_estimate', 0):,.0f}")
            if deal.get("buyer_id"):
                lines.append(f"  Buyer ID: {deal['buyer_id']}")
            if deal.get("manufacturer_id"):
                lines.append(f"  Manufacturer ID: {deal['manufacturer_id']}")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error: {e}"

    async def _pipeline_list_deals_tool(self, params: Dict) -> str:
        if not self._pipeline_ready():
            return _NOT_READY
        try:
            deals = await self.pipeline_skill.list_deals(
                stage=params.get("stage", ""),
                buyer_id=params.get("buyer_id", ""),
                manufacturer_id=params.get("manufacturer_id", ""),
                limit=params.get("limit", 20),
            )
            if not deals:
                return "📭 No deals found."
            lines = [f"💼 {len(deals)} deals:"]
            for d in deals:
                val = f" (€{d.get('value_estimate', 0):,.0f})" if d.get("value_estimate") else ""
                lines.append(f"  • [{d['id']}] {d['title']},  {d['stage']}{val}")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error: {e}"

    async def _pipeline_update_deal_tool(self, params: Dict) -> str:
        if not self._pipeline_ready():
            return _NOT_READY
        try:
            deal_id = params.pop("deal_id", "")
            if not deal_id:
                return "❌ deal_id is required"
            result = await self.pipeline_skill.update_deal(deal_id, **params)
            if "error" in result:
                return f"❌ {result['error']}"
            return f"✅ Deal {result['updated']} updated: {', '.join(result['fields'])}"
        except Exception as e:
            return f"❌ Error: {e}"

    async def _pipeline_stats_tool(self, params: Dict) -> str:
        if not self._pipeline_ready():
            return _NOT_READY
        try:
            stats = await self.pipeline_skill.get_pipeline_stats()
            lines = [f"📊 Pipeline Statistics:"]
            lines.append(f"  Total deals: {stats['total_deals']}")
            for stage, info in stats.get("by_stage", {}).items():
                lines.append(f"  {stage}: {info['count']} deals (€{info['value']:,.0f})")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error: {e}"

    async def _pipeline_match_tool(self, params: Dict) -> str:
        if not self._pipeline_ready():
            return _NOT_READY
        try:
            result = await self.pipeline_skill.match_buyer_manufacturers(
                deal_id=params.get("deal_id", ""),
            )
            if "error" in result:
                return f"❌ {result['error']}"
            deal = result["deal"]
            buyer = result.get("buyer")
            mfr = result.get("manufacturer")
            lines = [f"🔗 Match for deal: {deal['title']}"]
            lines.append(f"  Product: {deal.get('product', '?')},  {deal.get('quantity', '?')} {deal.get('unit', '')}")
            if buyer:
                lines.append(f"  🛒 Buyer: {buyer['name']} @ {buyer.get('company', '?')} ({buyer.get('country', '?')})")
            else:
                lines.append(f"  🛒 Buyer: not assigned")
            if mfr:
                lines.append(f"  🏭 Manufacturer: {mfr['name']} @ {mfr.get('company', '?')} ({mfr.get('country', '?')})")
            else:
                lines.append(f"  🏭 Manufacturer: not assigned")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error: {e}"

    # ══════════════════════════════════════════════════════════════════════
    # EMAIL TEMPLATES
    # ══════════════════════════════════════════════════════════════════════

    async def _template_list_tool(self, params: Dict) -> str:
        if not self._templates_ready():
            return _NOT_READY
        try:
            templates = await self.templates_skill.list_templates(
                category=params.get("category", ""),
            )
            if not templates:
                return "📭 No templates found."
            lines = [f"📋 {len(templates)} email templates:"]
            for t in templates:
                lines.append(f"  • {t['name']} [{t['category']}],  {t['subject'][:60]}")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error: {e}"

    async def _template_get_tool(self, params: Dict) -> str:
        if not self._templates_ready():
            return _NOT_READY
        try:
            tpl = await self.templates_skill.get_template(params.get("name", ""))
            if not tpl:
                return "❌ Template not found"
            fields = await self.templates_skill.preview_merge_fields(tpl["name"])
            lines = [
                f"📧 Template: {tpl['name']} [{tpl['category']}]",
                f"Subject: {tpl['subject']}",
                f"Merge fields: {', '.join(fields) if fields else 'none'}",
                f"---",
                tpl["body"],
            ]
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error: {e}"

    async def _template_save_tool(self, params: Dict) -> str:
        if not self._templates_ready():
            return _NOT_READY
        try:
            result = await self.templates_skill.save_template(
                name=params.get("name", ""),
                subject=params.get("subject", ""),
                body=params.get("body", ""),
                category=params.get("category", "outreach"),
                language=params.get("language", "en"),
                tags=params.get("tags", ""),
            )
            return f"✅ Template '{result['name']}' {result['action']}"
        except Exception as e:
            return f"❌ Error: {e}"

    async def _template_render_tool(self, params: Dict) -> str:
        if not self._templates_ready():
            return _NOT_READY
        try:
            fields = params.get("fields", {})
            if isinstance(fields, str):
                try:
                    fields = json.loads(fields)
                except (json.JSONDecodeError, TypeError):
                    return "❌ 'fields' must be a JSON object"
            result = await self.templates_skill.render_template(
                name=params.get("name", ""),
                fields=fields,
            )
            if not result:
                return "❌ Template not found"
            return f"📧 Rendered email:\nSubject: {result['subject']}\n---\n{result['body']}"
        except Exception as e:
            return f"❌ Error: {e}"

    async def _template_delete_tool(self, params: Dict) -> str:
        if not self._templates_ready():
            return _NOT_READY
        try:
            result = await self.templates_skill.delete_template(params.get("name", ""))
            return f"🗑️ Template '{result['deleted']}' deleted ({result['rows']} rows)"
        except Exception as e:
            return f"❌ Error: {e}"

    # ══════════════════════════════════════════════════════════════════════
    # FOLLOW-UPS
    # ══════════════════════════════════════════════════════════════════════

    async def _followup_recommendations_tool(self, params: Dict) -> str:
        if not self._followup_ready():
            return _NOT_READY
        try:
            recs = await self.followup_skill.get_followup_recommendations(
                max_recommendations=params.get("max_recommendations", 10),
                stale_days=params.get("stale_days", 5),
                stalling_days=params.get("stalling_days", 7),
            )
            if not recs:
                return "✅ No follow-ups needed,  everything is up to date!"
            lines = [f"📋 {len(recs)} follow-up recommendations:"]
            for i, r in enumerate(recs, 1):
                priority_icon = "🔴" if r["priority"] == "high" else "🟡"
                lines.append(f"  {i}. {priority_icon} [{r['type']}] {r['suggested_action']}")
                if r.get("template"):
                    lines.append(f"     Template: {r['template']}")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error: {e}"

    async def _followup_overdue_tool(self, params: Dict) -> str:
        if not self._followup_ready():
            return _NOT_READY
        try:
            contacts = await self.followup_skill.get_overdue_contacts(
                limit=params.get("limit", 20),
            )
            if not contacts:
                return "✅ No overdue follow-ups!"
            lines = [f"⚠️ {len(contacts)} overdue follow-ups:"]
            for c in contacts:
                lines.append(
                    f"  • {c['name']} @ {c.get('company', '?')} "
                    f"({c.get('email', 'no email')}),  due: {c.get('next_followup', '?')[:10]}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error: {e}"

    async def _followup_stale_tool(self, params: Dict) -> str:
        if not self._followup_ready():
            return _NOT_READY
        try:
            contacts = await self.followup_skill.get_stale_contacts(
                days_since_contact=params.get("days", 5),
                limit=params.get("limit", 20),
            )
            if not contacts:
                return "✅ No stale contacts!"
            lines = [f"⏰ {len(contacts)} contacts with no reply:"]
            for c in contacts:
                lines.append(
                    f"  • {c['name']} @ {c.get('company', '?')} "
                    f",  last contact: {c.get('last_contact', '?')[:10]}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error: {e}"

    async def _followup_summary_tool(self, params: Dict) -> str:
        if not self._followup_ready():
            return _NOT_READY
        try:
            summary = await self.followup_skill.get_business_summary()
            return f"📊 Business Summary:\n{summary}"
        except Exception as e:
            return f"❌ Error: {e}"
