"""
Core tools,  file system, commands, browser, voice, image, database, RAG, code execution, API, skills
"""

import json
import logging
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)


class CoreToolsMixin:
    """Mixin providing core tools,  file system, commands, browser, voice, image, database, rag, code execution, api, skills tool implementations."""

    # ========== COMPUTER CONTROL TOOLS ==========

    async def _execute_command_tool(self, params: Dict) -> str:
        """Execute shell command"""
        command = params.get("command", "")
        cwd = params.get("cwd")
        timeout = params.get("timeout", 30)

        result = await self.computer.execute_command(command, cwd=cwd, timeout=timeout)

        if result["success"]:
            output = result["stdout"] if result["stdout"] else "(no output)"
            return f"✅ Command executed successfully\n\n```\n{output}\n```\n\nExit code: {result['exit_code']}"
        else:
            return (
                f"❌ Command failed\n\nError: {result['stderr']}\nExit code: {result['exit_code']}"
            )

    async def _read_file_tool(self, params: Dict) -> str:
        """Read file contents"""
        path = params.get("path", "")

        result = await self.computer.read_file(path)

        if result["success"]:
            content = result["content"]
            # Truncate very long files
            if len(content) > 10000:
                content = content[:10000] + f"\n... (truncated, total size: {result['size']} bytes)"
            return f"📄 File: {result['path']}\n\n```\n{content}\n```"
        else:
            return f"❌ Failed to read file: {result['error']}"

    async def _write_file_tool(self, params: Dict) -> str:
        """Write content to file"""
        path = params.get("path", "")
        content = params.get("content", "")
        mode = params.get("mode", "w")

        result = await self.computer.write_file(path, content, mode=mode)

        if result["success"]:
            return f"✅ Wrote {result['bytes_written']} bytes to {result['path']}"
        else:
            return f"❌ Failed to write file: {result['error']}"

    async def _edit_file_tool(self, params: Dict) -> str:
        """Edit file by replacing content"""
        path = params.get("path", "")
        old_content = params.get("old_content", "")
        new_content = params.get("new_content", "")

        result = await self.computer.edit_file(path, old_content, new_content)

        if result["success"]:
            return f"✅ Made {result['replacements']} replacement(s) in {result['path']}"
        else:
            return f"❌ Failed to edit file: {result['error']}"

    async def _list_directory_tool(self, params: Dict) -> str:
        """List directory contents"""
        path = params.get("path", ".")
        include_hidden = params.get("include_hidden", False)
        recursive = params.get("recursive", False)

        result = await self.computer.list_directory(path, include_hidden, recursive)

        if result["success"]:
            files = result["files"]
            output = f"📁 Directory: {result['path']}\n\n"

            if not files:
                return output + "(empty directory)"

            for f in files[:50]:  # Limit to 50 items
                icon = "📁" if f["type"] == "directory" else "📄"
                size = f"({f['size']} bytes)" if f["type"] == "file" else ""
                output += f"{icon} {f['name']} {size}\n"

            if len(files) > 50:
                output += f"\n... and {len(files) - 50} more items"

            return output
        else:
            return f"❌ Failed to list directory: {result['error']}"

    async def _create_directory_tool(self, params: Dict) -> str:
        """Create directory"""
        path = params.get("path", "")

        result = await self.computer.create_directory(path)

        if result["success"]:
            return f"✅ Created directory: {result['path']}"
        else:
            return f"❌ Failed to create directory: {result['error']}"

    async def _delete_file_tool(self, params: Dict) -> str:
        """Delete file or directory"""
        path = params.get("path", "")

        result = await self.computer.delete_file(path)

        if result["success"]:
            return f"✅ Deleted: {result['path']}"
        else:
            return f"❌ Failed to delete: {result['error']}"

    async def _move_file_tool(self, params: Dict) -> str:
        """Move/rename file"""
        source = params.get("source", "")
        destination = params.get("destination", "")

        result = await self.computer.move_file(source, destination)

        if result["success"]:
            return f"✅ Moved: {result['source']} → {result['destination']}"
        else:
            return f"❌ Failed to move: {result['error']}"

    async def _copy_file_tool(self, params: Dict) -> str:
        """Copy file or directory"""
        source = params.get("source", "")
        destination = params.get("destination", "")

        result = await self.computer.copy_file(source, destination)

        if result["success"]:
            return f"✅ Copied: {result['source']} → {result['destination']}"
        else:
            return f"❌ Failed to copy: {result['error']}"

    async def _search_files_tool(self, params: Dict) -> str:
        """Search for files"""
        path = params.get("path", ".")
        pattern = params.get("pattern", "")
        content_search = params.get("content_search", False)

        result = await self.computer.search_files(path, pattern, content_search)

        if result["success"]:
            matches = result["matches"]
            output = f"🔍 Search results for '{pattern}' in {path}\n\n"

            if not matches:
                return output + "No matches found"

            for m in matches[:20]:  # Limit to 20 results
                output += f"• {m['path']}\n"

            if len(matches) > 20:
                output += f"\n... and {len(matches) - 20} more matches"

            return output
        else:
            return f"❌ Search failed: {result['error']}"

    async def _system_info_tool(self, params: Dict) -> str:
        """Get system information"""
        result = await self.computer.get_system_info()

        if result["success"]:
            return f"""💻 System Information

**Platform:** {result['system']} ({result['platform']})
**Python:** {result['python_version']}

**CPU:**
- Cores: {result['cpu_count']}
- Usage: {result['cpu_percent']}%

**Memory:**
- Total: {result['memory_total'] / (1024**3):.2f} GB
- Available: {result['memory_available'] / (1024**3):.2f} GB
- Usage: {result['memory_percent']}%

**Disk:**
- Total: {result['disk_usage']['total'] / (1024**3):.2f} GB
- Used: {result['disk_usage']['used'] / (1024**3):.2f} GB
- Free: {result['disk_usage']['free'] / (1024**3):.2f} GB
- Usage: {result['disk_usage']['percent']}%
"""
        else:
            return f"❌ Failed to get system info: {result['error']}"

    # ========== ORIGINAL TOOLS ==========

    # Built-in tools (simplified implementations)

    async def _email_tool(self, params: Dict) -> str:
        """Email operations via SMTP/IMAP"""
        action = params.get("action", "read")

        if action == "send":
            host = getattr(self.config, "smtp_host", None)
            if not host:
                return (
                    "⚠️ SMTP not configured. Add to .env:\n"
                    "  SMTP_HOST=smtp.gmail.com\n"
                    "  SMTP_USER=you@gmail.com\n"
                    "  SMTP_PASSWORD=your-app-password"
                )

            to = params.get("to", "")
            subject = params.get("subject", "(no subject)")
            body = params.get("body", "")
            if not to:
                return "⚠️ Missing 'to' field,  who should I send the email to?"

            try:
                import smtplib
                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart

                msg = MIMEMultipart()
                msg["From"] = getattr(self.config, "smtp_from", None) or self.config.smtp_user
                msg["To"] = to
                msg["Subject"] = subject
                msg.attach(MIMEText(body, "plain"))

                port = int(getattr(self.config, "smtp_port", 587))
                with smtplib.SMTP(host, port) as server:
                    server.starttls()
                    server.login(self.config.smtp_user, self.config.smtp_password)
                    server.send_message(msg)

                logger.info(f"📧 Email sent to {to}: {subject}")
                return f"✅ Email sent to **{to}**\nSubject: {subject}"
            except Exception as e:
                logger.error(f"Email send failed: {e}")
                return f"❌ Failed to send email: {e}"

        elif action == "read":
            host = getattr(self.config, "imap_host", None)
            if not host:
                return (
                    "⚠️ IMAP not configured. Add to .env:\n"
                    "  IMAP_HOST=imap.gmail.com\n"
                    "  IMAP_USER=you@gmail.com\n"
                    "  IMAP_PASSWORD=your-app-password"
                )

            count = int(params.get("count", 5))
            folder = params.get("folder", "INBOX")

            try:
                import imaplib
                import email as email_lib
                from email.header import decode_header

                port = int(getattr(self.config, "imap_port", 993))
                with imaplib.IMAP4_SSL(host, port) as imap:
                    imap.login(
                        getattr(self.config, "imap_user", None) or self.config.smtp_user,
                        getattr(self.config, "imap_password", None) or self.config.smtp_password,
                    )
                    imap.select(folder, readonly=True)
                    _, data = imap.search(None, "ALL")
                    ids = data[0].split()
                    if not ids:
                        return f"📧 No emails in {folder}."

                    latest = ids[-count:]
                    latest.reverse()
                    results = []
                    for mid in latest:
                        _, msg_data = imap.fetch(mid, "(RFC822)")
                        raw = msg_data[0][1]
                        msg = email_lib.message_from_bytes(raw)
                        subj = ""
                        for part, enc in decode_header(msg["Subject"] or ""):
                            subj += (
                                part.decode(enc or "utf-8")
                                if isinstance(part, bytes)
                                else str(part)
                            )
                        frm = msg["From"] or ""
                        date = msg["Date"] or ""
                        results.append(f"• **{subj}**\n  From: {frm}\n  Date: {date}")

                return f"📧 **Latest {len(results)} emails ({folder}):**\n\n" + "\n\n".join(results)
            except Exception as e:
                logger.error(f"Email read failed: {e}")
                return f"❌ Failed to read email: {e}"

        else:
            return f"Unknown email action: {action}. Use: send, read"

    async def _calendar_tool(self, params: Dict) -> str:
        """Internal calendar operations (stored locally)"""
        action = params.get("action", "list")

        try:
            # Load calendar events
            events = json.loads(self.calendar_file.read_text())

            if action == "list":
                # Show upcoming events
                now = datetime.now()
                upcoming = [e for e in events if datetime.fromisoformat(e["datetime"]) >= now]
                upcoming.sort(key=lambda x: x["datetime"])

                if not upcoming:
                    return "📅 No upcoming events in your calendar."

                result = "📅 **Upcoming Events:**\n\n"
                for event in upcoming[:10]:  # Show next 10
                    dt = datetime.fromisoformat(event["datetime"])
                    result += f"• **{event['title']}**\n"
                    result += f"  📆 {dt.strftime('%Y-%m-%d %H:%M')}\n"
                    if event.get("description"):
                        result += f"  📝 {event['description']}\n"
                    result += "\n"
                return result.strip()

            elif action == "add":
                title = params.get("title", "Untitled Event")
                date_str = params.get("date", "")
                description = params.get("description", "")

                if not date_str:
                    return "⚠️ Please provide a date/time (e.g., '2026-02-20 15:00' or 'tomorrow at 3pm')"

                # Parse date (simple ISO format support)
                try:
                    # Try ISO format first
                    event_dt = datetime.fromisoformat(date_str)
                except ValueError:
                    # Try common formats
                    for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y"]:
                        try:
                            event_dt = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        return f"⚠️ Could not parse date '{date_str}'. Use format: YYYY-MM-DD HH:MM"

                # Add event
                new_event = {
                    "id": len(events) + 1,
                    "title": title,
                    "datetime": event_dt.isoformat(),
                    "description": description,
                    "created_at": datetime.now().isoformat(),
                }
                events.append(new_event)

                # Save
                self.calendar_file.write_text(json.dumps(events, indent=2))

                return f"✅ Event added: **{title}** on {event_dt.strftime('%Y-%m-%d %H:%M')}"

            elif action == "delete":
                event_id = params.get("id")
                if not event_id:
                    return "⚠️ Please provide event ID to delete"

                events = [e for e in events if e["id"] != int(event_id)]
                self.calendar_file.write_text(json.dumps(events, indent=2))
                return f"✅ Event {event_id} deleted"

            else:
                return f"Unknown calendar action: {action}. Use: list, add, delete"

        except Exception as e:
            logger.error(f"Calendar error: {e}")
            return f"⚠️ Calendar error: {str(e)}"

    async def _browser_tool(self, params: Dict) -> str:
        """Browser automation and web scraping using Playwright"""
        action = params.get("action", "scrape")

        if action == "snapshot":
            url = params.get("url", "")
            format_type = params.get("format", "aria")

            result = await self.browser_engine.snapshot(url, format_type)
            if result.get("success"):
                refs_text = f"📸 Snapshot of {result.get('url')}\n"
                refs_text += f"Found {result.get('count', 0)} interactive elements:\n\n"

                for ref_data in result.get("refs", [])[:20]:  # Limit to first 20
                    ref_text = f"{ref_data['ref']}: {ref_data['role']}"
                    if ref_data.get("name"):
                        ref_text += f" '{ref_data['name']}'"
                    refs_text += ref_text + "\n"

                if result.get("count", 0) > 20:
                    refs_text += f"\n... and {result.get('count') - 20} more elements"

                return refs_text
            else:
                return f"❌ Snapshot failed: {result.get('error', 'Unknown error')}"

        elif action == "scrape":
            url = params.get("url")
            result = await self.browser_engine.scrape_page(url)

            if not result.get("success"):
                return f"⚠️ {result.get('error', 'Unknown error')}"

            return f"🌐 **{result['title']}**\n\nURL: {result['url']}\n\n{result['content']}"

        elif action == "search":
            query = params.get("query")
            if not query:
                return "⚠️ Please provide a search query"

            logger.info(f"🔍 Searching web for: '{query}'")
            num_results = params.get("num_results", 5)
            result = await self.browser_engine.search_web(query, num_results)

            logger.info(
                f"Search returned: success={result.get('success')}, count={result.get('count', 0)}"
            )

            if not result.get("success"):
                return f"⚠️ {result.get('error', 'Unknown error')}"

            # Check if we got results
            results_list = result.get("results", [])
            if not results_list or len(results_list) == 0:
                logger.warning(f"No search results found for '{query}'")
                return f"🔍 No search results found for '{query}'. The search engine returned 0 results."

            # Format results
            response = f"🔍 **Search Results for: {query}**\n\n"
            for i, res in enumerate(results_list, 1):
                response += f"**{i}. {res.get('title', 'Untitled')}**\n"
                response += f"{res.get('snippet', 'No description')}\n"
                response += f"🔗 {res.get('url', '')}\n\n"

            return response.strip()

        elif action == "screenshot":
            url = params.get("url")
            if not url:
                return "⚠️ Please provide a URL for screenshot"

            result = await self.browser_engine.get_page_screenshot(url)

            if not result.get("success"):
                return f"⚠️ {result.get('error', 'Unknown error')}"

            path = result["path"]
            msg = f"📸 Screenshot saved: {path}"

            # Auto-cleanup: delete temp screenshot after reading
            if result.get("auto_cleanup"):
                try:
                    import base64, os
                    with open(path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode()
                    os.remove(path)
                    msg = f"📸 Screenshot taken ({len(img_b64)//1024}KB base64). File cleaned up."
                except Exception:
                    pass  # file already gone or unreadable — fine

            return msg

        else:
            return f"Unknown browser action: {action}. Available: scrape, search, screenshot"

    async def _web_action_tool(self, params: Dict) -> str:
        """Execute interactive web actions using refs or selectors

        Actions: click, type, hover, drag, select, fill, press, evaluate, wait, submit
        Use refs from snapshot for stable automation
        """
        url = params.get("url")
        action = params.get("action", "")
        ref = params.get("ref")
        selector = params.get("selector")
        value = params.get("value")

        if not action:
            return "⚠️ Missing action parameter"

        try:
            result = await self.browser_engine.execute_action(
                url=url, action=action, ref=ref, selector=selector, value=value
            )

            if result.get("success"):
                action_name = result.get("action", action).capitalize()
                details = ""

                if "value" in result:
                    details = f": '{result['value']}'"
                elif "key" in result:
                    details = f": {result['key']}"
                elif "result" in result:
                    details = f" -> {result['result']}"

                return f"✅ {action_name}{details}"
            else:
                return f"❌ {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    async def _file_tool(self, params: Dict) -> str:
        """
        DEPRECATED: Use specific file tools instead
        (read_file, write_file, edit_file, list_directory, etc.)
        """
        action = params.get("action", "list")

        if action == "list":
            return await self._list_directory_tool({"path": params.get("path", ".")})
        elif action == "read":
            return await self._read_file_tool({"path": params.get("path", "")})
        else:
            return "⚠️ Use specific file tools: read_file, write_file, edit_file, list_directory"

    async def _weather_tool(self, params: Dict) -> str:
        """Weather information using wttr.in (no API key required)"""
        location = params.get("location", "")

        if not location:
            # Use IP-based auto-detection
            location = ""

        try:
            # Call wttr.in API (free, no API key needed)
            async with aiohttp.ClientSession() as session:
                # Format: ?format=j1 for JSON, ?m for metric
                url = f"https://wttr.in/{location}?format=j1&m"
                headers = {"User-Agent": "curl/7.68.0"}  # wttr.in prefers curl user agent

                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()

                        # Parse weather data
                        current = data["current_condition"][0]
                        location_info = data["nearest_area"][0]

                        temp = current["temp_C"]
                        feels_like = current["FeelsLikeC"]
                        humidity = current["humidity"]
                        description = current["weatherDesc"][0]["value"]
                        wind_speed = current["windspeedKmph"]
                        city_name = location_info.get("areaName", [{}])[0].get(
                            "value", location or "Your location"
                        )

                        # Weather emoji mapping
                        weather_code = int(current["weatherCode"])
                        weather_emojis = {
                            113: "☀️",  # Clear/Sunny
                            116: "⛅",  # Partly cloudy
                            119: "☁️",  # Cloudy
                            122: "☁️",  # Overcast
                            143: "🌫️",  # Mist
                            176: "🌦️",  # Patchy rain possible
                            200: "⛈️",  # Thundery outbreaks possible
                            248: "🌫️",  # Fog
                            263: "🌧️",  # Patchy light drizzle
                            266: "🌧️",  # Light drizzle
                            281: "🌧️",  # Freezing drizzle
                            284: "🌧️",  # Heavy freezing drizzle
                            293: "🌧️",  # Patchy light rain
                            296: "🌧️",  # Light rain
                            299: "🌧️",  # Moderate rain at times
                            302: "🌧️",  # Moderate rain
                            305: "🌧️",  # Heavy rain at times
                            308: "🌧️",  # Heavy rain
                            311: "🌧️",  # Light freezing rain
                            314: "🌧️",  # Moderate or heavy freezing rain
                            317: "🌨️",  # Light sleet
                            320: "🌨️",  # Moderate or heavy sleet
                            323: "❄️",  # Patchy light snow
                            326: "❄️",  # Light snow
                            329: "❄️",  # Patchy moderate snow
                            332: "❄️",  # Moderate snow
                            335: "❄️",  # Patchy heavy snow
                            338: "❄️",  # Heavy snow
                            350: "🌨️",  # Ice pellets
                            353: "🌧️",  # Light rain shower
                            356: "🌧️",  # Moderate or heavy rain shower
                            359: "🌧️",  # Torrential rain shower
                            362: "🌨️",  # Light sleet showers
                            365: "🌨️",  # Moderate or heavy sleet showers
                            368: "❄️",  # Light snow showers
                            371: "❄️",  # Moderate or heavy snow showers
                            374: "🌨️",  # Light showers of ice pellets
                            377: "🌨️",  # Moderate or heavy showers of ice pellets
                            386: "⛈️",  # Patchy light rain with thunder
                            389: "⛈️",  # Moderate or heavy rain with thunder
                            392: "⛈️",  # Patchy light snow with thunder
                            395: "⛈️",  # Moderate or heavy snow with thunder
                        }

                        emoji = weather_emojis.get(weather_code, "🌤️")

                        return (
                            f"{emoji} **Weather in {city_name}**\n"
                            f"🌡️ Temperature: {temp}°C (feels like {feels_like}°C)\n"
                            f"💧 Humidity: {humidity}%\n"
                            f"💨 Wind: {wind_speed} km/h\n"
                            f"📝 Conditions: {description}"
                        )
                    else:
                        return f"⚠️ Weather service error: {resp.status}"
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return f"⚠️ Failed to fetch weather data: {str(e)}"

    # ========== VOICE TOOLS ==========

    async def _voice_speak_tool(self, params: Dict) -> str:
        """Text-to-speech conversion"""
        text = params.get("text", "")
        output_file = params.get("output_file")

        try:
            audio_path = await self.voice.speak(text, output_file=output_file)
            return f"🔊 Audio generated: {audio_path}"
        except Exception as e:
            return f"❌ TTS failed: {str(e)}"

    async def _voice_listen_tool(self, params: Dict) -> str:
        """Speech-to-text conversion"""
        audio_file = params.get("audio_file")

        try:
            text = await self.voice.listen(audio_file=audio_file)
            return f"📝 Transcription:\n{text}"
        except Exception as e:
            return f"❌ STT failed: {str(e)}"

    # ========== IMAGE TOOLS ==========

    async def _generate_image_tool(self, params: Dict) -> str:
        """Generate images from text prompts"""
        prompt = params.get("prompt", "")
        model = params.get("model", "dall-e-3")
        size = params.get("size", "1024x1024")
        output_path = params.get("output_path", "generated_image.png")

        try:
            result = await self.image.generate(
                prompt=prompt, model=model, size=size, output_path=output_path
            )

            if result.get("success"):
                return f"🎨 Image generated: {result.get('path')}\nPrompt: {prompt}"
            else:
                return f"❌ Generation failed: {result.get('error')}"
        except Exception as e:
            return f"❌ Image generation error: {str(e)}"

    async def _analyze_image_tool(self, params: Dict) -> str:
        """Analyze image content"""
        image_path = params.get("image_path", "")

        try:
            result = await self.image.analyze(image_path)

            if result.get("success"):
                labels = ", ".join(result.get("labels", []))
                description = result.get("description", "No description")

                return f"🔍 Image Analysis:\n{description}\n\nDetected: {labels}"
            else:
                return f"❌ Analysis failed: {result.get('error')}"
        except Exception as e:
            return f"❌ Image analysis error: {str(e)}"

    async def _ocr_tool(self, params: Dict) -> str:
        """Extract text from images"""
        image_path = params.get("image_path", "")
        language = params.get("language", "eng")

        try:
            result = await self.image.ocr(image_path, language=language)

            if result.get("success"):
                text = result.get("text", "")
                confidence = result.get("confidence", 0)

                return f"📄 OCR Results (confidence: {confidence}%):\n\n{text}"
            else:
                return f"❌ OCR failed: {result.get('error')}"
        except Exception as e:
            return f"❌ OCR error: {str(e)}"

    # ========== DATABASE TOOLS ==========

    async def _database_query_tool(self, params: Dict) -> str:
        """Execute database queries"""
        query = params.get("query", "")
        db_type = params.get("db_type", "sqlite")
        database = params.get("database", "default.db")

        try:
            result = await self.database.execute(query=query, db_type=db_type, database=database)

            if result.get("success"):
                rows = result.get("rows", [])
                row_count = len(rows)

                return f"✅ Query executed successfully\nRows returned: {row_count}\n\n{json.dumps(rows[:10], indent=2)}"
            else:
                return f"❌ Query failed: {result.get('error')}"
        except Exception as e:
            return f"❌ Database error: {str(e)}"

    # ========== RAG TOOLS ==========

    async def _vector_search_tool(self, params: Dict) -> str:
        """Semantic search using vector database, falls back to web search if unavailable"""
        query = params.get("query", "")
        collection = params.get("collection", "default")
        top_k = int(params.get("top_k", 5))  # ensure int, not string

        try:
            results = await self.rag.search(query=query, collection=collection, top_k=top_k)

            if results:
                formatted = "\n\n".join(
                    [
                        f"**Result {i+1}** (score: {r.get('score', 0):.2f}):\n{r.get('content', '')}"
                        for i, r in enumerate(results)
                    ]
                )
                return f"🔍 Found {len(results)} results:\n\n{formatted}"
            else:
                # Local knowledge base is empty,  fall back to web search
                logger.info(f"Vector DB empty for '{query}', falling back to browser_search")
                return await self._browser_tool(
                    {"action": "search", "query": query, "num_results": int(top_k)}
                )
        except Exception as e:
            # Embedding model not available,  fall back to web search
            logger.warning(f"Vector search unavailable ({e}), falling back to browser_search")
            return await self._browser_tool({"action": "search", "query": query, "num_results": 5})

    # ========== CODE EXECUTION TOOLS ==========

    async def _execute_code_tool(self, params: Dict) -> str:
        """Execute code in sandbox"""
        code = params.get("code", "")
        language = params.get("language", "python")
        timeout = params.get("timeout", 30)

        try:
            result = await self.code_executor.execute(code=code, language=language, timeout=timeout)

            if result.get("success"):
                output = result.get("output", "")
                return f"✅ Code executed successfully:\n\n```\n{output}\n```"
            else:
                error = result.get("error", "Unknown error")
                return f"❌ Execution failed:\n{error}"
        except Exception as e:
            return f"❌ Code execution error: {str(e)}"

    # ========== API CLIENT TOOLS ==========

    async def _api_request_tool(self, params: Dict) -> str:
        """Make HTTP API requests"""
        url = params.get("url", "")
        method = params.get("method", "GET")
        headers = params.get("headers", {})
        data = params.get("data")

        try:
            result = await self.api_client.request(
                url=url, method=method, headers=headers, data=data
            )

            if result.get("success"):
                response_data = result.get("data", "")
                status_code = result.get("status_code", 200)

                return f"✅ API request successful (status: {status_code}):\n\n{json.dumps(response_data, indent=2)[:500]}"
            else:
                return f"❌ API request failed: {result.get('error')}"
        except Exception as e:
            return f"❌ API request error: {str(e)}"

    # ========== SKILL CREATION TOOLS ==========

    async def _create_skill_tool(self, params: Dict) -> str:
        """Create a new dynamic skill with full auto-wiring into the tool registry."""
        name = params.get("name", "")
        description = params.get("description", "")
        code = params.get("code", "")
        author = params.get("author", "sable")

        from datetime import datetime, timezone
        metadata = {"author": author, "created_at": datetime.now(timezone.utc).isoformat()}

        try:
            result = await self.skill_creator.create_skill(name, description, code, metadata)

            if result.get("success"):
                # Auto-wire the skill's tools into the live registry
                module = result.get("module")
                tool_info = result.get("tool_info", {})
                registered_tools = []

                if module and tool_info:
                    registered_tools = await self.register_dynamic_skill(
                        name, module, tool_info
                    )

                tool_list = ", ".join(registered_tools) if registered_tools else "none"
                return (
                    f"✅ Skill '{name}' created and auto-wired!\n\n"
                    f"Path: {result.get('path')}\n"
                    f"Tools registered: {tool_list}\n\n"
                    f"These tools are now live,  you can call them immediately."
                )
            else:
                return f"❌ Failed to create skill: {result.get('error')}"
        except Exception as e:
            return f"❌ Skill creation error: {str(e)}"

    async def _list_skills_tool(self, params: Dict) -> str:
        """List all custom skills with their registered tools"""
        try:
            skills = await self.skill_creator.list_skills()

            if not skills:
                return (
                    "📦 No custom skills created yet.\n\n"
                    "Use create_skill to add new functionality! The skill code "
                    "should follow the Dynamic Skill Protocol:\n"
                    "  - Define TOOL_SCHEMAS (list of OpenAI schema dicts)\n"
                    "  - Define handle_<name>(params) async functions\n"
                    "  - Optionally define TOOL_PERMISSIONS and initialize()"
                )

            formatted = "\n".join(
                [
                    f"• **{s['name']}**,  {s['description']}\n"
                    f"  Status: {'✅ Active' if s.get('active', True) else '❌ Disabled'}\n"
                    f"  Tools: {', '.join(s.get('tool_names', [])) or 'none'}\n"
                    f"  Author: {s.get('metadata', {}).get('author', 'unknown')}"
                    for s in skills
                ]
            )

            return f"📦 Custom Skills ({len(skills)}):\n\n{formatted}"
        except Exception as e:
            return f"❌ Error listing skills: {str(e)}"

    async def _delete_skill_tool(self, params: Dict) -> str:
        """Delete a dynamic skill and unregister its tools."""
        name = params.get("name", "")
        if not name:
            return "❌ Please provide the skill name to delete."

        try:
            result = self.skill_creator.delete_skill(name)
            if result.get("success"):
                tool_names = result.get("tool_names", [])
                self.unregister_dynamic_skill(tool_names)
                return (
                    f"🗑️  Skill '{name}' deleted.\n"
                    f"Unregistered tools: {', '.join(tool_names) or 'none'}"
                )
            else:
                return f"❌ {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"❌ Error deleting skill: {str(e)}"

    async def _disable_skill_tool(self, params: Dict) -> str:
        """Disable a dynamic skill (keeps it on disk but unregisters tools)."""
        name = params.get("name", "")
        if not name:
            return "❌ Please provide the skill name to disable."

        try:
            if self.skill_creator.disable_skill(name):
                # Unregister its tools from the live registry
                entry = self.skill_creator.registry.get(name, {})
                tool_names = entry.get("tool_names", [])
                self.unregister_dynamic_skill(tool_names)
                return (
                    f"⏸️  Skill '{name}' disabled.\n"
                    f"Tools unregistered: {', '.join(tool_names) or 'none'}\n"
                    f"Use enable_skill to re-activate it."
                )
            else:
                return f"❌ Skill '{name}' not found."
        except Exception as e:
            return f"❌ Error disabling skill: {str(e)}"

    async def _enable_skill_tool(self, params: Dict) -> str:
        """Re-enable a disabled dynamic skill and re-wire its tools."""
        name = params.get("name", "")
        if not name:
            return "❌ Please provide the skill name to enable."

        try:
            if self.skill_creator.enable_skill(name):
                # Reload and re-wire the skill
                entry = self.skill_creator.registry.get(name, {})
                skill_file = entry.get("file", "")
                if skill_file:
                    from pathlib import Path
                    module = self.skill_creator._load_module(name, Path(skill_file))
                    self.skill_creator._loaded_modules[name] = module
                    tool_info = self.skill_creator._extract_tool_info(module, name)
                    registered = await self.register_dynamic_skill(
                        name, module, tool_info
                    )
                    return (
                        f"▶️  Skill '{name}' re-enabled!\n"
                        f"Tools registered: {', '.join(registered) or 'none'}"
                    )
                return f"▶️  Skill '{name}' enabled (no tools to wire)."
            else:
                return f"❌ Skill '{name}' not found."
        except Exception as e:
            return f"❌ Error enabling skill: {str(e)}"

    # ── Meta-tool: lazy schema loading ────────────────────────────────────

    async def _load_tool_details(self, params: Dict) -> str:
        """Return the full parameter schemas for requested tools.

        When the agent's tool list is sent with compact schemas (name +
        description only), the model can call this meta-tool to fetch the
        complete parameter definitions before invoking a complex tool.
        """
        import json as _json
        from ._schemas import get_all_schemas

        requested = params.get("tool_names", [])
        if isinstance(requested, str):
            requested = [requested]

        all_schemas = get_all_schemas()
        schema_map = {
            s.get("function", {}).get("name", ""): s
            for s in all_schemas
        }

        results = []
        for name in requested:
            if name in schema_map:
                fn = schema_map[name]["function"]
                results.append(
                    f"**{name}**:\n```json\n"
                    f"{_json.dumps(fn.get('parameters', {}), indent=2)}\n```"
                )
            else:
                results.append(f"**{name}**: ⚠️ Unknown tool")

        if not results:
            return "⚠️ No tool names provided. Pass tool_names=['tool1', 'tool2']."

        return (
            "Here are the full parameter schemas for the requested tools. "
            "You can now call them with the correct arguments:\n\n"
            + "\n\n".join(results)
        )

