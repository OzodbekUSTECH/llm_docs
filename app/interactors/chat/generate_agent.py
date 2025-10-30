from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.dto.schemas import DocumentSchema
from app.agents.tools import AgentTools
from agents import Agent, Runner, function_tool, handoff, SQLiteSession
from agents.stream_events import RunItemStreamEvent, RawResponsesStreamEvent
from openai import AsyncOpenAI
from app.core.config import settings

class GenerateAgentAnswerInteractor:
    """Multi-agent style interactor using tool-calling loop.

    Exposes tools: search_documents, get_document, get_file_chunks.
    The LLM plans, calls tools iteratively until it has enough info, then answers.
    """

    def __init__(self) -> None:
        self.tools = AgentTools()
        self.model = "gpt-4o-mini"
        self.agent: Agent | None = None
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._conv_cache: dict[str, str] = {}
        self._sessions: dict[str, SQLiteSession] = {}

    def _ensure_agent(self, instructions: str) -> Agent:
        if self.agent is None:
            tools_ref = self.tools

            @function_tool(name_override="search_documents", description_override="Search documents by query and return list of chunks {doc_id, content, score, metadata}")
            async def search_documents(query: str, limit: int = 20, use_vector: bool = True, use_keyword: bool = True, file_ids: Optional[List[str]] = None) -> dict:
                return await tools_ref.search_documents(query=query, limit=limit, use_vector=use_vector, use_keyword=use_keyword, file_ids=file_ids)

            @function_tool(name_override="get_document", description_override="Fetch a single chunk by doc_id and return {item}")
            async def get_document(doc_id: str) -> dict:
                return tools_ref.get_document(doc_id)

            self.agent = Agent(
                model=self.model,
                name="DocSearchAgent",
                instructions=instructions,
                tools=[search_documents, get_document],
            )
        return self.agent

    async def stream(
        self,
        message: str,
        conv_id: str,
        history: Optional[List[Dict]] = None,
        **kwargs,
    ) -> AsyncGenerator[DocumentSchema, None]:
        sys_prompt = (
            "You are a diligent research agent with access to one tool: search_documents(query, limit, use_vector, use_keyword, file_ids)."
            " Work in steps: 1) outline a brief plan, 2) search with a focused query, 3) analyze whether all information needed to answer the user's request is present,"
            " 4) if not complete, generate 1–3 rephrased, more targeted queries (e.g., include any identifiers, entities, numbers, or domain tokens you observed) and search again,"
            " 5) repeat until reasonably complete, then answer with citations."
            " Be domain-agnostic: infer the set of attributes the user expects (from the user’s wording) and verify those are filled."
            " Match the user's requested format and be concise (no filler, no repeated tables, no long prefaces)."
            " Output ONLY the answer, nothing else."
        )

        agent = self._ensure_agent(sys_prompt)

        # Generic, single-agent loop without hardcoded subagents

        # Prepare condensed history as strings (if SDK supports passing history directly, adjust here)
        conv_history: List[Dict[str, str]] = []
        for m in (history or [])[-5:]:
            if isinstance(m, dict) and "role" in m and "content" in m:
                conv_history.append({"role": m["role"], "content": str(m["content"])})

        yield DocumentSchema(content="🧭 Planning and searching...", channel="debug")

        # Start streamed run (SDK expects a single input; inline recent history)
        history_prefix_parts: List[str] = []
        for m in conv_history:
            role = m.get("role", "user")
            content = m.get("content", "")
            history_prefix_parts.append(f"{role.capitalize()}: {content}")
        history_prefix = "\n".join(history_prefix_parts)
        combined_input = (history_prefix + "\n" if history_prefix else "") + f"User: {message}"

        # Use SDK sessions to persist conversation per chat
        sess = self._sessions.get(conv_id)
        if not sess:
            sess = SQLiteSession(conv_id)
            self._sessions[conv_id] = sess
        run = Runner.run_streamed(agent, input=combined_input, session=sess)
        tool_calls = 0
        seen_calls: set[str] = set()
        async for event in run.stream_events():
            et = getattr(event, "type", None)
            if et == 'run_item_stream_event':
                rie: RunItemStreamEvent = event  # type: ignore
                name = getattr(rie, 'name', '')
                if name == 'reasoning_item_created':
                    thought = getattr(getattr(rie, 'item', None), 'raw_item', None)
                    thought_text = getattr(thought, 'thought', None)
                    if thought_text:
                        trace = {"trace_type": "plan", "content": str(thought_text)}
                        yield DocumentSchema(content=json.dumps(trace), channel="debug")
                elif name == 'tool_called':
                    tool_item = getattr(rie, 'item', None)
                    raw = getattr(tool_item, 'raw_item', None)
                    tname = getattr(raw, 'name', '')
                    args = getattr(raw, 'arguments', '')
                    call_sig = f"{tname}:{args}"
                    if call_sig in seen_calls:
                        # Deduplicate repeated identical calls to reduce token and rate use
                        continue
                    seen_calls.add(call_sig)
                    msg = {"trace_type": "tool_call", "tool": tname, "args": args}
                    tool_calls += 1
                    yield DocumentSchema(content=json.dumps(msg), channel="debug")
                elif name == 'tool_output':
                    out_item = getattr(rie, 'item', None)
                    output = getattr(out_item, 'output', None)
                    if output is not None:
                        summary = output
                        if isinstance(output, dict) and 'items' in output:
                            items = output.get('items') or []
                            summary = {"count": len(items), "preview": items[:3]}
                        trace = {"trace_type": "tool_result", "result": summary}
                        yield DocumentSchema(content=json.dumps(trace), channel="debug")
                elif name == 'message_output_created':
                    msg_item = getattr(rie, 'item', None)
                    raw = getattr(msg_item, 'raw_item', None)
                    content = ''
                    if raw and getattr(raw, 'content', None):
                        cont = raw.content[0]
                        content = getattr(cont, 'text', '')
                    if content:
                        yield DocumentSchema(content=json.dumps({"trace_type": "decision", "content": content}), channel="debug")
            elif et == 'raw_response_event':
                rre: RawResponsesStreamEvent = event  # type: ignore
                # could log token usage here if needed
                pass

        # Determine need for follow-up before emitting final
        final_text = getattr(run, 'final_output', None)
        need_followup = False
        txt = (str(final_text or "")).lower()
        gap_markers = ["not specified", "not explicitly stated", "unknown", "n/a"]
        if tool_calls < 2:
            need_followup = True
        if any(g in txt for g in gap_markers):
            need_followup = True

        if final_text and not need_followup:
            yield DocumentSchema(content=self._constrain_output(str(final_text), message), channel="chat")
            yield DocumentSchema(content=json.dumps({"trace_type": "finalize"}), channel="debug")
            return

        # Gap-triggered follow-up if too few tool calls or gaps present
        if need_followup and tool_calls < 6:
            yield DocumentSchema(content=json.dumps({"trace_type": "retry", "reason": "gaps_detected"}), channel="debug")
            follow_input = combined_input + "\n\nAssistant: Continue searching to fill any missing attributes. Generate 2-3 targeted queries and call search_documents again until gaps are resolved, then finalize."
            run2 = Runner.run_streamed(agent, input=follow_input, session=sess)
            async for event in run2.stream_events():
                et = getattr(event, "type", None)
                if et == 'run_item_stream_event':
                    rie: RunItemStreamEvent = event  # type: ignore
                    name = getattr(rie, 'name', '')
                    if name == 'reasoning_item_created':
                        thought = getattr(getattr(rie, 'item', None), 'raw_item', None)
                        thought_text = getattr(thought, 'thought', None)
                        if thought_text:
                            yield DocumentSchema(content=json.dumps({"trace_type": "plan", "content": str(thought_text)}), channel="debug")
                    elif name == 'tool_called':
                        tool_item = getattr(rie, 'item', None)
                        raw = getattr(tool_item, 'raw_item', None)
                        tname = getattr(raw, 'name', '')
                        args = getattr(raw, 'arguments', '')
                        yield DocumentSchema(content=json.dumps({"trace_type": "tool_call", "tool": tname, "args": args}), channel="debug")
                    elif name == 'tool_output':
                        out_item = getattr(rie, 'item', None)
                        output = getattr(out_item, 'output', None)
                        if output is not None:
                            summary = output
                            if isinstance(output, dict) and 'items' in output:
                                items = output.get('items') or []
                                summary = {"count": len(items), "preview": items[:3]}
                            yield DocumentSchema(content=json.dumps({"trace_type": "tool_result", "result": summary}), channel="debug")
                    elif name == 'message_output_created':
                        msg_item = getattr(rie, 'item', None)
                        raw = getattr(msg_item, 'raw_item', None)
                        content = ''
                        if raw and getattr(raw, 'content', None):
                            cont = raw.content[0]
                            content = getattr(cont, 'text', '')
                        if content:
                            yield DocumentSchema(content=json.dumps({"trace_type": "decision", "content": content}), channel="debug")
            final2 = getattr(run2, 'final_output', None)
            if final2:
                yield DocumentSchema(content=self._constrain_output(str(final2), message), channel="chat")
                yield DocumentSchema(content=json.dumps({"trace_type": "finalize"}), channel="debug")

        # If still gaps, enumerate missing cells and force targeted search per row
        final_all = getattr(run, 'final_output', '')
        final_all = str(final_all or '')
        if any(m in final_all.lower() for m in gap_markers) and tool_calls < 8:
            # Try to extract contract identifiers/names for targeted completion
            import re as _re
            ids = _re.findall(r"\b([A-Z]{1,4}-\d{2,4}-?[A-Z]{0,3})\b", final_all)
            rows = []
            for line in final_all.splitlines():
                if 'not specified' in line.lower():
                    rows.append(line.strip())
            if ids or rows:
                yield DocumentSchema(content=json.dumps({"trace_type": "retry", "reason": "row_gaps", "ids": ids[:5]}), channel="debug")
                hints = "; ".join(ids[:5]) if ids else "; ".join(rows[:3])
                force_input = combined_input + f"\n\nAssistant: For these rows/ids [{hints}], run focused searches (2-3 attempts each) to find the missing values, then output an updated complete table with citations."
                run3 = Runner.run_streamed(agent, input=force_input, session=sess)
                async for event in run3.stream_events():
                    et = getattr(event, "type", None)
                    if et == 'run_item_stream_event':
                        rie: RunItemStreamEvent = event  # type: ignore
                        name = getattr(rie, 'name', '')
                        if name == 'tool_called':
                            raw = getattr(getattr(rie, 'item', None), 'raw_item', None)
                            tname = getattr(raw, 'name', '')
                            args = getattr(raw, 'arguments', '')
                            yield DocumentSchema(content=json.dumps({"trace_type": "tool_call", "tool": tname, "args": args}), channel="debug")
                        elif name == 'tool_output':
                            out_item = getattr(rie, 'item', None)
                            output = getattr(out_item, 'output', None)
                            if output is not None:
                                summary = output
                                if isinstance(output, dict) and 'items' in output:
                                    items = output.get('items') or []
                                    summary = {"count": len(items), "preview": items[:3]}
                                yield DocumentSchema(content=json.dumps({"trace_type": "tool_result", "result": summary}), channel="debug")
                final3 = getattr(run3, 'final_output', None)
                if final3:
                    yield DocumentSchema(content=self._constrain_output(str(final3), message), channel="chat")
                    yield DocumentSchema(content=json.dumps({"trace_type": "finalize"}), channel="debug")

    def _constrain_output(self, text: str, user_prompt: str) -> str:
        """Keep the response tight and aligned to the user prompt.
        - Remove filler sections like remarks/insights if present
        - Trim duplicate trailing tables/text
        - Cap length to a reasonable size
        """
        if not text:
            return text
        lines = [ln for ln in text.splitlines()]
        # Drop filler headers
        drop_prefixes = ("remarks", "insights", "additional notes", "sources:")
        cleaned = []
        for ln in lines:
            low = ln.strip().lower()
            if any(low.startswith(p) for p in drop_prefixes):
                continue
            cleaned.append(ln)
        out = "\n".join(cleaned).strip()
        # Remove duplicate consecutive blocks
        seen = set()
        uniq = []
        for ln in out.splitlines():
            key = ln.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(ln)
        out = "\n".join(uniq).strip()
        # Cap characters to avoid verbosity
        cap = 2000
        if len(out) > cap:
            out = out[:cap].rstrip()
        return out



