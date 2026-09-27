"""List the tools exposed by FMP's official MCP server (listing does not call any data endpoint).

Usage: python scripts/probe_fmp_mcp.py [tool_name '{"json": "args"}']
With a tool name, also calls that tool once (1 FMP request) and prints the first 1500 chars.
"""

from __future__ import annotations

import asyncio
import json
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

from jevbt.config import load_settings


async def main() -> int:
    s = load_settings()
    client = MultiServerMCPClient({"fmp": {
        "transport": "streamable_http",
        "url": f"https://financialmodelingprep.com/mcp?apikey={s.fmp_api_key}",
    }})
    try:
        tools = await client.get_tools()
    except Exception as e:  # noqa: BLE001 - report any connection problem without the key
        print(type(e).__name__, str(e).replace(s.fmp_api_key, "***")[:500])
        return 1
    print(f"{len(tools)} tools")
    for t in tools:
        args = list((t.args_schema or {}).get("properties", {})) if isinstance(t.args_schema, dict) else []
        print(f"- {t.name}({', '.join(args)}): {(t.description or '').splitlines()[0][:110] if t.description else ''}")
    if len(sys.argv) > 1:
        tool = next(t for t in tools if t.name == sys.argv[1])
        out = await tool.ainvoke(json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
        print("\n==", tool.name, "->", str(out).replace(s.fmp_api_key, "***")[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
