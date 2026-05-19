from __future__ import annotations

from aiohttp import web

try:
    from server import PromptServer
except Exception:
    PromptServer = None

from .layer13_persistent_counter import Layer13PersistentCounter


async def _extract_counter_name(request) -> str:
    name = request.rel_url.query.get("name", "").strip()
    if name:
        return name

    if request.can_read_body:
        try:
            data = await request.json()
        except Exception:
            data = {}
        if isinstance(data, dict):
            return str(data.get("name", "")).strip() or "default"

    return "default"


if PromptServer is not None:
    @PromptServer.instance.routes.get("/layer13/persistent_counter/reset")
    async def layer13_persistent_counter_reset_get(request):
        counter_name, existed = Layer13PersistentCounter.reset_counter(
            await _extract_counter_name(request)
        )
        return web.json_response(
            {
                "ok": True,
                "name": counter_name,
                "reset": True,
                "existed": bool(existed),
            },
            headers={"Cache-Control": "no-store"},
        )


    @PromptServer.instance.routes.post("/layer13/persistent_counter/reset")
    async def layer13_persistent_counter_reset_post(request):
        counter_name, existed = Layer13PersistentCounter.reset_counter(
            await _extract_counter_name(request)
        )
        return web.json_response(
            {
                "ok": True,
                "name": counter_name,
                "reset": True,
                "existed": bool(existed),
            },
            headers={"Cache-Control": "no-store"},
        )
