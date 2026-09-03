"""
Pipecat-native equivalent of the LangGraph safety_check node. Sits right
after STT in the pipeline — sees every final transcript BEFORE it reaches
the main conversational LLM/tool-calling. If flagged, responds with a
FIXED string (never generated fresh by the main LLM) and swallows the
frame — it never reaches the LLM/tools for that turn at all.

This is a hard route, not a prompt instruction: even if the main LLM's
own system-prompt safety line were somehow ignored, a flagged message
never gets that far in the first place.
"""
from loguru import logger
import time
from pipecat.frames.frames import TranscriptionFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from safety import classify_message, OUT_OF_SCOPE_RESPONSE


class SafetyGate(FrameProcessor):
    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            t0 = time.monotonic()
            flagged = await classify_message(frame.text)
            elapsed_ms = (time.monotonic() - t0) * 1000
            # Logged unconditionally (not just when flagged) — this is the
            # real, per-turn cost this gate adds in series before the main
            # LLM even starts, whether or not anything gets flagged. Bypassed
            # confirmations (see safety.py) should show ~0ms here; anything
            # that actually reaches the classifier will show its real network cost.
            logger.info(f"[safety-gate-latency] {elapsed_ms:.0f}ms for: {frame.text!r}")

            if flagged:
                logger.info(f"[safety-gate] FLAGGED: {frame.text!r}")
                await self.push_frame(
                    TTSSpeakFrame(text=OUT_OF_SCOPE_RESPONSE, append_to_context=False),
                    direction,
                )
                return  # swallow — never forward this transcript to the LLM/tools

        await self.push_frame(frame, direction)