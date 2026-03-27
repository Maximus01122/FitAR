import base64
import json
import logging
import time
from collections import deque

import cv2
import numpy as np
from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

from .services import POSE_BACKEND_NAME, build_active_backend


logger = logging.getLogger(__name__)


class PoseConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        logger.info(
            "WebSocket connection attempt received (backend=%s).",
            POSE_BACKEND_NAME,
        )
        self.backend = build_active_backend()
        self.frame_times = deque(maxlen=60)
        self.last_payload = None
        await self.accept()
        logger.info("WebSocket connection accepted.")

    async def disconnect(self, _close_code):
        if hasattr(self, "backend") and self.backend is not None:
            await sync_to_async(self.backend.close, thread_sensitive=True)()
        logger.info("Client connection closed")

    async def receive(self, text_data=None, bytes_data=None):
        if not text_data:
            return

        try:
            await self._handle_text_message(text_data)
        except Exception as exc:
            logger.error("WebSocket connection error: %s", exc)

    async def _handle_text_message(self, data: str):
        if data.startswith('{"command"'):
            command_data = json.loads(data)
            response = await sync_to_async(self.backend.handle_command, thread_sensitive=True)(command_data)
            if response:
                await self.send(text_data=json.dumps(response))
            return

        frame_timestamp = None
        if data.startswith("{"):
            try:
                parsed_payload = json.loads(data)
            except json.JSONDecodeError:
                parsed_payload = None
            if isinstance(parsed_payload, dict) and "frame" in parsed_payload:
                frame_timestamp = parsed_payload.get("ts")
                data = parsed_payload["frame"]

        start_time = time.time()

        if self.frame_times:
            avg_fps = len(self.frame_times) / sum(self.frame_times)
            if avg_fps < 20 and self.last_payload:
                payload = dict(self.last_payload)
                if frame_timestamp is not None:
                    payload["client_ts"] = frame_timestamp
                await self.send(text_data=json.dumps(payload))
                self.frame_times.append(time.time() - start_time)
                return

        if not data.startswith("data:image/jpeg;base64,"):
            logger.warning("Received malformed data packet")
            return

        try:
            _, encoded = data.split(",", 1)
            img_data = base64.b64decode(encoded)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as exc:
            logger.warning("Failed to decode frame: %s", exc)
            return

        if frame is None:
            logger.warning("Decoded frame is None")
            return

        backend_name = getattr(self.backend, "name", POSE_BACKEND_NAME)
        process_start = time.perf_counter()
        result = await sync_to_async(self.backend.process_frame, thread_sensitive=True)(frame)
        latency_ms = (time.perf_counter() - process_start) * 1000

        if result:
            result.setdefault("backend", backend_name)
            payload_to_send = result
            self.last_payload = result
        elif self.last_payload:
            payload_to_send = dict(self.last_payload)
        else:
            payload_to_send = {"landmarks": [], "backend": backend_name}

        payload_to_send["latency_ms"] = latency_ms
        if frame_timestamp is not None:
            payload_to_send["client_ts"] = frame_timestamp
        payload_to_send.setdefault("backend", backend_name)
        await self.send(text_data=json.dumps(payload_to_send))
        self.frame_times.append(time.time() - start_time)
