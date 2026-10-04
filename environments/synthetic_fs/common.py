from __future__ import annotations

import logging
import os
from typing import Literal


NodeKind = Literal["cluster", "merge_summary"]
JudgeBackend = Literal["openrouter", "gemini", "vllm"]
AnswererWorkspaceMode = Literal["synthetic_only", "synthetic_plus_raw"]
READ_FILE_MAX_LINES = 100
READ_FILE_MAX_CHARS = 12000
READ_MANY_MAX_FILES = 4
SYNTHETIC_PATH_SLUG_MAX_BYTES = 64
ANSWER_PREFIX = "Answer:"
STOP_MESSAGE = "STOP"
MODEL_HTTP_TIMEOUT_SECONDS = 300
MODEL_HTTP_MAX_ATTEMPTS = 8
MODEL_HTTP_RETRY_SLEEP_CAP_SECONDS = 60
BUILDER_COMPACTION_SUMMARY_MARKER = "[COMPACTED BUILDER CONTEXT SUMMARY]"
APPROX_CHARS_PER_TOKEN = 4
DEFAULT_BUILDER_COMPACTION_TRIGGER_TOKENS = 8000
DEFAULT_BUILDER_COMPACTION_KEEP_RECENT_TURNS = 2
DEFAULT_BUILDER_COMPACTION_MAX_OUTPUT_TOKENS = 1200
DEFAULT_BUILDER_COMPACTION_INPUT_MAX_CHARS = 60000
DEFAULT_BUILDER_EXECUTOR_BASE_URL = os.environ.get(
    "BUILDER_EXECUTOR_BASE_URL", "http://localhost:8000/v1/chat/completions"
)
BAD_SYNTHETIC_TEXT_VALUES = {"", "none", "null", "n/a", "na"}
LOGGER = logging.getLogger("environments.synthetic_fs")
