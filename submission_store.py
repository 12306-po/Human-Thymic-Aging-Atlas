"""Storage boundary for community submissions.

The v1 adapter uses only Streamlit's per-session mapping. It intentionally has
no filesystem, network, or release-directory write path.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


class SessionSubmissionStore:
    def __init__(self, session: MutableMapping[str, Any]):
        self.session = session

    def save(self, submission: dict[str, Any]) -> None:
        self.session["community_submission"] = submission

    def load(self) -> dict[str, Any] | None:
        return self.session.get("community_submission")

    def clear(self) -> None:
        self.session.pop("community_submission", None)


# TODO: Add an authenticated, reviewed persistence adapter only after storage,
# access control, retention, malware scanning, and audit requirements are defined.
# It must never write to release/ or automatically publish a dataset.
