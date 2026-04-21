"""
GitHub API client for fetching PR diffs and posting comments.

Uses the existing GitHub App credentials (same as PackagePullerService)
to generate installation tokens, then calls the GitHub REST API.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import jwt
import requests

from utilities.config.app_config import AppConfig

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"


class GitHubClient:
    """Thin wrapper around the GitHub REST API using App installation tokens."""

    def __init__(self, installation_id: str, config: AppConfig | None = None):
        self._config = config or AppConfig()
        self._installation_id = installation_id
        self._token: str | None = None
        self._token_expires: float = 0

    # ------------------------------------------------------------------
    # Authentication (same JWT flow as PackagePullerService)
    # ------------------------------------------------------------------

    def _get_private_key(self) -> str:
        import boto3

        secret_name = getattr(self._config, "github_private_key_secret_name", "")
        region = getattr(self._config, "aws_region", "us-east-1")
        sm = boto3.client("secretsmanager", region_name=region)
        raw = sm.get_secret_value(SecretId=secret_name)["SecretString"]
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed.get("apiSecret", raw).replace("\\n", "\n")
        except json.JSONDecodeError:
            pass
        return raw.replace("\\n", "\n")

    def _generate_jwt(self) -> str:
        app_id = getattr(self._config, "github_app_id", "") or os.getenv("GITHUB_APP_ID", "")
        private_key = self._get_private_key()
        now = int(time.time())
        payload = {
            "iat": now - 60,
            "exp": now + 300,
            "iss": str(app_id),
        }
        return jwt.encode(payload, private_key, algorithm="RS256")

    def _ensure_token(self) -> str:
        if self._token and time.time() < self._token_expires:
            return self._token
        jwt_token = self._generate_jwt()
        url = f"{GITHUB_API_BASE}/app/installations/{self._installation_id}/access_tokens"
        resp = requests.post(url, headers={
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }, timeout=15)
        if resp.status_code != 201:
            raise RuntimeError(f"GitHub token exchange failed: {resp.status_code} {resp.text[:500]}")
        data = resp.json()
        self._token = data["token"]
        self._token_expires = time.time() + 3500
        return self._token

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._ensure_token()}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    # ------------------------------------------------------------------
    # PR Diff
    # ------------------------------------------------------------------

    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> List[Dict[str, Any]]:
        """Fetch the list of changed files for a PR (paginated)."""
        files: List[Dict[str, Any]] = []
        page = 1
        while True:
            url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}/files"
            resp = requests.get(url, headers=self._headers(), params={"per_page": 100, "page": page}, timeout=30)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            files.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        logger.info("Fetched %d changed files for %s/%s#%d", len(files), owner, repo, pr_number)
        return files

    def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """Fetch the raw unified diff for a PR."""
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}"
        resp = requests.get(url, headers={
            **self._headers(),
            "Accept": "application/vnd.github.v3.diff",
        }, timeout=60)
        resp.raise_for_status()
        return resp.text

    # ------------------------------------------------------------------
    # PR Review Posting (uses Reviews API for "Resolve conversation" support)
    # ------------------------------------------------------------------

    def _create_pr_review(self, owner: str, repo: str, pr_number: int, body: str) -> Dict[str, Any]:
        """Submit a PR review with COMMENT event (gives 'Resolve conversation' button)."""
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
        resp = requests.post(url, headers=self._headers(), json={"body": body, "event": "COMMENT"}, timeout=30)
        if resp.status_code not in (200, 201):
            logger.error("Failed to create PR review: %s %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()
        logger.info("Created PR review on %s/%s#%d (review_id=%s)", owner, repo, pr_number, resp.json().get("id"))
        return resp.json()

    def _update_pr_review(self, owner: str, repo: str, pr_number: int, review_id: int, body: str) -> Dict[str, Any]:
        """Update the body of an existing PR review."""
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}/reviews/{review_id}"
        resp = requests.put(url, headers=self._headers(), json={"body": body}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _find_bot_review(self, owner: str, repo: str, pr_number: int, marker: str = "Architectural Impact Review") -> Optional[int]:
        """Find an existing Adapts bot review on the PR."""
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
        resp = requests.get(url, headers=self._headers(), params={"per_page": 100}, timeout=15)
        resp.raise_for_status()
        for review in resp.json():
            if marker in (review.get("body") or ""):
                user = review.get("user", {})
                if user.get("type") == "Bot" or "[bot]" in (user.get("login") or ""):
                    return review["id"]
        return None

    def post_or_update_comment(
        self, owner: str, repo: str, pr_number: int, body: str,
        marker: str = "Architectural Impact Review",
    ) -> Dict[str, Any]:
        """Create a new PR review or update the existing one (idempotent)."""
        existing_id = self._find_bot_review(owner, repo, pr_number, marker)
        if existing_id:
            logger.info("Updating existing review %d on %s/%s#%d", existing_id, owner, repo, pr_number)
            return self._update_pr_review(owner, repo, pr_number, existing_id, body)
        return self._create_pr_review(owner, repo, pr_number, body)
