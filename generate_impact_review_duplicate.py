"""
Orchestrator: Generate Architectural Impact Review.

Pipeline:
  1. Fetch PR diff from GitHub, Bitbucket
  2. Check if PR is trivial (skip detailed analysis for docs/config)
  3. Retrieve relevant knowledge-base context
  4. Generate the structured impact analysis via LLM
  5. Format as GitHub-compatible markdown
  6. Post (or update) the comment on the PR

Reuses the same LLM invocation patterns as GenerateReleaseNoteSummary
(Prompt Service, Bedrock, OpenAI) and the same config.json / AppConfig.
"""
from __future__ import annotations
#Adding Test Commit - Prod
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from model.architectural_review_request import (
    ArchitecturalReviewResult,
    DiffFile,
    KnowledgeBaseSource,
    PRDetails,
    ServiceImpact,
)
from utilities.config.app_config import AppConfig
from utilities.github_client import GitHubClient
from utilities.prompt_loader import PromptLoader
from activities.retrieve_knowledge_base import KnowledgeBaseRetriever

logger = logging.getLogger(__name__)


class GenerateImpactReview:
    """End-to-end orchestrator for the Architectural Impact Review."""

    _TRIVIAL_EXTENSIONS = {
        ".md", ".txt", ".rst", ".yml", ".yaml", ".json", ".toml",
        ".lock", ".gitignore", ".editorconfig", ".prettierrc",
        ".eslintrc", ".dockerignore", ".cfg", ".ini", ".csv",
    }
    _TRIVIAL_FILENAMES = {
        "changelog", "license", "readme", "contributing", "authors",
        "makefile", "dockerfile", ".env.example", ".prettierrc",
        ".eslintrc", ".editorconfig", ".gitignore", ".dockerignore",
        ".prettierignore", ".eslintignore", ".gitattributes",
    }

    def __init__(self, event: dict, config: AppConfig | None = None):
        self._event = event
        self._config = config or AppConfig()
        self._request_id = event.get("request_id") or str(uuid.uuid4())[:8]
        self._portal_base_url = getattr(self._config, "adapts_portal_base_url", "https://app.adapts.ai")
        self._max_diff_chars = int(getattr(self._config, "max_diff_chars", 15000))
        self._max_kb_chars = int(getattr(self._config, "max_knowledge_base_chars", 30000))

        self._llm_provider = (
            os.getenv("LLM_PROVIDER", "").strip().lower()
            or getattr(self._config, "llm_provider", "prompt_service")
        )
        self._default_model = getattr(self._config, "default_llm_model", "claude-4-sonnet")

    def run(self) -> Dict[str, Any]:
        start_time = time.time()
        logger.info("[%s] Starting Architectural Impact Review", self._request_id)

        try:
            pr_details = self._parse_pr_details()
            owner, repo = pr_details.repo_full_name.split("/", 1)
            pr_number = int(pr_details.pr_number)

            github = GitHubClient(pr_details.installation_id, self._config)

            # Step 1: Fetch diff
            logger.info("[%s] Fetching PR diff for %s#%d", self._request_id, pr_details.repo_full_name, pr_number)
            diff_files = self._fetch_diff_files(github, owner, repo, pr_number)
            diff_text = self._fetch_diff_text(github, owner, repo, pr_number)

            # Step 2: Trivial PR check
            if self._is_trivial_pr(diff_files, pr_details):
                result = self._build_trivial_result(pr_details)
                comment_body = self._format_comment(result)
                post_result = self._post_comment(github, owner, repo, pr_number, comment_body)
                elapsed = time.time() - start_time
                logger.info("[%s] Trivial PR — posted in %.1fs", self._request_id, elapsed)
                return {"result": result.model_dump(), "comment_posted": True, "elapsed_seconds": elapsed, **post_result}

            # Step 3: Retrieve knowledge base
            wiki_name = self._event.get("wiki_name") or pr_details.repo_full_name
            retriever = KnowledgeBaseRetriever(wiki_name=wiki_name, org_id=self._event.get("org_id", ""), config=self._config)
            kb_context, kb_sources, uncovered_paths = retriever.retrieve(
                diff_files=diff_files, pr_title=pr_details.pr_title, pr_body=pr_details.pr_body, max_chars=self._max_kb_chars,
            )

            # Step 4: Generate analysis
            result = self._generate_analysis(pr_details, diff_files, diff_text, kb_context, kb_sources, uncovered_paths)

            # Step 5+6: Format and post
            comment_body = self._format_comment(result)
            post_result = self._post_comment(github, owner, repo, pr_number, comment_body)

            elapsed = time.time() - start_time
            logger.info("[%s] Impact review posted in %.1fs", self._request_id, elapsed)
            return {"result": result.model_dump(), "comment_posted": True, "elapsed_seconds": elapsed, **post_result}

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("[%s] Pipeline failed after %.1fs: %s", self._request_id, elapsed, e, exc_info=True)
            return self._handle_failure(e, elapsed)

    # ------------------------------------------------------------------
    # Parse event
    # ------------------------------------------------------------------

    def _parse_pr_details(self) -> PRDetails:
        repo = self._event.get("repo_object") or self._event.get("repoDetails") or {}
        return PRDetails(
            pr_number=str(repo.get("pr_number") or self._event.get("pr_number", "")),
            pr_title=repo.get("pr_title") or self._event.get("pr_title", ""),
            pr_body=repo.get("pr_body") or self._event.get("pr_body", ""),
            pr_url=repo.get("pr_url") or self._event.get("pr_url", ""),
            repo_full_name=repo.get("repository_name") or self._event.get("repo_full_name", ""),
            base_branch=repo.get("branch") or self._event.get("base_branch", "main"),
            head_branch=self._event.get("head_branch", ""),
            author=repo.get("commit_author") or self._event.get("author", ""),
            installation_id=str(repo.get("installation_id") or self._event.get("installation_id", "")),
        )

    # ------------------------------------------------------------------
    # Diff fetching
    # ------------------------------------------------------------------

    def _fetch_diff_files(self, github: GitHubClient, owner: str, repo: str, pr_number: int) -> List[DiffFile]:
        raw_files = github.get_pr_files(owner, repo, pr_number)
        return [
            DiffFile(
                filename=f.get("filename", ""),
                status=f.get("status", "modified"),
                additions=f.get("additions", 0),
                deletions=f.get("deletions", 0),
                patch=f.get("patch", "")[:2000],
            )
            for f in raw_files
        ]

    def _fetch_diff_text(self, github: GitHubClient, owner: str, repo: str, pr_number: int) -> str:
        try:
            raw = github.get_pr_diff(owner, repo, pr_number)
            if len(raw) > self._max_diff_chars:
                return raw[:self._max_diff_chars] + "\n\n[... diff truncated ...]"
            return raw
        except Exception as e:
            logger.warning("Failed to fetch raw diff: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Trivial PR check
    # ------------------------------------------------------------------

    def _is_trivial_pr(self, diff_files: List[DiffFile], pr_details: PRDetails) -> bool:
        if not diff_files:
            return True
        total_changes = sum(f.additions + f.deletions for f in diff_files)
        if total_changes > 200 or len(diff_files) > 15:
            return False
        non_trivial = 0
        for f in diff_files:
            _, ext = os.path.splitext(f.filename)
            basename = os.path.basename(f.filename).lower()
            if ext.lower() not in self._TRIVIAL_EXTENSIONS and basename not in self._TRIVIAL_FILENAMES:
                non_trivial += 1
        return non_trivial == 0

    def _build_trivial_result(self, pr_details: PRDetails) -> ArchitecturalReviewResult:
        return ArchitecturalReviewResult(
            pr_number=pr_details.pr_number,
            pr_title=pr_details.pr_title,
            summary="This PR is a minor change (documentation, configuration, or non-code files) with no meaningful architectural implications.",
            is_trivial_pr=True,
        )

    # ------------------------------------------------------------------
    # LLM analysis generation
    # ------------------------------------------------------------------

    def _generate_analysis(
        self, pr_details: PRDetails, diff_files: List[DiffFile], diff_text: str,
        kb_context: str, kb_sources: List[KnowledgeBaseSource], uncovered_paths: List[str],
    ) -> ArchitecturalReviewResult:
        changed_files_summary = "\n".join(
            f"- {f.filename} ({f.status}: +{f.additions}/-{f.deletions})" for f in diff_files
        )

        prompt_data = PromptLoader.load_prompt_inline(
            "architectural_impact_review",
            repo_full_name=pr_details.repo_full_name,
            pr_number=pr_details.pr_number,
            pr_title=pr_details.pr_title,
            pr_author=pr_details.author,
            base_branch=pr_details.base_branch,
            pr_body=(pr_details.pr_body or "(No description provided)")[:2000],
            changed_files_summary=changed_files_summary,
            diff_text=diff_text or "(Diff not available)",
            knowledge_base_context=kb_context,
        )

        raw_response = self._invoke_llm(prompt_data["expectations"], prompt_data["prompt"])
        parsed = self._parse_llm_response(raw_response)

        services = [ServiceImpact(**s) for s in parsed.get("services_affected", [])]
        kb_used = [KnowledgeBaseSource(**s) for s in parsed.get("knowledge_base_used", [])] if parsed.get("knowledge_base_used") else kb_sources

        return ArchitecturalReviewResult(
            pr_number=pr_details.pr_number,
            pr_title=pr_details.pr_title,
            summary=parsed.get("summary", "Analysis could not be generated."),
            services_affected=services,
            boundaries_crossed=parsed.get("boundaries_crossed", []),
            flags_and_risks=parsed.get("flags_and_risks", []),
            knowledge_base_used=kb_used,
            knowledge_base_gaps=parsed.get("knowledge_base_gaps", uncovered_paths),
            is_trivial_pr=parsed.get("is_trivial_pr", False),
            relevant_resource_links=self._build_resource_links(kb_sources),
        )

    # ------------------------------------------------------------------
    # LLM invocation — same three providers as GenerateReleaseNoteSummary
    # ------------------------------------------------------------------

    def _invoke_llm(self, expectations: str, prompt: str) -> str:
        if self._llm_provider == "prompt_service":
            return self._invoke_prompt_service(expectations, prompt)
        elif self._llm_provider == "bedrock":
            return self._invoke_bedrock(expectations, prompt)
        elif self._llm_provider == "openai":
            return self._invoke_openai(expectations, prompt)
        raise ValueError(f"Unknown LLM provider: {self._llm_provider}")

    def _invoke_prompt_service(self, expectations: str, prompt: str) -> str:
        from prompt_service_client import PromptServiceClient
        from prompt_service_models import PromptRequest as PromptServiceRequest, LLMServiceType

        http_endpoint = os.getenv("PROMPT_SERVICE_HTTP_ENDPOINT") or getattr(self._config, "prompt_service_http_endpoint", "")
        ws_endpoint = os.getenv("WEBSOCKET_MANAGEMENT_ENDPOINT") or getattr(self._config, "websocket_management_endpoint", "")
        iam_role = os.getenv("PROMPT_SERVICE_IAM_ROLE") or getattr(self._config, "prompt_service_iam_role", "")

        request = PromptServiceRequest(
            action_type="promptWebSocket",
            request_id=f"arch-review-{os.urandom(8).hex()}",
            user_id="architectural_review",
            prompt=prompt,
            expectations=expectations,
            llm_model=self._default_model,
            provider_type=LLMServiceType.BEDROCK,
        )
        client = PromptServiceClient(endpoint_url=http_endpoint, websocket_url=ws_endpoint, iam_role_arn=iam_role)
        try:
            response = client.prompt_websocket(request)
            return (response.choices[0].message.content or "").strip()
        finally:
            client.close()

    def _invoke_bedrock(self, expectations: str, prompt: str) -> str:
        import boto3
        model_id = getattr(self._config, "release_notes_bedrock_model_id", "") or "anthropic.claude-3-5-sonnet-20241022-v2:0"
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "system": expectations,
            "messages": [{"role": "user", "content": prompt}],
        }
        client = boto3.client("bedrock-runtime", region_name=getattr(self._config, "aws_region", "us-east-1"))
        response = client.invoke_model(modelId=model_id, contentType="application/json", accept="application/json", body=json.dumps(body))
        result = json.loads(response["body"].read())
        text = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
        return text.strip()

    def _invoke_openai(self, expectations: str, prompt: str) -> str:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            import boto3
            secret_name = getattr(self._config, "openapi_key_secret_name", "")
            sm = boto3.client("secretsmanager", region_name=getattr(self._config, "aws_region", "us-east-1"))
            raw = sm.get_secret_value(SecretId=secret_name)["SecretString"]
            try:
                parsed = json.loads(raw)
                api_key = parsed.get("api_key") or parsed.get("OPENAI_API_KEY", "")
            except json.JSONDecodeError:
                api_key = raw.strip()
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=getattr(self._config, "openai_model", "gpt-4o"),
            messages=[{"role": "system", "content": expectations}, {"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()

    @staticmethod
    def _parse_llm_response(raw: str) -> Dict[str, Any]:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        logger.warning("Failed to parse LLM response as JSON; returning raw summary")
        return {"summary": raw[:2000]}

    # ------------------------------------------------------------------
    # Comment formatting (spec R-05)
    # ------------------------------------------------------------------

    def _format_comment(self, result: ArchitecturalReviewResult) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%B %d, %Y %H:%M UTC")
        wiki_link = f"{self._portal_base_url}/wiki/{result.pr_number}" if self._portal_base_url else ""
        link_part = f" | [View in Adapts]({wiki_link})" if wiki_link else ""

        lines = [
            f"## Architectural Impact Review \u2014 PR #{result.pr_number} \U0001F3D7\uFE0F",
            f"*Generated by Adapts.ai | {timestamp}{link_part}*",
            "",
        ]

        lines.append("### Summary")
        lines.append(result.summary)
        lines.append("")

        if result.is_trivial_pr:
            lines.append("---")
            lines.append("*This is a minor change with no architectural implications. No further analysis required.*")
            return "\n".join(lines)

        lines.append("### Services & Components Affected")
        if result.services_affected:
            lines.append("| Service / Component | Impact Type | Confidence |")
            lines.append("|---|---|---|")
            for svc in result.services_affected:
                lines.append(f"| {svc.service_name} | {svc.impact_type} | {svc.confidence} |")
        else:
            lines.append("No specific services identified as affected.")
        lines.append("")

        lines.append("### Architecture Boundaries Crossed")
        if result.boundaries_crossed:
            for b in result.boundaries_crossed:
                lines.append(f"- {b}")
        else:
            lines.append("No architecture boundaries identified as crossed in this change.")
        lines.append("")

        lines.append("### Flags & Risks \u26A0\uFE0F")
        if result.flags_and_risks:
            for r in result.flags_and_risks:
                lines.append(f"- {r}")
        else:
            lines.append("No architectural risks identified.")
        lines.append("")

        lines.append("### Knowledge Base Coverage")
        if result.knowledge_base_used:
            used_list = ", ".join(f"{s.title} ({s.source_type})" for s in result.knowledge_base_used[:10])
            lines.append(f"**Used:** {used_list}")
        else:
            lines.append("**Used:** No knowledge base content was available for this analysis.")
        if result.knowledge_base_gaps:
            gap_list = ", ".join(f"`{g}`" for g in result.knowledge_base_gaps[:20])
            lines.append(f"\n**Not covered:** {gap_list} \u2014 these files were not analyzed.")
        lines.append("")

        if result.relevant_resource_links:
            lines.append("### \U0001F4CE Relevant Resources")
            for link in result.relevant_resource_links:
                title = link.get("title", "Link")
                url = link.get("url", "")
                if url:
                    lines.append(f"- [{title}]({url})")
            lines.append("")

        return "\n".join(lines)

    def _build_resource_links(self, kb_sources: List[KnowledgeBaseSource]) -> List[dict]:
        links = []
        seen = set()
        for src in kb_sources[:5]:
            if src.wiki_name and src.wiki_name not in seen:
                seen.add(src.wiki_name)
                links.append({"title": f"{src.title} ({src.source_type})", "url": f"{self._portal_base_url}/wiki/{src.wiki_name}"})
        return links

    # ------------------------------------------------------------------
    # Comment posting
    # ------------------------------------------------------------------

    def _post_comment(self, github: GitHubClient, owner: str, repo: str, pr_number: int, body: str) -> Dict[str, Any]:
        try:
            result = github.post_or_update_comment(owner, repo, pr_number, body)
            return {"comment_id": result.get("id"), "comment_url": result.get("html_url", "")}
        except Exception as e:
            logger.error("[%s] Failed to post comment — posting fallback: %s", self._request_id, e)
            return self._post_fallback_comment(github, owner, repo, pr_number, str(e))

    def _post_fallback_comment(self, github: GitHubClient, owner: str, repo: str, pr_number: int, error_msg: str) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).strftime("%B %d, %Y %H:%M UTC")
        fallback = (
            f"## Architectural Impact Review \u2014 PR #{pr_number} \U0001F3D7\uFE0F\n"
            f"*Generated by Adapts.ai | {timestamp}*\n\n"
            "### \u26A0\uFE0F Generation Issue\n\n"
            "The architectural impact review could not be completed for this PR. "
            "This does not affect your ability to review, approve, or merge this PR.\n\n"
            f"**Reason:** {error_msg[:500]}\n\n"
            "The Adapts team has been notified. You can re-trigger the review by pushing a new commit."
        )
        try:
            result = github.post_or_update_comment(owner, repo, pr_number, fallback)
            return {"comment_id": result.get("id"), "comment_url": result.get("html_url", ""), "fallback": True}
        except Exception as fallback_err:
            logger.error("[%s] Fallback comment also failed: %s", self._request_id, fallback_err)
            return {"comment_posted": False, "error": str(fallback_err)}

    def _handle_failure(self, error: Exception, elapsed: float) -> Dict[str, Any]:
        try:
            pr_details = self._parse_pr_details()
            owner, repo = pr_details.repo_full_name.split("/", 1)
            pr_number = int(pr_details.pr_number)
            github = GitHubClient(pr_details.installation_id, self._config)
            post_result = self._post_fallback_comment(github, owner, repo, pr_number, str(error))
            return {"error": str(error), "elapsed_seconds": elapsed, **post_result}
        except Exception as e2:
            logger.error("[%s] Even fallback posting failed: %s", self._request_id, e2)
            return {"error": str(error), "elapsed_seconds": elapsed, "comment_posted": False}
