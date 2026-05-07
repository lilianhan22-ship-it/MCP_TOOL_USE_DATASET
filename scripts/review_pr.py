#!/usr/bin/env python3
"""Review MCP/ScienceToolBench task PRs and post GitHub comments.

The workflow runs from trusted base-branch code under pull_request_target. It
does not execute pull-request code. It only reads changed PR files through the
GitHub API, summarizes raw task bundles, asks an LLM for a structured review,
and posts the findings back to the PR.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import io
import json
import os
from pathlib import PurePosixPath
import re
import textwrap
import zipfile
from typing import Any

from github import Github
from openai import OpenAI


MAX_FILES_PER_TASK = 220
MAX_TEXT_BYTES = 80_000
MAX_TOTAL_PROMPT_CHARS = 240_000
REVIEW_COMMENT_MARKER = "<!-- MCP_TOOL_USE_DATA_REVIEW -->"

TEXT_EXTENSIONS = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".r",
    ".txt",
    ".tsv",
    ".yaml",
    ".yml",
}

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "task_id",
        "overall_status",
        "benchmark_fit",
        "summary",
        "findings",
        "merge_candidates",
        "recommended_next_steps",
    ],
    "properties": {
        "task_id": {"type": "string"},
        "overall_status": {
            "type": "string",
            "enum": [
                "usable",
                "needs_minor_fix",
                "needs_tool_fix",
                "needs_major_rework",
            ],
        },
        "benchmark_fit": {
            "type": "string",
            "enum": ["good", "borderline", "poor"],
        },
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "severity",
                    "category",
                    "title",
                    "evidence",
                    "recommended_action",
                ],
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "tools",
                            "task_quality",
                            "generalization",
                            "paths",
                            "artifacts",
                            "answer",
                            "dependencies",
                            "other",
                        ],
                    },
                    "title": {"type": "string"},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "recommended_action": {"type": "string"},
                },
            },
        },
        "merge_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["source_path", "target_module", "action", "reason"],
                "properties": {
                    "source_path": {"type": "string"},
                    "target_module": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": [
                            "none",
                            "merge_as_is",
                            "merge_with_generalization",
                            "split_then_merge",
                            "do_not_merge",
                        ],
                    },
                    "reason": {"type": "string"},
                },
            },
        },
        "recommended_next_steps": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


@dataclass(frozen=True)
class ChangedFile:
    path: str
    status: str


@dataclass(frozen=True)
class TaskTarget:
    task_id: str
    kind: str
    path: str


@dataclass
class FileSnapshot:
    path: str
    size: int
    changed: bool
    content: str | None
    note: str


def main() -> int:
    github_token = require_env("GITHUB_TOKEN")
    repo_name = require_env("REPO_NAME")
    pr_number = int(require_env("PR_NUMBER"))

    github = Github(github_token)
    repo = github.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    head_repo = pr.head.repo or repo
    head_ref = pr.head.sha

    changed_files = [
        ChangedFile(path=file.filename, status=file.status)
        for file in pr.get_files()
        if file.status != "removed"
    ]
    targets = detect_task_targets(changed_files, head_repo, head_ref)

    if not targets:
        post_comment(
            pr,
            build_no_target_comment(pr_number, changed_files),
        )
        return 0

    llm_api_key = require_env("LLM_API_KEY")
    llm_base_url = os.getenv("LLM_BASE_URL") or None
    llm_model = os.getenv("LLM_MODEL", "gpt-5.4")
    client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
    records = []
    for target in targets:
        try:
            snapshot = collect_target_snapshot(target, changed_files, head_repo, head_ref)
            prompt = build_review_prompt(target, snapshot, changed_files)
            record = call_review_model(client, llm_model, prompt)
        except Exception as exc:
            record = failed_review_record(target, exc)
        records.append(normalize_record(target, record))

    post_comment(pr, build_review_comment(pr_number, records, targets))
    return 0


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def detect_task_targets(
    changed_files: list[ChangedFile],
    head_repo: Any,
    ref: str,
) -> list[TaskTarget]:
    targets: dict[str, TaskTarget] = {}

    for item in changed_files:
        path = item.path
        if path.endswith(".zip"):
            task_id = PurePosixPath(path).stem
            targets[f"zip:{path}"] = TaskTarget(task_id=task_id, kind="zip", path=path)
            continue

        root = infer_directory_task_root(path)
        if root and github_path_exists(
            head_repo,
            f"{root}/task_content/task_content.json",
            ref,
        ):
            task_id = PurePosixPath(root).name
            targets[f"dir:{root}"] = TaskTarget(task_id=task_id, kind="directory", path=root)

    # Some PRs add a whole task directory but the changed file path may not hit a
    # marker. Probe plausible parent directories for task_content/task_content.json.
    for item in changed_files:
        for candidate in candidate_parent_dirs(item.path):
            key = f"dir:{candidate}"
            if key in targets:
                continue
            if github_path_exists(
                head_repo,
                f"{candidate}/task_content/task_content.json",
                ref,
            ):
                targets[key] = TaskTarget(
                    task_id=PurePosixPath(candidate).name,
                    kind="directory",
                    path=candidate,
                )
                break

    return sorted(targets.values(), key=lambda target: (target.kind, target.path))


def infer_directory_task_root(path: str) -> str | None:
    parts = path.split("/")
    if "task_content" in parts:
        idx = parts.index("task_content")
        if idx >= 1:
            return "/".join(parts[:idx])
    if "tools" in parts:
        idx = parts.index("tools")
        if idx >= 1:
            return "/".join(parts[:idx])
    if "input_data" in parts:
        idx = parts.index("input_data")
        if idx >= 1:
            return "/".join(parts[:idx])
    return None


def candidate_parent_dirs(path: str) -> list[str]:
    parts = path.split("/")
    candidates = []
    for idx in range(len(parts) - 1, 0, -1):
        candidates.append("/".join(parts[:idx]))
    return candidates[:5]


def github_path_exists(repo: Any, path: str, ref: str) -> bool:
    try:
        repo.get_contents(path, ref=ref)
        return True
    except Exception:
        return False


def collect_target_snapshot(
    target: TaskTarget,
    changed_files: list[ChangedFile],
    head_repo: Any,
    ref: str,
) -> list[FileSnapshot]:
    changed_paths = {item.path for item in changed_files}
    if target.kind == "zip":
        raw = fetch_file_bytes(head_repo, target.path, ref)
        return snapshot_zip(target.path, raw, changed_paths)
    if target.kind == "directory":
        return snapshot_directory(head_repo, target.path, ref, changed_paths)
    raise ValueError(f"Unsupported target kind: {target.kind}")


def fetch_file_bytes(repo: Any, path: str, ref: str) -> bytes:
    content = repo.get_contents(path, ref=ref)
    if isinstance(content, list):
        raise ValueError(f"Expected file but got directory: {path}")
    try:
        return content.decoded_content
    except Exception:
        blob = repo.get_git_blob(content.sha)
        return base64.b64decode(blob.content)


def snapshot_zip(
    zip_path: str,
    raw: bytes,
    changed_paths: set[str],
) -> list[FileSnapshot]:
    snapshots: list[FileSnapshot] = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        members = [info for info in zf.infolist() if not info.is_dir()]
        for info in members[:MAX_FILES_PER_TASK]:
            normalized = normalize_zip_member(info.filename)
            if normalized is None:
                snapshots.append(
                    FileSnapshot(
                        path=info.filename,
                        size=info.file_size,
                        changed=True,
                        content=None,
                        note="unsafe zip member path skipped",
                    )
                )
                continue
            content, note = read_zip_member_text(zf, info)
            snapshots.append(
                FileSnapshot(
                    path=f"{zip_path}!/{normalized}",
                    size=info.file_size,
                    changed=(zip_path in changed_paths),
                    content=content,
                    note=note,
                )
            )
        if len(members) > MAX_FILES_PER_TASK:
            snapshots.append(
                FileSnapshot(
                    path=f"{zip_path}!/...",
                    size=0,
                    changed=True,
                    content=None,
                    note=f"truncated after {MAX_FILES_PER_TASK} files",
                )
            )
    return snapshots


def normalize_zip_member(name: str) -> str | None:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        return None
    return str(pure)


def read_zip_member_text(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
) -> tuple[str | None, str]:
    suffix = PurePosixPath(info.filename).suffix.lower()
    if suffix not in TEXT_EXTENSIONS:
        return None, "binary or non-text file; content omitted"
    with zf.open(info) as handle:
        raw = handle.read(MAX_TEXT_BYTES + 1)
    truncated = len(raw) > MAX_TEXT_BYTES
    if truncated:
        raw = raw[:MAX_TEXT_BYTES]
    text = raw.decode("utf-8", errors="replace")
    if truncated:
        text += "\n\n...[truncated]..."
    return text, "text"


def snapshot_directory(
    repo: Any,
    root: str,
    ref: str,
    changed_paths: set[str],
) -> list[FileSnapshot]:
    paths = fetch_directory_paths(repo, root, ref)
    snapshots: list[FileSnapshot] = []
    for path in paths[:MAX_FILES_PER_TASK]:
        content, size, note = fetch_text_snapshot(repo, path, ref)
        snapshots.append(
            FileSnapshot(
                path=path,
                size=size,
                changed=(path in changed_paths),
                content=content,
                note=note,
            )
        )
    if len(paths) > MAX_FILES_PER_TASK:
        snapshots.append(
            FileSnapshot(
                path=f"{root}/...",
                size=0,
                changed=False,
                content=None,
                note=f"truncated after {MAX_FILES_PER_TASK} files",
            )
        )
    return snapshots


def fetch_directory_paths(repo: Any, root: str, ref: str) -> list[str]:
    paths: list[str] = []

    def walk(path: str) -> None:
        if len(paths) >= MAX_FILES_PER_TASK + 1:
            return
        contents = repo.get_contents(path, ref=ref)
        if not isinstance(contents, list):
            paths.append(contents.path)
            return
        for item in contents:
            if item.type == "dir":
                walk(item.path)
            else:
                paths.append(item.path)

    walk(root)
    return paths


def fetch_text_snapshot(repo: Any, path: str, ref: str) -> tuple[str | None, int, str]:
    suffix = PurePosixPath(path).suffix.lower()
    try:
        raw = fetch_file_bytes(repo, path, ref)
    except Exception as exc:
        return None, 0, f"could not fetch file: {type(exc).__name__}: {exc}"

    size = len(raw)
    if suffix not in TEXT_EXTENSIONS:
        return None, size, "binary or non-text file; content omitted"
    raw = raw[:MAX_TEXT_BYTES]
    text = raw.decode("utf-8", errors="replace")
    if size > MAX_TEXT_BYTES:
        text += "\n\n...[truncated]..."
    return text, size, "text"


def build_review_prompt(
    target: TaskTarget,
    snapshot: list[FileSnapshot],
    changed_files: list[ChangedFile],
) -> str:
    file_context = render_file_context(snapshot)
    changed_context = "\n".join(f"- {item.status}: {item.path}" for item in changed_files)
    schema_text = json.dumps(REVIEW_SCHEMA, indent=2)
    prompt = textwrap.dedent(
        f"""\
        You are auditing a newly delivered MCP/ScienceToolBench raw task bundle
        submitted through a GitHub pull request.

        Task ID: {target.task_id}
        Target kind: {target.kind}
        Target path: {target.path}

        Work only from the file contents included below. Do not use the web.
        Do not ask for more information. Do not suggest running project code.

        Changed files in this PR:
        {changed_context}

        File context for the detected task target:
        {file_context}

        Your goal is to identify real data-quality or task-quality problems
        that should be fixed before this task is merged. Do not nitpick. Focus
        on issues that would make the task content invalid, unsupported by the
        released data, misaligned with its scoring target, or impossible to
        solve with the provided domain-specific tools.

        Important scope rules:

        1. Do not flag task_content.json for containing expected answers,
           checklists, scoring rubrics, or expected artifacts. In our benchmark
           runner, the model only receives the model-facing ask field, not the
           full task folder.
        2. Do not flag generic utility tools with pass, NotImplemented, or
           placeholder bodies if they are clearly shared-framework utilities,
           such as read_excel, read_csv, read_parquet, search, generic file
           readers, or generic database lookup stubs. Only flag unfinished
           domain-specific tools that are necessary for this task.
        3. Do not require the task to reproduce every part of the original
           paper. A task is acceptable if it is derived from the paper, the
           question is reasonable, and the expected answer/checklist matches
           what the question asks.
        4. Do not judge whether the task absolutely requires tool use. We are
           not rejecting tasks just because a strong model could also solve
           parts with code. Focus on whether the released data and tools support
           the requested scientific work.
        5. Do not over-audit expected answer formatting, uniqueness, or numeric
           tolerance. For answers/checklists, only check whether they capture
           the core conclusion(s) and core figure/artifact point(s) that reflect
           the main finding asked by the task.
        6. Do not flag extra files merely because they are present in the data
           bundle. Extra source files are only a problem if they make the
           required inputs ambiguous, hide required data from the task question,
           or directly expose/contradict the core answer.
        7. Do not flag common or framework-level dependencies as data-provider
           issues. Only report a dependency issue if it is task-specific,
           blocks the core domain workflow, and is not something the shared
           benchmark runtime should reasonably provide.

        Review dimensions:

        A. Task content validity
        - Is the ask clear and content-wise coherent?
        - Is the task aligned with the source paper or source study it appears
          to come from?
        - Does the expected answer/checklist answer the same question that the
          ask asks?
        - Are the requested outputs and conclusions supported by the released
          input data?
        - Does the task ask for claims, comparisons, variables, cohorts,
          figures, or analyses that are not present in the released data?
        - Are there obvious mismatches between the task description and the
          actual files, such as wrong year ranges, units, dataset names, cohort
          names, or missing required inputs?

        B. Core answer and core figure/artifact coverage
        - Does the checklist or expected answer cover the core conclusion(s)
          the task is asking for?
        - If the task asks for figures or artifacts, do the expected
          figure/artifact points correspond to the core finding rather than
          irrelevant side outputs?
        - Are there missing core findings that should be scored?
        - Are there checklist items that reward conclusions unrelated to the
          task question?
        - Are there obvious cases where an incorrect or superficial answer could
          receive substantial credit because the checklist misses the main
          scientific point?

        C. Tool and data support
        Audit whether the provided domain-specific tools and released files can
        support the task. Focus on:
        - Domain tools that are too weak, brittle, incomplete, or internally
          inconsistent for the instructed workflow.
        - Tool functions that do not compose correctly, such as one function
          returning a schema that another function cannot consume.
        - Tools that are too one-shot or too paper-specific, especially if they
          directly generate the final conclusion, final table, or final figure
          instead of exposing reusable analysis primitives.
        - Required analysis that cannot be completed from the released files.
        - Task-specific dependencies that are undeclared and block the core
          domain workflow.
        - Tools that hard-code local paths, sheet names, row numbers, filenames,
          or figure layouts in ways that make the workflow brittle.
        - Missing data-alignment logic, such as column aliases, metadata joins,
          file manifests, sample ID mapping, cohort mapping, country mapping, or
          chronology-to-parameter mapping.

        Severity guidance:
        - high: principle-level issue that should be fixed by the data provider,
          such as invalid task/answer alignment, missing data needed for the
          requested analysis, domain tools unable to support the core workflow,
          or answer/checklist missing the core scientific target.
        - medium: important but fixable issue, such as incomplete domain tool
          implementation, brittle schema assumptions, unclear metadata joins, or
          tools needing generalization.
        - low: small engineering issue that our side can likely fix, such as
          minor path parameterization, small column alias adaptation, or
          straightforward input/output schema cleanup.

        Be concrete. Reference file paths, tool file names, and function names
        in the evidence strings. Keep the summary short and factual.
        Only report findings that fit the review dimensions above.

        Return only valid JSON matching this schema:
        {schema_text}
        """
    )
    if len(prompt) > MAX_TOTAL_PROMPT_CHARS:
        prompt = prompt[:MAX_TOTAL_PROMPT_CHARS] + "\n\n...[prompt truncated]..."
    return prompt


def render_file_context(snapshot: list[FileSnapshot]) -> str:
    blocks = []
    for item in snapshot:
        status = "CHANGED" if item.changed else "CONTEXT"
        header = f"### [{status}] {item.path} ({item.size} bytes; {item.note})"
        if item.content is None:
            blocks.append(header)
        else:
            blocks.append(f"{header}\n```text\n{item.content}\n```")
    return "\n\n".join(blocks) if blocks else "(No files were collected.)"


def call_review_model(client: OpenAI, model: str, prompt: str) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": "You are a strict benchmark data reviewer. Return JSON only.",
        },
        {"role": "user", "content": prompt},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
    text = response.choices[0].message.content or "{}"
    return parse_json_response(text)


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def failed_review_record(target: TaskTarget, exc: BaseException) -> dict[str, Any]:
    return {
        "task_id": target.task_id,
        "overall_status": "needs_major_rework",
        "benchmark_fit": "poor",
        "summary": f"Automated review failed: {type(exc).__name__}: {exc}",
        "findings": [
            {
                "severity": "high",
                "category": "other",
                "title": "Automated review failed",
                "evidence": [f"target={target.kind}:{target.path}"],
                "recommended_action": "Inspect the GitHub Actions log and rerun the review.",
            }
        ],
        "merge_candidates": [],
        "recommended_next_steps": ["Fix the review execution issue and rerun the workflow."],
    }


def normalize_record(target: TaskTarget, record: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "task_id": str(record.get("task_id") or target.task_id),
        "overall_status": str(record.get("overall_status") or "needs_major_rework"),
        "benchmark_fit": str(record.get("benchmark_fit") or "poor"),
        "summary": str(record.get("summary") or ""),
        "findings": record.get("findings") if isinstance(record.get("findings"), list) else [],
        "merge_candidates": (
            record.get("merge_candidates")
            if isinstance(record.get("merge_candidates"), list)
            else []
        ),
        "recommended_next_steps": (
            record.get("recommended_next_steps")
            if isinstance(record.get("recommended_next_steps"), list)
            else []
        ),
        "_target": {"kind": target.kind, "path": target.path},
    }
    if normalized["overall_status"] not in {
        "usable",
        "needs_minor_fix",
        "needs_tool_fix",
        "needs_major_rework",
    }:
        normalized["overall_status"] = "needs_major_rework"
    if normalized["benchmark_fit"] not in {"good", "borderline", "poor"}:
        normalized["benchmark_fit"] = "poor"
    return normalized


def build_review_comment(
    pr_number: int,
    records: list[dict[str, Any]],
    targets: list[TaskTarget],
) -> str:
    lines = [
        REVIEW_COMMENT_MARKER,
        f"## MCP Tool Use Data Review for PR #{pr_number}",
        "",
        "| Task | Target | Status | Fit | High | Medium | Low | Summary |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        counts = severity_counts(record.get("findings", []))
        target = record.get("_target", {})
        lines.append(
            "| {task} | `{kind}:{path}` | `{status}` | `{fit}` | {high} | {medium} | {low} | {summary} |".format(
                task=escape_md(record["task_id"]),
                kind=escape_md(target.get("kind", "")),
                path=escape_md(target.get("path", "")),
                status=escape_md(record["overall_status"]),
                fit=escape_md(record["benchmark_fit"]),
                high=counts["high"],
                medium=counts["medium"],
                low=counts["low"],
                summary=escape_md(record["summary"].replace("\n", " ")),
            )
        )

    for record in records:
        lines.extend(["", f"### {record['task_id']}", ""])
        findings = record.get("findings", [])
        if findings:
            lines.append("**Findings**")
            for finding in findings:
                lines.append(
                    "- [{severity}] [{category}] {title}: {action}".format(
                        severity=escape_md(str(finding.get("severity", ""))),
                        category=escape_md(str(finding.get("category", ""))),
                        title=escape_md(str(finding.get("title", ""))),
                        action=escape_md(str(finding.get("recommended_action", ""))),
                    )
                )
                for evidence in finding.get("evidence", [])[:5]:
                    lines.append(f"  - Evidence: `{escape_md(str(evidence))}`")
        else:
            lines.append("**Findings**: None")

        next_steps = record.get("recommended_next_steps", [])
        if next_steps:
            lines.extend(["", "**Recommended Next Steps**"])
            for step in next_steps[:8]:
                lines.append(f"- {escape_md(str(step))}")

    if not targets:
        lines.append("")
        lines.append("No task targets were detected.")

    return "\n".join(lines)


def build_no_target_comment(pr_number: int, changed_files: list[ChangedFile]) -> str:
    changed = "\n".join(f"- {item.status}: `{item.path}`" for item in changed_files[:100])
    return "\n".join(
        [
            REVIEW_COMMENT_MARKER,
            f"## MCP Tool Use Data Review for PR #{pr_number}",
            "",
            "No MCP task bundle was detected in this PR.",
            "",
            "Detected task targets are either raw task zip files or directories containing `task_content/task_content.json`.",
            "",
            "Changed files:",
            changed or "(none)",
        ]
    )


def severity_counts(findings: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"high": 0, "medium": 0, "low": 0}
    for finding in findings:
        severity = finding.get("severity")
        if severity in counts:
            counts[severity] += 1
    return counts


def escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def post_comment(pr: Any, body: str) -> None:
    body = body[:60_000]
    pr.create_issue_comment(body)


if __name__ == "__main__":
    raise SystemExit(main())
