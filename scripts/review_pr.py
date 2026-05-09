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
MAX_PDF_TEXT_CHARS = 40_000
MAX_TOTAL_PROMPT_CHARS = 240_000
MAX_MEDIA_BYTES = int(os.getenv("MAX_REVIEW_MEDIA_BYTES", str(25 * 1024 * 1024)))
REVIEW_COMMENT_MARKER = "<!-- MCP_TOOL_USE_DATA_REVIEW -->"
REVIEW_SCRIPT_VERSION = "tool-assessment-v1"

TEXT_EXTENSIONS = {
    ".csv",
    ".dat",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".r",
    ".tab",
    ".txt",
    ".tsv",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {"readme", "license", "manifest"}

ATTACHMENT_MIME_TYPES = {
    ".pdf": "application/pdf",
}

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "task_id",
        "model",
        "overall_status",
        "benchmark_fit",
        "summary",
        "reasoning",
        "tool_assessment",
        "findings",
        "merge_candidates",
        "todo_items",
    ],
    "properties": {
        "task_id": {"type": "string"},
        "model": {"type": "string"},
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
        "reasoning": {"type": "string"},
        "tool_assessment": {
            "type": "string",
            "description": (
                "Explicitly assess whether the provided domain tools are "
                "sufficient, too weak, too one-shot, or unnecessary for the "
                "requested workflow. Mention key tool files/functions."
            ),
        },
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
        "todo_items": {
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
    mime_type: str | None = None
    media_base64: str | None = None


@dataclass(frozen=True)
class MediaAttachment:
    path: str
    size: int
    changed: bool
    mime_type: str
    media_base64: str


@dataclass(frozen=True)
class PdfSmokeTestResult:
    task_id: str
    model: str
    target_path: str
    attachment_path: str | None
    attachment_mode: str
    status: str
    output: str


def main() -> int:
    github_token = require_env("GITHUB_TOKEN")
    repo_name = require_env("REPO_NAME")
    pr_number = int(require_env("PR_NUMBER"))

    github = Github(github_token)
    repo = github.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    head_repo = pr.head.repo or repo
    head_ref = pr.head.sha
    base_ref = pr.base.sha

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
    llm_models = parse_review_models(os.getenv("LLM_REVIEW_MODELS"))
    client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
    records = []
    smoke_results: list[PdfSmokeTestResult] = []
    for target in targets:
        try:
            snapshot = collect_target_snapshot(target, changed_files, head_repo, head_ref)
            base_snapshot = collect_base_snapshot(target, repo, base_ref)
            prompt = build_review_prompt(target, snapshot, base_snapshot, changed_files)
            media_attachments, media_omissions = collect_media_attachments(snapshot)
            for model in llm_models:
                try:
                    record = call_review_model(
                        client,
                        model,
                        prompt,
                        media_attachments,
                        media_omissions,
                    )
                except Exception as exc:
                    record = failed_review_record(target, model, exc)
                records.append(normalize_record(target, model, record))
            smoke_results.extend(
                run_pdf_smoke_tests(client, target, llm_models, media_attachments)
            )
        except Exception as exc:
            for model in llm_models:
                record = failed_review_record(target, model, exc)
                records.append(normalize_record(target, model, record))
                smoke_results.append(
                    PdfSmokeTestResult(
                        task_id=target.task_id,
                        model=model,
                        target_path=target.path,
                        attachment_path=None,
                        attachment_mode="not_run",
                        status="failed",
                        output=(
                            "PDF smoke test was not run because file collection "
                            f"failed: {type(exc).__name__}: {exc}"
                        ),
                    )
                )

    post_comment(pr, build_review_comment(pr_number, records, targets, smoke_results))
    return 0


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def parse_review_models(raw: str | None) -> list[str]:
    aliases = {"claude-opus-4.7": "claude-opus-4-7"}
    models = []
    for item in (raw or "").split(","):
        model = item.strip()
        if model:
            models.append(aliases.get(model, model))
    return models or ["gpt-5.5", "claude-opus-4-7"]


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


def collect_base_snapshot(
    target: TaskTarget,
    base_repo: Any,
    base_ref: str,
) -> list[FileSnapshot]:
    try:
        if target.kind == "zip":
            raw = fetch_file_bytes(base_repo, target.path, base_ref)
            snapshots = snapshot_zip(target.path, raw, set())
        elif target.kind == "directory":
            if not github_path_exists(
                base_repo,
                f"{target.path}/task_content/task_content.json",
                base_ref,
            ):
                return []
            snapshots = snapshot_directory(base_repo, target.path, base_ref, set())
        else:
            return []
    except Exception:
        return []

    for item in snapshots:
        item.path = f"BASE_BRANCH/{item.path}"
        item.changed = False
    return snapshots


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
            content, note, mime_type, media_base64 = read_zip_member_text(zf, info)
            snapshots.append(
                FileSnapshot(
                    path=f"{zip_path}!/{normalized}",
                    size=info.file_size,
                    changed=(zip_path in changed_paths),
                    content=content,
                    note=note,
                    mime_type=mime_type,
                    media_base64=media_base64,
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
) -> tuple[str | None, str, str | None, str | None]:
    suffix = PurePosixPath(info.filename).suffix.lower()
    if suffix == ".pdf":
        raw_pdf = zf.read(info)
        content, note = extract_pdf_text(raw_pdf)
        mime_type, media_base64, media_note = encode_target_paper_attachment(
            info.filename,
            raw_pdf,
        )
        return content, append_note(note, media_note), mime_type, media_base64
    if not is_text_like_path(info.filename):
        return None, "binary or non-text file; content omitted", None, None
    with zf.open(info) as handle:
        raw = handle.read(MAX_TEXT_BYTES + 1)
    truncated = len(raw) > MAX_TEXT_BYTES
    if truncated:
        raw = raw[:MAX_TEXT_BYTES]
    text = raw.decode("utf-8", errors="replace")
    if truncated:
        text += "\n\n...[truncated]..."
    return text, "text", None, None


def snapshot_directory(
    repo: Any,
    root: str,
    ref: str,
    changed_paths: set[str],
) -> list[FileSnapshot]:
    paths = fetch_directory_paths(repo, root, ref)
    snapshots: list[FileSnapshot] = []
    for path in paths[:MAX_FILES_PER_TASK]:
        content, size, note, mime_type, media_base64 = fetch_text_snapshot(repo, path, ref)
        snapshots.append(
            FileSnapshot(
                path=path,
                size=size,
                changed=(path in changed_paths),
                content=content,
                note=note,
                mime_type=mime_type,
                media_base64=media_base64,
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


def fetch_text_snapshot(
    repo: Any,
    path: str,
    ref: str,
) -> tuple[str | None, int, str, str | None, str | None]:
    suffix = PurePosixPath(path).suffix.lower()
    try:
        raw = fetch_file_bytes(repo, path, ref)
    except Exception as exc:
        return None, 0, f"could not fetch file: {type(exc).__name__}: {exc}", None, None

    size = len(raw)
    if suffix == ".pdf":
        content, note = extract_pdf_text(raw)
        mime_type, media_base64, media_note = encode_target_paper_attachment(path, raw)
        return content, size, append_note(note, media_note), mime_type, media_base64
    if not is_text_like_path(path):
        return None, size, "binary or non-text file; content omitted", None, None
    raw = raw[:MAX_TEXT_BYTES]
    text = raw.decode("utf-8", errors="replace")
    if size > MAX_TEXT_BYTES:
        text += "\n\n...[truncated]..."
    return text, size, "text", None, None


def is_text_like_path(path: str) -> bool:
    pure = PurePosixPath(path)
    suffix = pure.suffix.lower()
    return suffix in TEXT_EXTENSIONS or pure.name.lower() in TEXT_FILENAMES


def media_mime_type(path: str) -> str | None:
    return ATTACHMENT_MIME_TYPES.get(PurePosixPath(path).suffix.lower())


def is_target_paper_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    if "!/" in normalized:
        normalized = normalized.split("!/", 1)[1]
    target_suffix = "input_data/reference_paper/target.pdf"
    return normalized == target_suffix or normalized.endswith(f"/{target_suffix}")


def encode_target_paper_attachment(
    path: str,
    raw: bytes,
) -> tuple[str | None, str | None, str]:
    if not is_target_paper_path(path):
        return (
            None,
            None,
            "not attached as multimodal input; only input_data/reference_paper/target.pdf is attached",
        )
    mime_type = media_mime_type(path)
    if mime_type is None:
        return None, None, ""
    if len(raw) > MAX_MEDIA_BYTES:
        return (
            mime_type,
            None,
            f"{mime_type} not attached because file exceeds {format_bytes(MAX_MEDIA_BYTES)}",
        )
    return (
        mime_type,
        base64.b64encode(raw).decode("utf-8"),
        f"{mime_type} available as multimodal input",
    )


def append_note(note: str, extra: str) -> str:
    if not extra:
        return note
    return f"{note}; {extra}"


def format_bytes(value: int) -> str:
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.1f} MiB"
    if value >= 1024:
        return f"{value / 1024:.1f} KiB"
    return f"{value} bytes"


def extract_pdf_text(raw_pdf: bytes) -> tuple[str | None, str]:
    try:
        import fitz
    except Exception:
        return None, "PDF text omitted; PyMuPDF is not installed"

    try:
        doc = fitz.open(stream=raw_pdf, filetype="pdf")
        chunks = []
        total = 0
        for page in doc:
            text = page.get_text("text")
            if not text:
                continue
            remaining = MAX_PDF_TEXT_CHARS - total
            if remaining <= 0:
                break
            chunks.append(text[:remaining])
            total += min(len(text), remaining)
        doc.close()
    except Exception as exc:
        return None, f"PDF text extraction failed: {type(exc).__name__}: {exc}"

    text = "\n".join(chunks).strip()
    if not text:
        return None, "PDF text extraction produced no text"
    note = "pdf_text"
    if len(text) >= MAX_PDF_TEXT_CHARS:
        text += "\n\n...[pdf text truncated]..."
        note = "pdf_text_truncated"
    return text, note


def build_review_prompt(
    target: TaskTarget,
    snapshot: list[FileSnapshot],
    base_snapshot: list[FileSnapshot],
    changed_files: list[ChangedFile],
) -> str:
    file_context = render_file_context(snapshot)
    base_context = render_file_context(base_snapshot)
    changed_context = "\n".join(f"- {item.status}: {item.path}" for item in changed_files)
    schema_text = json.dumps(REVIEW_SCHEMA, indent=2)
    prompt = textwrap.dedent(
        f"""\
        You are auditing a newly delivered MCP/ScienceToolBench raw task bundle
        submitted through a GitHub pull request.

        Task ID: {target.task_id}
        Target kind: {target.kind}
        Target path: {target.path}

        Work only from the file contents included below and the target-paper
        multimodal attachment, if present in the same model request. Do not use
        the web. Do not ask for more information. Do not suggest running
        project code.

        Changed files in this PR:
        {changed_context}

        File context for the detected task target:
        {file_context}

        Base-branch context for the same target, if it already existed before
        this PR:
        {base_context}

        Your goal is to identify real data-quality or task-quality problems
        that should be fixed before this task is merged. Do not nitpick. Focus
        on issues that would make the task content invalid, unsupported by the
        released data, misaligned with its scoring target, or impossible to
        solve with the provided domain-specific tools.

        Important scope rules:

        1. Do not flag task_content.json for containing expected answers,
           checklists, scoring rubrics, or expected artifacts. In our benchmark
           runner, the model only receives the model-facing instruction/ask
           field, not the full task folder.
        2. Do not flag generic utility tools with pass, NotImplemented, or
           placeholder bodies if they are clearly shared-framework utilities,
           such as read_excel, read_csv, read_parquet, search, generic file
           readers, or generic database lookup stubs. Only flag unfinished
           domain-specific tools that are necessary for this task.
        3. Do not require the task to reproduce every part of the original
           paper. A task is acceptable if it is derived from the paper, the
           question is reasonable, and the expected answer/checklist matches
           what the question asks. However, the task must still cover the main
           scientific thread or central result that it claims to evaluate; do
           not accept a task that only samples a marginal side detail while the
           answer/checklist claims the paper's main contribution.
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
        8. Do not require every task bundle to implement generic PDF parsing,
           image inspection, or generic file-reading tools if those are shared
           runtime capabilities and the source files are included. Only flag
           source-access issues when the task relies on paper-specific or
           source-specific values that are not reachable from released data,
           included source content, or a reasonable generic parsing path.
        9. Treat an attached `input_data/reference_paper/target.pdf` as an
           accessible source for this automated review. Do not tell the data
           provider to add extracted target.pdf text, a PDF parser, a PDF table
           tool, or a structured paper-parameter file solely because the values
           come from target.pdf. Only flag this class of issue if target.pdf is
           missing/omitted, the attached paper genuinely does not contain the
           needed information, or the task requires values not present in either
           released data or the target paper.

        Review dimensions:

        A. Task content validity
        - Is the instruction/ask clear and content-wise coherent?
        - Is the instruction/ask appropriately fuzzy: it should state what the
          model needs to accomplish without revealing detailed step-by-step
          execution instructions, exact analysis commands, or answer-producing
          implementation details. Do not be overly strict; only flag this when
          the instruction clearly says things like "first do X, then run Y, then
          compute Z" in a way that removes the need for agent planning.
        - Is the task aligned with the source paper or source study it appears
          to come from?
        - Does the task cover the paper/source's main scientific thread at an
          appropriate scope, rather than an isolated detail that misses the
          central result?
        - Does the expected answer/checklist answer the same question that the
          instruction/ask asks?
        - Do the instruction/ask, expected answer/checklist, released data, and
          source paper all point to the same scientific claim, variables,
          dataset, cohort/object, method, and output artifacts?
        - Are the requested outputs and conclusions supported by the released
          input data?
        - Does the task ask for claims, comparisons, variables, cohorts,
          figures, or analyses that are not present in the released data?
        - Are there obvious mismatches between the task description and the
          actual files, such as wrong year ranges, units, dataset names, cohort
          names, or missing required inputs?

        Flag as high severity if the task question and expected answer do not
        match, if the task/answer is not aligned with the source paper's main
        claim for the chosen scope, if released data cannot support the
        requested conclusion, or if the task seems scientifically/content-wise
        invalid.

        B. Core answer and core figure/artifact coverage
        - Does the checklist or expected answer cover the core conclusion(s)
          the task is asking for?
        - Do the answer weights or checklist items prioritize the central
          scientific conclusion, not only surface artifacts or peripheral
          details?
        - If the task asks for figures or artifacts, do the expected
          figure/artifact points correspond to the core finding rather than
          irrelevant side outputs?
        - Are there missing core findings that should be scored?
        - Are there checklist items that reward conclusions unrelated to the
          task question?
        - Are there obvious cases where an incorrect or superficial answer could
          receive substantial credit because the checklist misses the main
          scientific point?

        Flag as high severity only if the scoring target is fundamentally
        misaligned with the task. Use medium severity for missing but fixable
        core checklist or figure points.

        C. Source-paper alignment

        If the bundle includes a reference paper, source PDF, README, extracted
        text, or citation metadata, use it to check:
        - Whether the task instruction faithfully represents the paper/source.
        - Whether the expected answer/checklist includes the central
          paper-supported conclusion for the scoped task.
        - Whether quoted numbers, named figures, units, periods, thresholds,
          cohorts, materials, samples, or variables match the source.
        - Whether released data/tools provide enough evidence for the model to
          reproduce or justify the requested source-aligned conclusion.

        If `input_data/reference_paper/target.pdf` is attached as multimodal
        input, inspect that attachment directly for source-paper quantities,
        figures, tables, and claims. Do not describe that target paper as "only
        a binary PDF" or inaccessible merely because the task bundle's domain
        tools do not implement PDF extraction.

        Flag as high severity for paper/source-answer contradictions, invented
        claims, wrong key numbers, wrong figure mapping, or a task that asks for
        a conclusion the source does not support. Use medium severity when the
        source alignment is mostly correct but missing an important caveat,
        uncertainty, or supporting evidence from the source.

        D. Tool and data support
        Audit whether the provided domain-specific tools and released files can
        support the task. Focus on:
        - Domain tools that are too weak, brittle, incomplete, or internally
          inconsistent for the instructed workflow.
        - Tool granularity: tools should be composable domain primitives that
          expose useful analysis operations, not only low-level file I/O and not
          one-click functions that directly return the final report, final
          conclusion, or hidden answer.
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

        Generic source inspection is not a required domain-tool feature here.
        Do not flag missing PDF/image parsing helpers, read_text limitations on
        binary PDFs, or missing structured paper-parameter files when the
        original target paper itself is included in the provided context,
        extracted PDF text, or multimodal attachment. Instead, evaluate whether
        the task, expected answer, data, tools, and target paper are mutually
        consistent.

        For tools that are not general enough:
        - If the tool can be naturally split into reusable steps using the
          existing code and data, recommend generalization or splitting.
        - If the tool hides final answers, hard-coded scientific thresholds, or
          paper-specific conclusions that cannot be justified from released data,
          flag it as a task/data-provider issue.

        E. Bundle hygiene

        Check for avoidable packaging problems that would make review,
        execution, or merge harder:
        - Generated caches or transient files, such as `__pycache__`, `.pyc`,
          notebook checkpoint folders, logs, scratch files, or temporary outputs.
        - Local absolute paths, machine-specific paths, or references to files
          outside the task bundle.
        - Executed notebook outputs that reveal answers or make the notebook
          non-clean as a source artifact.
        - Hidden-answer leakage in tool names, docstrings, README text,
          parameter names, return fields, or comments.
        - Undeclared task-specific dependencies that are needed for the core
          workflow.

        Use low or medium severity for ordinary hygiene issues. Use high
        severity only if the issue leaks the answer, blocks task execution, or
        makes the released bundle scientifically ambiguous.

        Comparison with current active task root:
        If this PR updates an existing task already present on the base branch:
        - Identify net-new useful domain tools or functions.
        - Identify regressions, duplicated low-quality tools, or tools that
          should not be merged.
        - For each merge candidate, say whether it can be merged as-is, needs
          generalization, needs splitting, or should not be merged.
        If no base-branch version is included in the file context, do not invent
        a comparison.

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
        Only report findings that fit the review dimensions above. If an issue
        is outside this scope, ignore it.

        Always fill `tool_assessment`, even when you return no findings.
        Explicitly state whether the provided domain tools are sufficient for
        the requested workflow, too weak, too one-shot, or not necessary for the
        task. Mention the key relevant tool files/functions by path or name.

        Put the detailed rationale in `reasoning`. Put concise actionable TODOs
        in `todo_items`; these TODOs will be shown outside the folded detail
        block in the GitHub PR comment.

        Return only valid JSON matching this schema:
        {schema_text}
        """
    )
    if len(prompt) > MAX_TOTAL_PROMPT_CHARS:
        prompt = prompt[:MAX_TOTAL_PROMPT_CHARS] + "\n\n...[prompt truncated]..."
    return prompt


def render_file_context(snapshot: list[FileSnapshot]) -> str:
    blocks = []
    for item in sorted(snapshot, key=snapshot_prompt_priority):
        status = "CHANGED" if item.changed else "CONTEXT"
        header = f"### [{status}] {item.path} ({item.size} bytes; {item.note})"
        if item.content is None:
            blocks.append(header)
        else:
            blocks.append(f"{header}\n```text\n{item.content}\n```")
    return "\n\n".join(blocks) if blocks else "(No files were collected.)"


def snapshot_prompt_priority(item: FileSnapshot) -> tuple[int, str]:
    path = item.path
    lower = path.lower()
    if "/task_content/" in lower:
        rank = 0
    elif "/tools/" in lower:
        rank = 1
    elif lower.endswith("/readme") or lower.endswith("readme.md"):
        rank = 2
    elif "/input_data/reference_paper/target.pdf" in lower:
        rank = 3
    elif "/input_data/reference_paper/" in lower and lower.endswith(".pdf"):
        rank = 4
    elif "/input_data/" in lower and item.content is not None:
        rank = 5
    elif "/artifacts/" in lower:
        rank = 6
    else:
        rank = 7
    return (rank, path)


def collect_media_attachments(
    snapshot: list[FileSnapshot],
) -> tuple[list[MediaAttachment], list[str]]:
    attachments: list[MediaAttachment] = []
    omissions: list[str] = []

    for item in sorted(snapshot, key=snapshot_prompt_priority):
        if item.mime_type is None:
            continue
        label = "CHANGED" if item.changed else "CONTEXT"
        descriptor = (
            f"{label} {item.path} ({format_bytes(item.size)}; {item.mime_type})"
        )
        if item.media_base64 is None:
            omissions.append(f"- omitted: {descriptor}; {item.note}")
            continue
        if attachments:
            omissions.append(
                f"- omitted: {descriptor}; only one target paper attachment is sent"
            )
            continue
        attachments.append(
            MediaAttachment(
                path=item.path,
                size=item.size,
                changed=item.changed,
                mime_type=item.mime_type,
                media_base64=item.media_base64,
            )
        )

    return attachments, omissions


def render_media_manifest(
    attachments: list[MediaAttachment],
    omissions: list[str],
) -> str:
    lines: list[str] = []
    if attachments:
        lines.append("Target paper attached as a multimodal file item:")
        for item in attachments:
            label = "CHANGED" if item.changed else "CONTEXT"
            lines.append(
                f"- attached: {label} {item.path} "
                f"({format_bytes(item.size)}; {item.mime_type})"
            )
    if omissions:
        if lines:
            lines.append("")
        lines.append("Target paper files not attached as multimodal input:")
        lines.extend(omissions)
    return "\n".join(lines)


def call_review_model(
    client: OpenAI,
    model: str,
    prompt: str,
    media_attachments: list[MediaAttachment],
    media_omissions: list[str],
) -> dict[str, Any]:
    if model_uses_text_only_review(model):
        media_attachments = []

    prompt_with_media = append_media_manifest(
        prompt,
        media_attachments,
        media_omissions,
    )
    prompt_without_media = append_text_only_fallback_note(
        prompt,
        media_attachments,
        media_omissions,
    )

    response: Any | None = None
    for attachment_mode, json_mode in review_call_attempts(media_attachments):
        attempt_prompt = (
            prompt_without_media if attachment_mode == "none" else prompt_with_media
        )
        try:
            response = call_responses_model(
                client,
                model,
                attempt_prompt,
                media_attachments if attachment_mode != "none" else [],
                json_mode=json_mode,
                attachment_mode=attachment_mode,
            )
            break
        except Exception:
            response = None

    if response is None:
        response = call_chat_model(client, model, prompt_without_media)

    return parse_json_response(extract_response_text(response))


def append_media_manifest(
    prompt: str,
    media_attachments: list[MediaAttachment],
    media_omissions: list[str],
) -> str:
    media_manifest = render_media_manifest(media_attachments, media_omissions)
    if not media_manifest:
        return prompt
    return (
        prompt
        + "\n\nMultimodal attachment manifest:\n"
        + media_manifest
        + "\n\n"
        + "Inspect the attached target paper directly when checking source "
        + "alignment, figures, plots, tables, or paper-supported claims. Do "
        + "not claim that the target paper is inaccessible unless the manifest "
        + "says it was omitted or the model cannot parse it. Do not recommend "
        + "adding PDF extraction tools or structured paper-parameter files "
        + "solely to expose information that is already present in this "
        + "attached target paper."
    )


def append_text_only_fallback_note(
    prompt: str,
    media_attachments: list[MediaAttachment],
    media_omissions: list[str],
) -> str:
    if not media_attachments and not media_omissions:
        return prompt
    media_manifest = render_media_manifest([], media_omissions)
    note = (
        "\n\nMultimodal attachment manifest:\n"
        "No target paper is attached in this fallback model request. Use the "
        "extracted PDF text and file context included above; do not fail the "
        "review solely because the fallback request is text-only."
    )
    if media_manifest:
        note += "\n\n" + media_manifest
    return prompt + note


def review_call_attempts(
    media_attachments: list[MediaAttachment],
) -> list[tuple[str, bool]]:
    if media_attachments:
        return [
            ("file", True),
            ("file", False),
            ("relay_image", True),
            ("relay_image", False),
            ("none", True),
            ("none", False),
        ]
    return [("none", True), ("none", False)]


def call_responses_model(
    client: OpenAI,
    model: str,
    prompt: str,
    media_attachments: list[MediaAttachment],
    json_mode: bool,
    attachment_mode: str,
) -> Any:
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": "You are a strict benchmark data reviewer. Return JSON only.\n\n"
            + prompt,
        }
    ]
    for item in media_attachments:
        content.append(
            {
                "type": "input_text",
                "text": (
                    f"Attached file: {item.path} "
                    f"({format_bytes(item.size)}; {item.mime_type})"
                ),
            }
        )
        content.append(media_content_item(item, attachment_mode))

    kwargs: dict[str, Any] = {
        "model": model,
        "input": [{"role": "user", "content": content}],
    }
    add_temperature_if_supported(kwargs, model)
    if json_mode:
        kwargs["text"] = {"format": {"type": "json_object"}}
    return client.responses.create(**kwargs)


def media_content_item(item: MediaAttachment, attachment_mode: str) -> dict[str, Any]:
    if attachment_mode == "file":
        return {
            "type": "input_file",
            "filename": PurePosixPath(item.path).name,
            "file_data": f"data:{item.mime_type};base64,{item.media_base64}",
        }
    if attachment_mode == "relay_image":
        return {
            "type": "input_image",
            "image_base64": item.media_base64,
            "mime_type": item.mime_type,
        }
    raise ValueError(f"Unsupported attachment mode: {attachment_mode}")


def call_chat_model(client: OpenAI, model: str, prompt: str) -> Any:
    messages = [
        {
            "role": "system",
            "content": "You are a strict benchmark data reviewer. Return JSON only.",
        },
        {"role": "user", "content": prompt},
    ]
    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        add_temperature_if_supported(kwargs, model)
        response = client.chat.completions.create(**kwargs)
    except Exception:
        kwargs = {"model": model, "messages": messages}
        add_temperature_if_supported(kwargs, model)
        response = client.chat.completions.create(**kwargs)
    return response


def add_temperature_if_supported(kwargs: dict[str, Any], model: str) -> None:
    if model_supports_temperature(model):
        kwargs["temperature"] = 0.3


def model_supports_temperature(model: str) -> bool:
    lowered = model.lower()
    return "claude" not in lowered and "anthropic" not in lowered


def model_uses_text_only_review(model: str) -> bool:
    return not model_supports_temperature(model)


def run_pdf_smoke_tests(
    client: OpenAI,
    target: TaskTarget,
    models: list[str],
    media_attachments: list[MediaAttachment],
) -> list[PdfSmokeTestResult]:
    if not media_attachments:
        return [
            PdfSmokeTestResult(
                task_id=target.task_id,
                model=model,
                target_path=target.path,
                attachment_path=None,
                attachment_mode="not_run",
                status="failed",
                output="No target.pdf attachment was available for the PDF smoke test.",
            )
            for model in models
        ]

    attachment = media_attachments[0]
    results = []
    for model in models:
        if model_uses_text_only_review(model):
            continue
        results.append(call_pdf_smoke_test(client, target, model, attachment))
    return results


def call_pdf_smoke_test(
    client: OpenAI,
    target: TaskTarget,
    model: str,
    attachment: MediaAttachment,
) -> PdfSmokeTestResult:
    errors = []
    for attachment_mode in ("file", "relay_image"):
        try:
            response = call_pdf_smoke_model(client, model, attachment, attachment_mode)
            return PdfSmokeTestResult(
                task_id=target.task_id,
                model=model,
                target_path=target.path,
                attachment_path=attachment.path,
                attachment_mode=attachment_mode,
                status="ok",
                output=extract_response_text(response).strip(),
            )
        except Exception as exc:
            errors.append(f"{attachment_mode}: {type(exc).__name__}: {exc}")

    return PdfSmokeTestResult(
        task_id=target.task_id,
        model=model,
        target_path=target.path,
        attachment_path=attachment.path,
        attachment_mode="failed",
        status="failed",
        output="PDF attachment smoke test failed for all attachment modes:\n"
        + "\n".join(f"- {error}" for error in errors),
    )


def call_pdf_smoke_model(
    client: OpenAI,
    model: str,
    attachment: MediaAttachment,
    attachment_mode: str,
) -> Any:
    content = [
        {
            "type": "input_text",
            "text": (
                "Read only the attached PDF. Translate the paper abstract into "
                "Chinese. If you cannot access or read the attached PDF, say so "
                "explicitly and briefly."
            ),
        },
        media_content_item(attachment, attachment_mode),
    ]
    kwargs = {
        "model": model,
        "input": [{"role": "user", "content": content}],
    }
    return client.responses.create(**kwargs)


def extract_response_text(response: Any) -> str:
    if isinstance(response, dict):
        output_text = response.get("output_text")
        if output_text:
            return str(output_text)
        choices = response.get("choices")
        if choices:
            return choices[0].get("message", {}).get("content") or "{}"
        output = response.get("output")
        if output:
            chunks: list[str] = []
            for item in output:
                for content in item.get("content", []) if isinstance(item, dict) else []:
                    if isinstance(content, dict) and content.get("text"):
                        chunks.append(str(content["text"]))
            if chunks:
                return "\n".join(chunks)
        return json.dumps(response, ensure_ascii=False)

    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    choices = getattr(response, "choices", None)
    if choices:
        return choices[0].message.content or "{}"

    output = getattr(response, "output", None)
    if output:
        chunks: list[str] = []
        for item in output:
            if isinstance(item, dict):
                contents = item.get("content", [])
            else:
                contents = getattr(item, "content", []) or []
            for content in contents:
                if isinstance(content, dict):
                    text = content.get("text")
                else:
                    text = getattr(content, "text", None)
                if text:
                    chunks.append(str(text))
        if chunks:
            return "\n".join(chunks)

    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        return json.dumps(dumped, ensure_ascii=False)
    return str(response)


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


def failed_review_record(target: TaskTarget, model: str, exc: BaseException) -> dict[str, Any]:
    return {
        "task_id": target.task_id,
        "model": model,
        "overall_status": "needs_major_rework",
        "benchmark_fit": "poor",
        "summary": f"Automated review failed: {type(exc).__name__}: {exc}",
        "reasoning": "The reviewer could not complete because the model call or file collection failed.",
        "tool_assessment": "Tool assessment was not produced because the automated review failed before completion.",
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
        "todo_items": ["Fix the review execution issue and rerun the workflow."],
    }


def normalize_record(target: TaskTarget, model: str, record: dict[str, Any]) -> dict[str, Any]:
    findings = record.get("findings") if isinstance(record.get("findings"), list) else []
    todo_items = record.get("todo_items")
    if not isinstance(todo_items, list):
        todo_items = derive_todos(findings)
    normalized = {
        "task_id": str(record.get("task_id") or target.task_id),
        "model": str(record.get("model") or model),
        "overall_status": str(record.get("overall_status") or "needs_major_rework"),
        "benchmark_fit": str(record.get("benchmark_fit") or "poor"),
        "summary": str(record.get("summary") or ""),
        "reasoning": str(record.get("reasoning") or ""),
        "tool_assessment": str(record.get("tool_assessment") or ""),
        "findings": findings,
        "merge_candidates": (
            record.get("merge_candidates")
            if isinstance(record.get("merge_candidates"), list)
            else []
        ),
        "todo_items": [str(item) for item in todo_items],
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
    if not normalized["tool_assessment"]:
        normalized["tool_assessment"] = derive_tool_assessment(record)
    return normalized


def derive_tool_assessment(record: dict[str, Any]) -> str:
    summary = str(record.get("summary") or "").strip()
    reasoning = str(record.get("reasoning") or "").strip()
    combined = " ".join(part for part in [summary, reasoning] if part)
    if combined:
        return (
            "No separate tool_assessment field was returned. Tool-related "
            f"context from the model response: {combined}"
        )
    return "No tool assessment was returned by the model."


def derive_todos(findings: list[dict[str, Any]]) -> list[str]:
    todos = []
    for finding in findings:
        title = str(finding.get("title") or "").strip()
        action = str(finding.get("recommended_action") or "").strip()
        if title and action:
            todos.append(f"{title}: {action}")
        elif action:
            todos.append(action)
    return todos or ["No blocking TODOs identified by this reviewer."]


def build_review_comment(
    pr_number: int,
    records: list[dict[str, Any]],
    targets: list[TaskTarget],
    smoke_results: list[PdfSmokeTestResult],
) -> str:
    lines = [
        REVIEW_COMMENT_MARKER,
        f"## MCP Tool Use Data Review for PR #{pr_number}",
        "",
        f"Reviewer script version: `{REVIEW_SCRIPT_VERSION}`",
        "",
        "| Task | Model | Target | Status | Fit | High | Medium | Low | Summary |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        counts = severity_counts(record.get("findings", []))
        target = record.get("_target", {})
        lines.append(
            "| {task} | `{model}` | `{kind}:{path}` | `{status}` | `{fit}` | {high} | {medium} | {low} | {summary} |".format(
                task=escape_md(record["task_id"]),
                model=escape_md(record["model"]),
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

    append_pdf_smoke_test_section(lines, smoke_results)

    for record in records:
        lines.extend(["", f"### {record['task_id']} / `{escape_md(record['model'])}`", ""])
        lines.append("**TODO**")
        todo_items = record.get("todo_items", [])
        if todo_items:
            for item in todo_items:
                lines.append(f"- {escape_md(str(item))}")
        else:
            lines.append("- No TODOs returned.")
        if record.get("tool_assessment"):
            lines.extend(
                [
                    "",
                    "**Tool Assessment**",
                    "",
                    escape_md(str(record["tool_assessment"])),
                ]
            )
        lines.extend(["", "<details>", "<summary>Reasoning and evidence</summary>", ""])
        if record.get("reasoning"):
            lines.extend(["**Reasoning**", "", escape_md(str(record["reasoning"])), ""])
        if record.get("summary"):
            lines.extend(["**Summary**", "", escape_md(str(record["summary"])), ""])
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

        merge_candidates = record.get("merge_candidates", [])
        if merge_candidates:
            lines.extend(["", "**Merge Candidates**"])
            for candidate in merge_candidates[:8]:
                lines.append(
                    "- `{action}` from `{source}` to `{target}`: {reason}".format(
                        action=escape_md(str(candidate.get("action", ""))),
                        source=escape_md(str(candidate.get("source_path", ""))),
                        target=escape_md(str(candidate.get("target_module", ""))),
                        reason=escape_md(str(candidate.get("reason", ""))),
                    )
                )
        lines.extend(["", "</details>"])

    if not targets:
        lines.append("")
        lines.append("No task targets were detected.")

    return "\n".join(lines)


def append_pdf_smoke_test_section(
    lines: list[str],
    smoke_results: list[PdfSmokeTestResult],
) -> None:
    if not smoke_results:
        return
    lines.extend(
        [
            "",
            "## Temporary PDF Attachment Smoke Tests",
            "",
            "This extra GPT model call is temporary and is not part of the data review score.",
        ]
    )
    for result in smoke_results:
        lines.extend(
            [
                "",
                f"### PDF smoke test: {escape_md(result.task_id)} / `{escape_md(result.model)}`",
                "",
                f"- Status: `{escape_md(result.status)}`",
                f"- Attachment mode: `{escape_md(result.attachment_mode)}`",
                f"- Target: `{escape_md(result.target_path)}`",
                f"- Attachment: `{escape_md(result.attachment_path or '(none)')}`",
                "",
                "<details>",
                "<summary>Translated abstract smoke-test output</summary>",
                "",
                "```text",
                trim_comment_text(result.output, 6_000),
                "```",
                "",
                "</details>",
            ]
        )


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


def trim_comment_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n...[truncated]..."


def post_comment(pr: Any, body: str) -> None:
    body = body[:60_000]
    pr.create_issue_comment(body)


if __name__ == "__main__":
    raise SystemExit(main())
