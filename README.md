# MCP Tool Use Dataset

This repository is intended to receive MCP/ScienceToolBench task data through
GitHub pull requests.

When a PR is opened or updated, `.github/workflows/ai_review.yml` runs
`scripts/review_pr.py`. The reviewer reads changed raw task zip files or task
directories containing `task_content/task_content.json`, asks the configured LLM
for a structured data-quality review, and posts the findings as a PR comment.

Required repository settings:

- `LLM_API_KEY` secret
- `LLM_BASE_URL` secret, if using a non-default OpenAI-compatible endpoint
- optional `LLM_MODEL` repository variable, defaulting to `gpt-5.4`
