# 数据提交指南

## 提交流程

### 1. Fork 仓库

请先 fork 本仓库，并在自己的 fork 中准备任务数据。

### 2. 为每条数据新建分支

每条任务数据对应一个独立分支，建议分支名与 `task_id` 保持一致：

```bash
git checkout -b chem_example_0001
```

本仓库支持两种提交形式：

1. 提交 raw task 压缩包，例如：

```text
datasets/raw/chem/chem_example_0001.zip
```

2. 提交已解压的 raw task 目录，例如：

```text
datasets/raw/chem/chem_example_0001/
├── task_content/
│   └── task_content.json
├── tools/
├── input_data/
└── artifacts/
```

完成后推送分支：

```bash
git push origin chem_example_0001
```

### 3. 发起 Pull Request

在 GitHub 上从该分支向本仓库的 `main` 分支发起 PR。

**一条任务数据一个 PR**，不要在一个 PR 中混入多条任务。这样自动审查和人工审核都能围绕单条任务给出明确反馈。

---

## 自动审查

PR 创建或更新后，GitHub Actions 会自动运行 MCP task 数据审查流程。

自动审查会读取本次 PR 中新增或修改的 raw task zip / raw task 目录，并分别调用两个审查模型进行质量检查：

- `gpt-5.5`
- `claude-opus-4-7`

审查输入会包含任务目录中的文本文件内容；`input_data/reference_paper/target.pdf` 会作为多模态附件发送给模型，以便检查论文、图表和任务内容是否对应。图片和其他 PDF 不会作为多模态附件发送。

审查结果会以 comment 形式发布在 PR 页面。每个模型会单独给出一份结果：

- `TODO` 部分会直接展开显示，提交者应优先阅读并修复
- `Reasoning and evidence` 部分会折叠显示，用于查看详细理由、证据路径和工具/数据问题

**请在等待人工审核之前，先仔细阅读自动审查报告，自行排查并修复问题。**

> 注意：GitHub Actions 流程成功完成不代表数据质量合格。该流程的职责是发布自动审查意见；是否合并仍取决于人工审核。

---

## 目录结构

本仓库按领域组织 raw task，目前包含 10 个领域目录：

```text
datasets/raw/
├── astro/
├── chem/
├── energy/
├── geo/
├── info/
├── life/
├── mat/
├── math/
├── neuro/
└── phys/
```

建议每条 raw task 放在对应领域目录下，使用如下结构：

```text
datasets/raw/<domain>/<task_id>/
├── task_content/
│   └── task_content.json          # 任务 instruction、input_data、answer/checklist 等
├── tools/                         # MCP/domain-specific tools
├── input_data/                    # 任务所需输入数据，可包含 data/、reference_paper/ 等
└── artifacts/                     # 参考图表或标准产物（如任务需要）
```

如果提交 zip 包，zip 解压后也应能看到等价结构：

```text
<task_id>/
├── task_content/
│   └── task_content.json
├── tools/
├── input_data/
└── artifacts/
```

例如现有 task `astro_f2b1` 的结构包含：

```text
astro_f2b1/
├── task_content/task_content.json
├── tools/astro_tools.py
├── tools/data_processing.py
├── tools/database_retrieval.py
├── input_data/data/...
├── input_data/reference_paper/...
└── artifacts/Fig*.png
```

---

## 审查关注点

自动审查主要关注以下问题：

- 任务指令是否清晰，是否与参考答案/checklist 对齐
- 发布的数据是否足以支持任务要求的分析和结论
- `tools/` 中的领域工具是否能支撑核心工作流
- 工具是否过于 one-shot、硬编码最终答案或不可泛化
- checklist 是否覆盖核心科学结论，而不是奖励无关产物
- 是否存在明显路径、字段、单位、数据集、样本/队列名称不一致

自动审查不会把 `task_content.json` 中包含参考答案或 checklist 本身作为问题，因为评测时模型只会看到模型可见的 instruction/ask，而不会看到完整任务目录。
