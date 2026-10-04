from __future__ import annotations

from environments.synthetic_fs.common import (
    ANSWER_PREFIX,
    READ_FILE_MAX_LINES,
    STOP_MESSAGE,
    AnswererWorkspaceMode,
)


BUILDER_SYSTEM_PROMPT = f"""You are building a synthetic filesystem for a later answering agent.

Use the provided filesystem tools to inspect files under:
- raw_docs/
- synthetic_fs/
- history/

Your job is to leave behind a final active synthetic filesystem that helps later agents answer as many plausible future questions about this corpus as possible.
Do not optimize for one guessed target question; build broad, reusable knowledge structure over the corpus.
The later answering agent will start with root-level synthetic entries, browse deeper synthetic files if needed, and use raw_docs/ only as fallback.

In this pipeline, a cluster is one synthetic memory bank for one coherent family of plausible future information needs.
Choose cluster boundaries by downstream retrieval need:
- group facts together when a later answerer would likely want to retrieve them together
- split facts apart when they would support different kinds of future questions
- do not assume a fixed template; let the evidence determine the right cluster shape and scope

Use the tools as follows:
- list_files and read_file inspect the available files; use list_files on paths like raw_docs/ or synthetic_fs/ to browse incrementally
- create_cluster creates exactly one new synthetic file from selected inputs
- merge_clusters groups similar or related active synthetic files under a parent summary; do not merge arbitrary unrelated files
- delete removes active synthetic_fs/ files from the final active view

Important rules:
- only call read_file on exact paths returned by list_files or prior successful tool results; never guess raw_docs/ paths from memory or URL names
- if read_file says a raw document is not visible, do not retry that path; call list_files("raw_docs") and continue with only listed files
- you may include brief reasoning when it helps choose the next action, but keep it short and action-oriented
- avoid long planning monologues, numbered lists of possible next steps, or repeated status summaries
- after reasoning, take a concrete tool action unless you are returning STOP
- use only information supported by the files
- do not produce final answers directly; your job is to build useful synthetic files
- make each cluster a coherent memory bank rather than a loose dump of unrelated facts
- if two facts would likely be retrieved for different purposes, put them in different clusters
- if several clusters belong under a broader theme, create the local clusters first and merge them later
- do not wait to read everything before synthesizing; once you have enough evidence for one coherent memory bank, create it
- work in short synthesis cycles: after a few read/list calls, create or merge a durable synthetic file
- avoid long stretches of read/list-only turns; extra reading is useful only when it leads to a new or improved synthetic file
- returning STOP without creating any synthetic file is usually a failure
- a good early success is to create one focused cluster from a small set of closely related files, then refine or merge later if needed
- when files are long, read only the chunk you need instead of trying to ingest the whole document at once
- preserve salient facts, names, dates, places, relationships, aliases, titles, and other high-signal details that would help with many plausible questions about these documents
- prefer many focused leaf clusters organized under a smaller number of parent merge summaries
- root-level entries should be useful navigation summaries, not a flat dump of every leaf cluster
- use delete only on active synthetic_fs/ paths; never pass raw_docs/ or history/ to delete
- if a raw document is irrelevant, ignore it instead of trying to delete it
- if the final synthetic filesystem has broad coverage, many focused leaf clusters, and useful merge-summary structure, stop immediately
- do not spend the full turn budget just because more turns are available; stop proactively when the filesystem is mature

When you are ready to finish, your entire final assistant message must be exactly one non-empty line:
{STOP_MESSAGE}

Do not emit <think> tags or long hidden-reasoning dumps.
Outside tool calls, keep any free text short and directly tied to the next action or STOP decision.
"""


def build_builder_system_prompt(
    *,
    executor_enabled: bool,
    batch_tools_enabled: bool,
    coverage_first: bool = False,
    coverage_leaf_max_browse_turns: int = 0,
    task_question: str | None = None,
) -> str:
    prompt = BUILDER_SYSTEM_PROMPT
    if executor_enabled:
        prompt += """

Executor mode is active:
- you are the planner/controller, not the long-form cluster writer
- create_cluster and merge_clusters call a frozen executor model to write the synthetic file content
- provide concise titles, exact source paths/handles, and short generation_instructions describing the intended scope
- do not spend tokens writing full cluster_text or merge_summary_text; the executor writes those from the selected source files
- the executor may only use the files you pass as sources, so source selection and grouping remain your responsibility
"""
    if batch_tools_enabled:
        prompt += """

Batched tools are available:
- read_many reads several exact paths in one tool call
- create_clusters creates several independent leaf clusters in one tool call
- merge_many creates several parent summaries in one tool call
Use batched tools when the operations are independent and already well-scoped. Prefer batching to avoid long read/read/create loops, but keep each operation coherent.
"""
    if coverage_first:
        prompt += """

Coverage-first construction condition:
- this condition overrides the earlier guidance to stop proactively or avoid reading every document
- systematically enumerate visible raw_docs/ paths and build traceable coverage for every visible document before STOP
- each raw document should be represented in the provenance of at least one focused active leaf cluster; use batched reads and batched leaf creation to make this practical
- organize coverage into coherent leaves and parent navigation summaries rather than a single dump
- do not treat an early mature-looking tree as sufficient while visible source documents remain uncovered
- this requirement is corpus-wide and question-blind: do not infer or optimize for any particular future question
"""
        if coverage_leaf_max_browse_turns:
            prompt += (
                "\nConstruction validity rule: while source coverage is incomplete, "
                f"do not take more than {coverage_leaf_max_browse_turns} consecutive "
                "browse-only turns without creating an active leaf with direct raw-source provenance. "
                "A violation invalidates the construction.\n"
            )
    if task_question is not None:
        prompt += (
            "\nFew-shot task-adaptation condition:\n"
            "- the following is a frozen seed-task prompt, not a future-question guess\n"
            "- prioritize source selection and synthetic files that will let a later answerer resolve this task\n"
            "- preserve direct raw-source provenance for every task-relevant claim\n"
            "- do not answer the task yourself; construct the evidence workspace for the later answerer\n"
            "- this task-specific instruction overrides the generic corpus-wide/question-blind guidance above\n\n"
            f"Frozen seed-task prompt:\n{task_question}\n"
        )
    return prompt


def build_builder_user_prompt(
    *,
    executor_enabled: bool,
    batch_tools_enabled: bool,
    coverage_first: bool = False,
    coverage_leaf_max_browse_turns: int = 0,
    task_question: str | None = None,
) -> str:
    coverage_instruction = ""
    if coverage_first:
        coverage_instruction = (
            "This is a coverage-first construction condition: systematically enumerate raw_docs/ and, "
            "before STOP, ensure every visible raw document has traceable provenance in at least one "
            "focused active leaf. Use batch tools for coverage, then organize the resulting leaves into a navigable tree.\\n"
        )
        if coverage_leaf_max_browse_turns:
            coverage_instruction += (
                f"After at most {coverage_leaf_max_browse_turns} browse-only turns with incomplete coverage, "
                "your next valid action must create a focused provenance-bearing leaf.\n"
            )
    if task_question is not None:
        inspect_tools = "list_files/read_many" if batch_tools_enabled else "list_files/read_file"
        return (
            "Use the filesystem tools as a planner/controller. Build a compact, source-grounded "
            "workspace for the frozen seed task below. Prioritize the evidence that resolves it, "
            "with coherent task-relevant clusters and direct source provenance. Do not answer it directly. "
            "Start by listing raw_docs/, then read relevant sources and create at least one focused "
            "provenance-bearing cluster before STOP. After at most two browse-only tool calls, your "
            "next tool call must create a focused cluster from the evidence already inspected; do not "
            "keep browsing for exhaustive coverage.\n\n"
            f"Frozen seed-task prompt:\n{task_question}\n\n"
            f"Raw documents are available under raw_docs/. Use {inspect_tools} to inspect only relevant evidence.\n"
            f"When the workspace is sufficient for a later answerer, return exactly one line: {STOP_MESSAGE}"
        )
    if executor_enabled:
        inspect_tools = "list_files/read_many" if batch_tools_enabled else "list_files/read_file"
        construction_cycle = (
            "batched reads, batched focused leaf-cluster creation"
            if batch_tools_enabled
            else "focused reads and focused leaf-cluster creation"
        )
        if coverage_first:
            finish_guidance = (
                f"Use {construction_cycle} until every visible raw document has traceable provenance in a focused active leaf, "
                "then merge related leaves into parent summaries and STOP.\n"
            )
        else:
            finish_guidance = (
                f"Prefer cycles of {construction_cycle}, merge related clusters into parent summaries, then STOP once the filesystem is mature.\n"
            )
        return (
            "Use the filesystem tools as a planner/controller. Inspect useful files, then propose concise operations; "
            "the frozen executor will write cluster and merge content from your selected sources.\n"
            "Build a broad, tree-like synthetic filesystem of reusable memory banks that will help later answering agents answer as many plausible future questions as possible.\n"
            f"Raw documents are available under raw_docs/. Use {inspect_tools} to inspect only what is needed to choose good sources and grouping.\n"
            f"{coverage_instruction}"
            f"{finish_guidance}"
            f"When you are done, return exactly one line: {STOP_MESSAGE}"
        )
    inspect_tools = "list_files/read_many" if batch_tools_enabled else "list_files/read_file"
    if coverage_first:
        stop_guidance = (
            "Continue systematic coverage until every visible raw document has traceable provenance in a focused active leaf; "
            "then merge related leaves into parent summaries and STOP.\n"
        )
    else:
        stop_guidance = (
            "Do not treat reading every raw document as a goal; the goal is the final synthetic filesystem. "
            "Prefer cycles of read a few files, create focused leaf clusters, merge related clusters into parent summaries, then STOP once the filesystem is mature.\n"
        )
    return (
        "Use the filesystem tools to inspect whichever files you think are useful. "
        "Build a broad, tree-like synthetic filesystem of reusable memory banks that will help later answering agents answer as many plausible future questions as possible.\n"
        f"Raw documents are available under raw_docs/. Use {inspect_tools} to inspect only the files needed to build useful synthetic memory banks.\n"
        f"{coverage_instruction}"
        f"{stop_guidance}"
        f"When you are done, return exactly one line: {STOP_MESSAGE}"
    )


def build_answerer_system_prompt(mode: AnswererWorkspaceMode) -> str:
    workspace_lines = [
        "You are the answering agent for a read-only workspace.",
        "",
        "You have filesystem tools over a final workspace that contains:",
        "- README.md explaining the workspace conventions and navigation only",
        "- root-level synthetic entries that are already loaded into context as the primary entry points",
        "- nested synthetic folders and summary files for more detail",
    ]
    if mode == "synthetic_plus_raw":
        workspace_lines.append(
            "- raw_docs/ as a strict fallback exposed only through synthetic-file provenance"
        )

    order_lines = [
        "",
        "Use this order:",
        "1. start from the already-loaded README.md and root-level synthetic entries",
        "2. browse deeper synthetic files if needed",
    ]
    if mode == "synthetic_plus_raw":
        order_lines.append("3. use raw_docs/ only if the synthetic files still lack a decisive detail")
        guidance = (
            "Do not guess. If the synthetic files you have read so far do not contain enough evidence to "
            "uniquely identify the answer, continue exploring. If the synthetic filesystem is still "
            "insufficient, follow provenance to raw_docs/ as a fallback before answering."
        )
    else:
        guidance = (
            "Do not guess. If the synthetic files you have read so far do not contain enough evidence to "
            "uniquely identify the answer, continue exploring the synthetic filesystem before answering. "
            "README.md only explains the workspace and is not sufficient evidence by itself. "
            "Answer only when the synthetic workspace contains enough evidence."
        )

    footer = [
        "",
        guidance,
        "",
        "At each step, either:",
        "- return exactly one compact JSON action:",
        '  {"action":"list_files","path":"."}',
        f'  {{"action":"read_file","path":"some/path.txt","start_line":1,"num_lines":{READ_FILE_MAX_LINES}}}',
        "- or return exactly one non-empty line:",
        f"  {ANSWER_PREFIX} <final answer>",
        "",
        "Do not include explanation outside the JSON action or final answer line.",
    ]
    return "\n".join(workspace_lines + order_lines + footer)


def build_answerer_user_prompt(question: str, mode: AnswererWorkspaceMode) -> str:
    guidance = (
        "Answer from the synthetic filesystem if possible; use raw_docs/ only as provenance-linked fallback."
        if mode == "synthetic_plus_raw"
        else "Answer using only the synthetic workspace."
    )
    return (
        f"Question: {question}\n\n"
        "Use the workspace tools to inspect the final workspace. "
        "README.md and the current root-level synthetic entries are already loaded into context. "
        + guidance
    )
