from __future__ import annotations

import json
from typing import Any

from agent_harness.models import Provider

CLAUDE_NATIVE_INPUT_SETTING_PREFIX = "claude_native_input_request."


def claude_native_input_setting_key(session_id: str, tool_use_id: str) -> str:
    return f"{CLAUDE_NATIVE_INPUT_SETTING_PREFIX}{Provider.CLAUDE.value}.{session_id}.{tool_use_id}"


def claude_ask_user_question_tool_use(message: dict[str, Any]) -> dict[str, Any] | None:
    content = message.get("content")
    if not isinstance(content, list):
        return None
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "tool_use" and item.get("name") == "AskUserQuestion":
            return item
    return None


def claude_ask_user_question_id(message: dict[str, Any]) -> str | None:
    item = claude_ask_user_question_tool_use(message)
    if item is None:
        return None
    tool_use_id = item.get("id")
    return tool_use_id if isinstance(tool_use_id, str) and tool_use_id.strip() else None


def slack_request_params_for_claude_ask_user_question(
    tool_input: dict[str, Any],
) -> dict[str, Any] | None:
    questions = _questions(tool_input)
    if not questions:
        return None
    return {
        "questions": [
            {
                "id": _question_id(index),
                "header": _string_field(question, "header") or "Question",
                "question": _string_field(question, "question"),
                "options": _options(question.get("options")),
            }
            for index, question in enumerate(questions)
        ]
    }


def claude_ask_user_question_updated_input(
    tool_input: dict[str, Any],
    slack_response: object,
) -> dict[str, Any] | None:
    answers = claude_ask_user_question_answers(tool_input, slack_response)
    if answers is None:
        return None
    updated = dict(tool_input)
    updated["answers"] = answers
    return updated


def claude_ask_user_question_answers(
    tool_input: dict[str, Any],
    slack_response: object,
) -> dict[str, str | list[str]] | None:
    if not isinstance(slack_response, dict):
        return None
    response_answers = slack_response.get("answers")
    if not isinstance(response_answers, dict):
        return None
    questions = _questions(tool_input)
    if not questions:
        return None
    answers: dict[str, str | list[str]] = {}
    for index, question in enumerate(questions):
        question_key = _question_answer_key(question, index)
        selected = _selected_answers(response_answers, _question_id(index))
        if not selected:
            return None
        if question.get("multiSelect") is True:
            answers[question_key] = selected
        else:
            answers[question_key] = selected[0]
    return answers


def claude_ask_user_question_tool_result_text(
    tool_input: dict[str, Any],
    answers: dict[str, str | list[str]],
) -> str:
    questions = _questions(tool_input)
    pairs: list[str] = []
    for index, question in enumerate(questions):
        question_key = _question_answer_key(question, index)
        answer = answers.get(question_key)
        value = ", ".join(answer) if isinstance(answer, list) else str(answer or "")
        if value:
            pairs.append(f"{json.dumps(question_key)}={json.dumps(value)}")
    if not pairs:
        return "No answer was selected."
    return (
        "Your questions have been answered: "
        + ", ".join(pairs)
        + ". You can now continue with these answers in mind."
    )


def _questions(tool_input: dict[str, Any]) -> list[dict[str, Any]]:
    questions = tool_input.get("questions")
    if not isinstance(questions, list):
        return []
    return [question for question in questions if isinstance(question, dict)]


def _question_id(index: int) -> str:
    return f"q{index}"


def _question_answer_key(question: dict[str, Any], index: int) -> str:
    question_text = _string_field(question, "question")
    if question_text:
        return question_text
    header = _string_field(question, "header")
    return header or _question_id(index)


def _selected_answers(response_answers: dict[str, Any], question_id: str) -> list[str]:
    value = response_answers.get(question_id)
    if not isinstance(value, dict):
        return []
    raw_answers = value.get("answers")
    if not isinstance(raw_answers, list):
        return []
    return [answer for answer in (str(item).strip() for item in raw_answers) if answer]


def _options(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    options: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str):
            label = item.strip()
            if label:
                options.append({"label": label})
            continue
        if not isinstance(item, dict):
            continue
        label = _string_field(item, "label")
        if not label:
            continue
        option = {"label": label}
        description = _string_field(item, "description")
        if description:
            option["description"] = description
        options.append(option)
    return options


def _string_field(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    return item.strip() if isinstance(item, str) else ""
