import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from TexNetWebToolGPWrappers import TexNetWebToolLaunchHelper
from fsp_step4 import (
    SKIPPED_GEOMECHANICS_TERMINAL_MESSAGE,
    _add_hydrology_run_message,
    _geomechanics_steps_were_skipped,
)


def _write_args(tmp_path, step_states):
    args_path = tmp_path / "args.json"
    args_path.write_text(
        json.dumps({
            "SessionState": {
                "StepState": step_states,
            },
        }),
        encoding="utf-8",
    )


def test_is_step_skipped_reads_session_state_safely(tmp_path):
    _write_args(tmp_path, [
        {"IsSkipped": False},
        {"IsSkipped": True},
        {},
    ])
    helper = TexNetWebToolLaunchHelper(str(tmp_path))

    assert helper.isStepSkipped(0) is False
    assert helper.isStepSkipped(1) is True
    assert helper.isStepSkipped(2) is False
    assert helper.isStepSkipped(10) is False
    assert helper.isStepSkipped(-1) is False


class MessageHelper:
    def __init__(self, skipped_steps=None):
        self.skipped_steps = set(skipped_steps or [])
        self.messages = []

    def isStepSkipped(self, step_index):
        return step_index in self.skipped_steps

    def addMessageWithStepIndex(self, step_index, message_content, message_level):
        self.messages.append((step_index, message_content, message_level))


def test_skipped_geomechanics_uses_terminal_hydrology_message():
    helper = MessageHelper(skipped_steps={1, 2})

    geomechanics_steps_skipped = _geomechanics_steps_were_skipped(helper)
    _add_hydrology_run_message(
        helper,
        has_faults=False,
        geomechanics_steps_skipped=geomechanics_steps_skipped,
    )

    assert helper.messages == [
        (3, SKIPPED_GEOMECHANICS_TERMINAL_MESSAGE, 1),
    ]
    assert SKIPPED_GEOMECHANICS_TERMINAL_MESSAGE == (
        "Geomechanics steps have been skipped, so this is the last step of FSP for this run"
    )


def test_missing_fault_message_is_retained_without_skipped_geomechanics():
    helper = MessageHelper()

    _add_hydrology_run_message(
        helper,
        has_faults=False,
        geomechanics_steps_skipped=False,
    )

    assert helper.messages == [
        (
            3,
            "No fault dataset was provided, so deterministic hydrology fault pressure outputs were skipped. Pressure grid and well-based outputs are still available.",
            1,
        ),
    ]
