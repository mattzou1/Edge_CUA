"""Tests for controller.py -- VNC action dispatch."""

import sys
import os
import pytest
from unittest.mock import MagicMock, call

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# pyautogui connects to X at import time -- stub it before the controller loads
sys.modules["pyautogui"] = MagicMock()

from ecua2_agent.controller_module.controller import Controller, ControllerError


def test_click_dispatches_move_and_press():
    client = MagicMock()
    ctrl = Controller(vnc_client=client)
    result = ctrl.execute_action_vnc("CLICK 640 400")
    assert result == "OK"
    client.mouseMove.assert_called_once_with(640, 400)
    client.mousePress.assert_called_once_with(1)


def test_execute_action_vnc_raises_without_client():
    ctrl = Controller()
    with pytest.raises(ControllerError, match="No VNC client"):
        ctrl.execute_action_vnc("CLICK 100 200")


def test_typing_strips_quotes():
    client = MagicMock()
    ctrl = Controller(vnc_client=client)
    result = ctrl.execute_action_vnc('TYPING "hello world"')
    assert result == "OK"
    client.type.assert_called_once_with("hello world")


def test_hotkey_replaces_plus_with_hyphen():
    client = MagicMock()
    ctrl = Controller(vnc_client=client)
    ctrl.execute_action_vnc("HOTKEY ctrl+t")
    client.keyPress.assert_called_once_with("ctrl-t")


def test_done_returns_done():
    client = MagicMock()
    ctrl = Controller(vnc_client=client)
    assert ctrl.execute_action_vnc("DONE") == "DONE"


def test_fail_returns_fail():
    client = MagicMock()
    ctrl = Controller(vnc_client=client)
    assert ctrl.execute_action_vnc("FAIL") == "FAIL"


def test_unknown_verb_returns_error():
    client = MagicMock()
    ctrl = Controller(vnc_client=client)
    result = ctrl.execute_action_vnc("EXPLODE 1 2 3")
    assert result.startswith("ERROR")


def test_scroll_up_uses_button_4():
    client = MagicMock()
    ctrl = Controller(vnc_client=client)
    ctrl.execute_action_vnc("SCROLL 0 3")
    assert client.mousePress.call_args_list == [call(4), call(4), call(4)]


def test_empty_action_returns_ok():
    client = MagicMock()
    ctrl = Controller(vnc_client=client)
    assert ctrl.execute_action_vnc("") == "OK"
    assert ctrl.execute_action_vnc("# comment") == "OK"
