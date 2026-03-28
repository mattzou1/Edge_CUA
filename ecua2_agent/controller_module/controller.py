#!/usr/bin/env python3

import time
import shlex
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class ControllerError(Exception):
    """Custom exception for controller errors."""
    pass


class Controller:
    def __init__(self, default_wait: float = 0.2, vnc_client=None):
        self.default_wait = default_wait
        self._vnc_client = vnc_client

    # ------------------------------------------------------------------
    # VNC mode (primary path for the product)
    # ------------------------------------------------------------------

    def execute_action_vnc(self, action: str) -> str:
        """
        Execute action via vncdotool VNC client.
        Requires vnc_client= passed to __init__.

        Supported tokens: CLICK, DOUBLE_CLICK, MOVE_TO, TYPING, PRESS,
                          HOTKEY, SCROLL, WAIT, DONE, FAIL
        """
        if self._vnc_client is None:
            raise ControllerError(
                "No VNC client. Instantiate Controller(vnc_client=client)."
            )

        action = action.strip()
        if not action or action.startswith("#"):
            return "OK"

        try:
            parts = shlex.split(action)
        except ValueError as e:
            return f"ERROR: shlex parse failed: {e}"

        if not parts:
            return "OK"

        verb = parts[0].upper()
        args = parts[1:]

        try:
            if verb == "CLICK":
                x, y = int(args[0]), int(args[1])
                self._vnc_client.mouseMove(x, y)
                self._vnc_client.mousePress(1)

            elif verb == "DOUBLE_CLICK":
                x, y = int(args[0]), int(args[1])
                self._vnc_client.mouseMove(x, y)
                self._vnc_client.mousePress(1)
                time.sleep(0.08)
                self._vnc_client.mousePress(1)

            elif verb == "MOVE_TO":
                x, y = int(args[0]), int(args[1])
                self._vnc_client.mouseMove(x, y)

            elif verb == "RIGHT_CLICK":
                x, y = int(args[0]), int(args[1])
                self._vnc_client.mouseMove(x, y)
                self._vnc_client.mousePress(3)

            elif verb == "TYPING":
                text = " ".join(args).strip("\"'")
                self._vnc_client.type(text)

            elif verb == "PRESS":
                if not args:
                    raise ControllerError("PRESS requires a key name")
                self._vnc_client.keyPress(args[0])

            elif verb == "HOTKEY":
                if not args:
                    raise ControllerError("HOTKEY requires keys")
                # "ctrl+alt+t" or ["ctrl", "alt", "t"] -> "ctrl-alt-t"
                combo = args[0] if len(args) == 1 else "+".join(args)
                self._vnc_client.keyPress(combo.replace("+", "-"))

            elif verb == "KEY_DOWN":
                if not args:
                    raise ControllerError("KEY_DOWN requires a key name")
                self._vnc_client.keyDown(args[0])

            elif verb == "KEY_UP":
                if not args:
                    raise ControllerError("KEY_UP requires a key name")
                self._vnc_client.keyUp(args[0])

            elif verb == "SCROLL":
                dx, dy = int(args[0]), int(args[1])
                # VNC button 4 = scroll up, button 5 = scroll down
                if dy != 0:
                    button = 4 if dy > 0 else 5
                    for _ in range(abs(dy)):
                        self._vnc_client.mousePress(button)
                # Horizontal scroll: not natively supported by VNC protocol; ignored.

            elif verb == "WAIT":
                # Cap at 30s -- LLM hallucinations like WAIT 9999 would stall the loop.
                secs = min(float(args[0]) if args else 1.0, 30.0)
                time.sleep(secs)

            elif verb in ("DONE", "FAIL"):
                return verb

            else:
                raise ControllerError(f"Unknown VNC action: {verb}")

        except (IndexError, ValueError) as e:
            return f"ERROR: bad arguments for {verb}: {e}"
        except ControllerError as e:
            return f"ERROR: {e}"
        except Exception as e:
            return f"ERROR: {e}"

        return "OK"

    # ------------------------------------------------------------------
    # pyautogui mode (host-side, kept for local testing)
    # ------------------------------------------------------------------

    def execute_action(self, comd: str) -> str:
        comd = comd.strip()
        if not comd or comd.startswith("#"):
            return "Not Valid"

        action, args = self._parse_line(comd)
        action = action.upper()

        logger.info("Executing: %s %s", action, " ".join(args))

        if action == "MOVE_TO":
            self._move_to(args)
        elif action == "CLICK":
            self._click(args)
        elif action == "MOUSE_DOWN":
            self._mouse_down(args)
        elif action == "MOUSE_UP":
            self._mouse_up(args)
        elif action == "RIGHT_CLICK":
            self._right_click(args)
        elif action == "DOUBLE_CLICK":
            self._double_click(args)
        elif action == "DRAG_TO":
            self._drag_to(args)
        elif action == "SCROLL":
            self._scroll(args)
        elif action == "TYPING":
            self._typing(args)
        elif action == "PRESS":
            self._press(args)
        elif action == "KEY_DOWN":
            self._key_down(args)
        elif action == "KEY_UP":
            self._key_up(args)
        elif action == "HOTKEY":
            self._hotkey(args)
        elif action == "WAIT":
            self._wait(args)
        elif action == "FAIL":
            logger.warning("FAIL action received")
            return "FAIL"
        elif action == "DONE":
            logger.info("DONE action received")
            return "DONE"
        else:
            raise ControllerError(f"Unknown action: {action}")

        return "OK"

    def execute_actions(self, lines: List[str]) -> str:
        for line in lines:
            status = self.execute_action(line)
            if status in ("FAIL", "DONE"):
                return status
        return "OK"

    # ------------------------------------------------------------------
    # pyautogui internals
    # ------------------------------------------------------------------

    @staticmethod
    def _get_pyautogui():
        import pyautogui  # noqa: PLC0415 — lazy import, requires a display
        pyautogui.FAILSAFE = True
        return pyautogui

    @staticmethod
    def _parse_line(line: str) -> Tuple[str, List[str]]:
        tokens = shlex.split(line)
        if not tokens:
            raise ControllerError("Empty command line")
        return tokens[0], tokens[1:]

    @staticmethod
    def _parse_button(arg: Optional[str]) -> str:
        if not arg:
            return "left"
        b = arg.lower()
        if b in ("left", "right", "middle"):
            return b
        raise ControllerError(f"Invalid mouse button: {arg}")

    @staticmethod
    def _parse_int_pair(args: List[str], offset: int = 0) -> Tuple[int, int]:
        try:
            return int(args[offset]), int(args[offset + 1])
        except (IndexError, ValueError):
            raise ControllerError(
                f"Expected two integers at positions {offset},{offset+1}, got: {args}"
            )

    def _move_to(self, args):
        if len(args) != 2:
            raise ControllerError("MOVE_TO requires exactly 2 arguments: x y")
        x, y = self._parse_int_pair(args)
        self._get_pyautogui().moveTo(x, y)

    def _click(self, args):
        pg = self._get_pyautogui()
        button = "left"
        x = y = None
        clicks = 1
        idx = 0
        if idx < len(args) and args[idx].lower() in ("left", "right", "middle"):
            button = self._parse_button(args[idx])
            idx += 1
        if idx + 1 < len(args):
            try:
                x, y = int(args[idx]), int(args[idx + 1])
                idx += 2
            except ValueError:
                pass
        if idx < len(args):
            try:
                clicks = int(args[idx])
            except ValueError:
                raise ControllerError(f"Invalid num_clicks: {args[idx]}")
        if x is None or y is None:
            pg.click(button=button, clicks=clicks)
        else:
            pg.click(x=x, y=y, button=button, clicks=clicks)

    def _mouse_down(self, args):
        self._get_pyautogui().mouseDown(button=self._parse_button(args[0] if args else None))

    def _mouse_up(self, args):
        self._get_pyautogui().mouseUp(button=self._parse_button(args[0] if args else None))

    def _right_click(self, args):
        pg = self._get_pyautogui()
        x = y = None
        if len(args) >= 2:
            x, y = self._parse_int_pair(args)
        if x is None:
            pg.click(button="right")
        else:
            pg.click(x=x, y=y, button="right")

    def _double_click(self, args):
        pg = self._get_pyautogui()
        x = y = None
        if len(args) >= 2:
            x, y = self._parse_int_pair(args)
        if x is None:
            pg.click(clicks=2)
        else:
            pg.click(x=x, y=y, clicks=2)

    def _drag_to(self, args):
        if len(args) != 2:
            raise ControllerError("DRAG_TO requires exactly 2 arguments: x y")
        x, y = self._parse_int_pair(args)
        self._get_pyautogui().dragTo(x, y, button="left")

    def _scroll(self, args):
        if len(args) != 2:
            raise ControllerError("SCROLL requires exactly 2 arguments: dx dy")
        pg = self._get_pyautogui()
        dx, dy = self._parse_int_pair(args)
        if dy != 0:
            pg.scroll(dy)
        if dx != 0 and hasattr(pg, "hscroll"):
            pg.hscroll(dx)

    def _typing(self, args):
        if not args:
            raise ControllerError("TYPING requires a text argument")
        self._get_pyautogui().typewrite(" ".join(args))

    def _press(self, args):
        if len(args) != 1:
            raise ControllerError("PRESS requires exactly 1 key")
        self._get_pyautogui().press(args[0])

    def _key_down(self, args):
        if len(args) != 1:
            raise ControllerError("KEY_DOWN requires exactly 1 key")
        self._get_pyautogui().keyDown(args[0])

    def _key_up(self, args):
        if len(args) != 1:
            raise ControllerError("KEY_UP requires exactly 1 key")
        self._get_pyautogui().keyUp(args[0])

    def _hotkey(self, args):
        if not args:
            raise ControllerError("HOTKEY requires at least one key")
        pg = self._get_pyautogui()
        keys = args[0].split("+") if len(args) == 1 else args
        keys = [k.strip() for k in keys if k.strip()]
        if not keys:
            raise ControllerError("HOTKEY parsed empty key list")
        pg.hotkey(*keys)

    def _wait(self, args):
        if args:
            try:
                time.sleep(float(args[0]))
            except ValueError:
                raise ControllerError(f"Invalid WAIT time: {args[0]}")
        else:
            time.sleep(self.default_wait)
