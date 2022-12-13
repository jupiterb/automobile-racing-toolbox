import sys
from racing_toolbox.interface.capturing.abstract import GameActionCapturing
if "linux" not in sys.platform:
    from racing_toolbox.interface.capturing.keyboard import KeyboardCapturing
    from racing_toolbox.interface.capturing.gamepad import GamepadCapturing
