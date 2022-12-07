import sys 
from racing_toolbox.interface.controllers.abstract import GameActionController
if "linux" not in sys.platform:
    from racing_toolbox.interface.controllers.keyboard import KeyboardController
    from racing_toolbox.interface.controllers.gamepad import GamepadController
