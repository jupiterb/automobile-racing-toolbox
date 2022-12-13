import sys
from racing_toolbox.interface.screen.abstract import ScreenProvider
if "linux" not in sys.platform:
    from racing_toolbox.interface.screen.local import LocalScreen
