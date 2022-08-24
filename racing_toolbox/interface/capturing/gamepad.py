import vgamepad as vg

import pygame
from pygame.joystick import Joystick

import threading

from interface.capturing.abstract import GameActionCapturing
from interface.models import SteeringAction, GamepadAction
from interface.models.gamepad_action import GamepadControl
from interface.exceptions import JoystickNotFound


clock_frequency = 60

pygame_to_vg_button = {
    pygame.CONTROLLER_BUTTON_A: vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
    pygame.CONTROLLER_BUTTON_B: vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
    pygame.CONTROLLER_BUTTON_X: vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
    pygame.CONTROLLER_BUTTON_Y: vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
    pygame.CONTROLLER_AXIS_TRIGGERLEFT: vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB,
    pygame.CONTROLLER_AXIS_TRIGGERRIGHT: vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB,
}

pygame_to_gamepad_control = {
    pygame.CONTROLLER_AXIS_LEFTX: GamepadControl.LEFT_JOYSTICK_X,
    pygame.CONTROLLER_AXIS_LEFTY: GamepadControl.LEFT_JOYSTICK_Y,
    pygame.CONTROLLER_AXIS_RIGHTX: GamepadControl.RIGHT_JOYSTICK_X,
    pygame.CONTROLLER_AXIS_RIGHTY: GamepadControl.RIGHT_JOYSTICK_Y,
    pygame.CONTROLLER_AXIS_TRIGGERLEFT: GamepadControl.LEFT_TRIGGER,
    pygame.CONTROLLER_AXIS_TRIGGERRIGHT: GamepadControl.RIGHT_TRIGGER,
}


class GamepadCapturing(GameActionCapturing):
    def __init__(
        self, gamepad_action_mapping: dict[GamepadAction, SteeringAction]
    ) -> None:
        self._gamepad_action_mapping = gamepad_action_mapping
        self._actions = {action: 0.0 for action in list(SteeringAction)}
        self._keep_capturing = False
        self._listener: threading.Thread = threading.Thread(
            target=self._listen, args=()
        )

    def start(self) -> None:
        self._keep_capturing = True
        self._listener.start()

    def stop(self) -> None:
        for action in self._actions:
            self._actions[action] = 0.0
        self._keep_capturing = False
        try:
            self._listener.join()
        except RuntimeError:
            pass

    def get_captured(self) -> dict[SteeringAction, float]:
        return self._actions

    def _listen(self) -> None:
        joysticks = GamepadCapturing._init_joysticks()
        clock = pygame.time.Clock()
        while self._keep_capturing:
            clock.tick(clock_frequency)
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONUP:
                    self._handle_button_event(event, False)
                elif event.type == pygame.JOYBUTTONDOWN:
                    self._handle_button_event(event, True)
                elif event.type == pygame.JOYAXISMOTION:
                    self._handle_axis_event(event)
        GamepadCapturing._del_joysticks(joysticks)

    @staticmethod
    def _init_joysticks() -> list[Joystick]:
        pygame.init()
        pygame.joystick.init()
        count = pygame.joystick.get_count()
        joysticks = []
        if count:
            for i in range(count):
                joysticks.append(pygame.joystick.Joystick(i))
                joysticks[-1].init()
        else:
            raise JoystickNotFound
        return joysticks

    @staticmethod
    def _del_joysticks(joysticks: list[Joystick]) -> None:
        for joystick in joysticks:
            del joystick

    def _handle_button_event(self, event: pygame.event.Event, pressed: bool):
        try:
            button = pygame_to_vg_button[event.button]
        except KeyError:
            return
        if button in self._gamepad_action_mapping:
            action = self._gamepad_action_mapping[button]
            self._actions[action] = 1.0 if pressed else 0.0

    def _handle_axis_event(self, event: pygame.event.Event):
        try:
            control = pygame_to_gamepad_control[event.axis]
        except KeyError:
            return
        if control in self._gamepad_action_mapping:
            action = self._gamepad_action_mapping[control]
            self._actions[action] = event.value
