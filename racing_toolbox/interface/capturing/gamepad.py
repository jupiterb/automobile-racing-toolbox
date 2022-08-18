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
        self._joysticks: list[Joystick] = []
        self._actions = {action: 0.0 for action in list(SteeringAction)}
        self._keep_capturing = False
        self._listener: threading.Thread = threading.Thread(
            target=self._listen, args=()
        )
        pygame.init()

    def start(self) -> None:
        count = pygame.joystick.get_count()
        if count:
            for i in range(0, count):
                pygame.joystick.Joystick(i).init()
        else:
            raise JoystickNotFound
        self._keep_capturing = True
        self._listener.start()

    def stop(self) -> None:
        for joystick in self._joysticks:
            del joystick
        self._joysticks = []
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

    def _handle_button_event(self, event: pygame.event.Event, pressed: bool):
        value = 1.0 if pressed else 0.0
        button = pygame_to_vg_button[event.button]
        if button in self._gamepad_action_mapping:
            action = self._gamepad_action_mapping[button]
            self._actions[action] = value

    def _handle_axis_event(self, event: pygame.event.Event):
        control = pygame_to_gamepad_control[event.axis]
        if control in self._gamepad_action_mapping:
            action = self._gamepad_action_mapping[control]
            self._actions[action] = event.value
