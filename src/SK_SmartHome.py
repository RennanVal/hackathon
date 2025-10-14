import os
import asyncio
from dataclasses import dataclass, field
from pydoc import text
from typing import Dict, Optional

from semantic_kernel.kernel import Kernel
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.kernel_pydantic import KernelBaseModel
from pydantic import PrivateAttr
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
)

# ---------------------------
# Smart Home "Devices" State
# ---------------------------
@dataclass
class SmartHomeState():
    lights: Dict[str, bool] = field(
        default_factory=lambda: {"living room": False, "kitchen": False, "bedroom": False}
    )
    thermostat_c: float = 20.0
    doors_locked: bool = True
    music_playing: Optional[str] = None

    def snapshot(self) -> str:
        lights_str = ", ".join(f"{room}: {'ON' if on else 'OFF'}" for room, on in self.lights.items())
        return (
            f"Lights -> [{lights_str}]\n"
            f"Thermostat -> {self.thermostat_c:.1f}°C\n"
            f"Doors -> {'LOCKED' if self.doors_locked else 'UNLOCKED'}\n"
            f"Music -> {self.music_playing or 'None'}"
        )


# ---------------------------
# Smart Home Plugin (SK)
# ---------------------------
class SmartHomePlugin(KernelBaseModel):
    """Semantic Kernel native-plugin exposing smart-home actions."""

    _state: SmartHomeState = PrivateAttr(default_factory=SmartHomeState)

    @kernel_function(name="set_light", description="Turn a light ON or OFF in a given room.")
    def set_light(self, room: str, turn_on: bool) -> str:
        """Turn a light ON or OFF in a given room."""
        room = room.strip().lower()
        if room not in self._state.lights:
            self._state.lights[room] = False
        self._state.lights[room] = bool(turn_on)
        return f"Light in '{room}' is now {'ON' if turn_on else 'OFF'}."

    @kernel_function(name="set_temperature", description="Set thermostat temperature in Celsius (e.g., 21.5).")
    def set_temperature(self, temperature_c: float) -> str:
        """Set thermostat temperature in Celsius (e.g., 21.5)."""
        self._state.thermostat_c = float(temperature_c)
        return f"Thermostat set to {self._state.thermostat_c:.1f}°C."

    @kernel_function(name="lock_doors", description="Lock or unlock all doors.")
    def lock_doors(self, lock: bool = True) -> str:
        """Lock or unlock all doors."""
        self._state.doors_locked = bool(lock)
        return f"Doors {'LOCKED' if self._state.doors_locked else 'UNLOCKED'}."

    @kernel_function(name="play_music", description="Play background music by genre (e.g., jazz, pop, lo-fi).")
    def play_music(self, genre: str) -> str:
        """Play background music by genre (e.g., jazz, pop, lo-fi)."""
        genre = genre.strip()
        self._state.music_playing = genre
        return f"Playing {genre} music."

    @kernel_function(name="stop_music", description="Stop any playing music.")
    def stop_music(self) -> str:
        """Stop any playing music."""
        self._state.music_playing = None
        return "Music stopped."

    @kernel_function(name="status", description="Get a human-friendly smart home status.")
    def status(self) -> str:
        """Get a human-friendly smart home status."""
        return self._state.snapshot()