from pathlib import Path
from typing import Any, Dict, Literal, Optional, Self, Union

from pydantic import Field

from guidellm.backend import BackendType
from guidellm.core import Serializable
from guidellm.executor import ProfileGenerationMode

__ALL__ = ["Scenario", "ScenarioManager"]

scenarios_path = Path(__name__).parent / "scenarios"


class Scenario(Serializable):
    backend: BackendType = "openai_http"
    backend_kwargs: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    tokenizer: Optional[str] = None
    data: Union[str, Dict[str, Any]] = Field(default_factory=dict)
    data_type: Literal["emulated", "file", "transformers"] = "emulated"
    rate_type: ProfileGenerationMode = "sweep"
    rate: Optional[float] = None
    max_seconds: int = 120
    max_requests: Optional[Union[int, Literal["dataset"]]] = None

    def _update(self, **fields: Mapping[str, Any]) -> Self:
        for k, v in fields.items():
            if not hasattr(self, k):
                raise ValueError(f"Invalid field {k}")
            setattr(self, k, v)

        return self

    def update(self, **fields: Mapping[str, Any]) -> Self:
        return self._update(**{k: v for k, v in fields.items() if v is not None})


class ScenarioManager:
    def __init__(self, scenarios_dir: Optional[str] = None):
        self.scenarios: Dict[str, Scenario] = {}

        if scenarios_dir is None:
            global scenarios_path
        else:
            scenarios_path = Path(scenarios_dir)

        # Load built-in scenarios
        for scenario_path in scenarios_path.glob("*.json"):
            scenario = Scenario.from_json(scenario_path.read_text())
            self[scenario_path.stem] = scenario

    def __getitem__(self, scenario_name: str) -> Scenario:
        return self.scenarios[scenario_name]

    def __setitem__(self, scenario_name: str, scenario: Scenario):
        if scenario_name in self.scenarios:
            raise ValueError(f"Scenario {scenario_name} already exists")

        self.scenarios[scenario_name] = scenario

    def list(self):
        return tuple(self.scenarios.keys())
