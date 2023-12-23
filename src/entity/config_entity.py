from dataclasses import dataclass



@dataclass(frozen=True)
class OptimizatorConfig:

    data: dict
    n_runs: int
    model_name: str
    metric_name: str
    optimizator_name: str
    test_ratio: float
    dt: float
    seed: int
    path_to_results: str