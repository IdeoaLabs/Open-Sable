import random
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Union
from enum import Enum, auto
import numpy as np

class MutationType(Enum):
    """
    Supported mutation strategies defining how genetic variation is introduced.

    These types determine the specific mechanism used to modify individual
    parameters or genes in the population during evolution.
    """

    SINGLE_POINT = auto()
    """Modifies exactly one random parameter at a time."""

    MULTI_POINT = auto()
    """Modifies multiple discrete parameters simultaneously."""

    UNIFORM = auto()
    """Randomly mutates each parameter with a uniform probability across the range."""

    GAUSSIAN = auto()
    """Applies Gaussian (normal) distribution-based mutations around current value."""

    CREATIVE = auto()
    """Uses innovative or non-standard mutation patterns exploring complex variations."""

@dataclass
class MutationConfig:
    def __init__(self, mutation_rate: Optional[float] = None, 
                 num_points: Optional[int] = None, 
                 std_dev: Optional[float] = None):
        if mutation_rate is not None:
            self.mutation_rate = mutation_rate
        self.mutation_rate = self.mutation_rate or 0.1
        
        if num_points is not None:
            self.num_points = num_points
        self.num_points = self.num_points or 2
        
        if std_dev is not None:
            self.std_dev = std_dev
        self.std_dev = self.std_dev or 0.5
    mutation_rate: float = 0.1
    num_points: int = 2
    std_dev: float = 0.5

class RiskAssessment:
    @staticmethod
    def calculate_risk(current_state, proposed_state) -> Tuple[float, dict]:
        """
        Calculates the risk of transitioning from current_state to proposed_state.

        Args:
            current_state: The current system state
            proposed_state: The proposed state to transition to

        Returns:
            A tuple containing the total risk score (0-1) and a detailed assessment dictionary
        """
        risk_score = 0.0
        assessment = {'total_risk': 0.0, 'components': {}}

        # State difference risk
        # Measures how different the proposed state is from the current state
        difference = abs(hash(tuple(current_state)) - hash(tuple(proposed_state)))
        assessment['components']['state_difference'] = difference
        if difference > 1000:
            risk_score += 0.7

        # Stability check
        try:
            stability = self._evaluate_stability(proposed_state)
            assessment['components']['stability'] = round(stability, 2)
            if stability < 0.5:
                risk_score += 0.3
        except Exception as e:
            assessment['components']['stability'] = {'error': str(e)}

        # Final risk score normalization
        risk_score = min(risk_score, 1.0)
        assessment['total_risk'] = round(risk_score, 2)
        return risk_score

    def _evaluate_stability(self, state):
        """
        Evaluates the stability of a proposed state.

        This is a simplified metric that could be expanded with more sophisticated
        stability checking mechanisms.

        Args:
            state: The state to evaluate

        Returns:
            A stability score between 0 and 1, where 1 indicates perfect stability
        """
        # Simple implementation - returns 1.0 for stable (needs improvement)
        # In practice, this should analyze various stability indicators

        try:
            # Example: check if state has valid ranges
            if hasattr(state, '__iter__'):
                valid_ranges = all(isinstance(item, tuple) and len(item) == 2 for item in state)
                if not valid_ranges:
                    return 0.3
            
            # Example: check numerical stability (large values may indicate issues)
            if isinstance(state, dict):
                large_values = any(abs(val) > 1e6 for val in state.values())
                if large_values:
                    return 0.5
            
            # Default stability score (moderately stable)
            return 0.8
        except Exception:
            return 0.3
        return random.uniform(0.4, 0.9)

class Mutator:
    def __init__(self, config: Optional[MutationConfig] = None):
        self.config = config or MutationConfig()

    def mutate(self, individual: List[Any]) -> List[Any]:
        raise NotImplementedError

    def should_mutate(self) -> bool:
        return random.random() < self.config.mutation_rate

class SinglePointMutator(Mutator):
    def mutate(self, individual: List[Any]) -> List[Any]:
        mutated = individual.copy()
        point = random.randint(0, len(mutated) - 1)
        mutated[point] = self._mutate_value(mutated[point])
        return mutated

    def _mutate_value(self, value):
        if isinstance(value, int):
            return value + random.randint(-1, 1)
        elif isinstance(value, float):
            return value + random.uniform(-0.5, 0.5)
        return value

class MultiPointMutator(Mutator):
    def mutate(self, individual: List[Any]) -> List[Any]:
        mutated = individual.copy()
        points_to_mutate = min(self.config.num_points, len(mutated))
        for _ in range(points_to_mutate):
            point = random.randint(0, len(mutated) - 1)
            mutated[point] = self._mutate_value(mutated[point])

        return mutated

    def _mutate_value(self, value):
        mutation_types = [
            lambda v: v + random.randint(-2, 2) if isinstance(v, int) else v,
            lambda v: v + random.uniform(-1.0, 1.0) if isinstance(v, float) else v,
            lambda v: not v if isinstance(v, bool) else v
        ]
        return random.choice(mutation_types)(value)

class UniformMutator(Mutator):
    def mutate(self, individual: List[Any]) -> List[Any]:
        mutated = []
        for value in individual:
            if random.random() < self.config.mutation_rate:
                mutated.append(self._mutate_value(value))
            else:
                mutated.append(value)
        return mutated

    def _mutate_value(self, value):
        if isinstance(value, bool):
            return random.choice([True, False])
        return super()._mutate_value(value)

class GaussianMutator(Mutator):
    def mutate(self, individual: List[Any]) -> List[Any]:
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.config.mutation_rate:
                mutated[i] += np.random.normal(0, self.config.std_dev)
        return mutated

class CreativeMutator(Mutator):
    def __init__(self, config: MutationConfig, risk_assessor: RiskAssessment):
        super().__init__(config)
        self.risk_assessor = risk_assessor

    def mutate(self, individual: List[Any]) -> Tuple[List[Any], dict]:
        original_hash = hash(tuple(individual))
        candidates = self._generate_candidates(individual)
        assessed = self._assess_candidates(candidates, individual)
        return self._select_lowest_risk(assessed), {'original_hash': original_hash}

    def _generate_candidates(self, individual):
        strategies = [
            SinglePointMutator(),
            MultiPointMutator(),
            UniformMutator(),
            GaussianMutator()
        ]
        return [strat.mutate(individual.copy()) for strat in strategies]

    def _assess_candidates(self, candidates, original):
        return [
            (
                self.risk_assessor.calculate_risk(original, cand)[0],
                cand,
                f'Risk: {self.risk_assessor.calculate_risk(original, cand)[0]:.2f}'
            ) for cand in candidates
        ]

    def _select_lowest_risk(self, assessed):
        assessed.sort(key=lambda x: x[0])
        return assessed[0][1]

def main():
    config = MutationConfig(mutation_rate=0.3, std_dev=0.8)
    risk_assessor = RiskAssessment()

    initial_individual = [42, True, 3.14, 'hello', False]
    print(f"Original: {initial_individual}")
    print("-" * 50)

    print("\nSingle Point Mutation:")
    single_result = SinglePointMutator(config).mutate(initial_individual.copy())
    print(f'Result: {single_result}')
    print(f'Risk: {risk_assessor.calculate_risk(initial_individual, single_result)[0]:.2f}')

    print("\nMulti-Point Mutation:")
    multi_result = MultiPointMutator(config).mutate(initial_individual.copy())
    print(f'Result: {multi_result}')
    print(f'Risk: {risk_assessor.calculate_risk(initial_individual, multi_result)[0]:.2f}')

    print("\nUniform Mutation:")
    uniform_result = UniformMutator(config).mutate(initial_individual.copy())
    print(f'Result: {uniform_result}')
    print(f'Risk: {risk_assessor.calculate_risk(initial_individual, uniform_result)[0]:.2f}')

if __name__ == "__main__":
    main()