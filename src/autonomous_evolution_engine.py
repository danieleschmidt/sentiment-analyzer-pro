"""Autonomous Evolution Engine - Next-generation self-improving AI system."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
import threading
import hashlib

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance tracking."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency_ms: float
    throughput_rps: float
    memory_mb: float
    cpu_percent: float
    timestamp: str
    version: str


@dataclass
class EvolutionCandidate:
    """Represents a potential model evolution."""

    model_id: str
    performance: PerformanceMetrics
    configuration: Dict[str, Any]
    fitness_score: float
    generation: int
    parent_id: Optional[str] = None


class AutonomousEvolutionEngine:
    """Self-improving AI system that evolves without human intervention."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.current_generation = 0
        self.population: List[EvolutionCandidate] = []
        self.elite_models: List[EvolutionCandidate] = []
        self.performance_history: List[PerformanceMetrics] = []
        self.evolution_running = False
        self.mutation_strategies = self._initialize_mutation_strategies()
        self.fitness_evaluator = self._create_fitness_evaluator()

        # Autonomous learning parameters
        self.population_size = self.config.get("population_size", 20)
        self.elite_ratio = self.config.get("elite_ratio", 0.2)
        self.mutation_rate = self.config.get("mutation_rate", 0.1)
        self.crossover_rate = self.config.get("crossover_rate", 0.7)
        self.evolution_interval = self.config.get("evolution_interval", 3600)  # 1 hour

        # Performance tracking
        self.performance_threshold = self.config.get("performance_threshold", 0.85)
        self.improvement_tolerance = self.config.get("improvement_tolerance", 0.01)
        self.stagnation_limit = self.config.get("stagnation_limit", 5)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load evolution configuration."""
        default_config = {
            "population_size": 20,
            "elite_ratio": 0.2,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7,
            "evolution_interval": 3600,
            "performance_threshold": 0.85,
            "improvement_tolerance": 0.01,
            "stagnation_limit": 5,
            "enable_quantum_mutations": True,
            "enable_neuromorphic_adaptation": True,
            "enable_photonic_optimization": True,
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        return default_config

    def _initialize_mutation_strategies(self) -> Dict[str, Callable]:
        """Initialize various mutation strategies."""
        return {
            "parameter_drift": self._mutate_parameters,
            "architecture_evolution": self._mutate_architecture,
            "hybrid_fusion": self._hybrid_mutation,
            "quantum_tunneling": self._quantum_mutation,
            "neuromorphic_adaptation": self._neuromorphic_mutation,
            "photonic_enhancement": self._photonic_mutation,
        }

    def _create_fitness_evaluator(self) -> Callable:
        """Create comprehensive fitness evaluation function."""

        def evaluate_fitness(candidate: EvolutionCandidate) -> float:
            metrics = candidate.performance

            # Multi-objective fitness function
            accuracy_score = metrics.accuracy * 0.3
            f1_score = metrics.f1_score * 0.25
            speed_score = min(1.0, 100.0 / metrics.latency_ms) * 0.2
            efficiency_score = min(1.0, 1000.0 / metrics.memory_mb) * 0.15
            throughput_score = min(1.0, metrics.throughput_rps / 1000.0) * 0.1

            # Bonus for balanced performance
            balance_bonus = 1.0 - np.std([accuracy_score, f1_score, speed_score])

            return (
                accuracy_score
                + f1_score
                + speed_score
                + efficiency_score
                + throughput_score
                + balance_bonus * 0.1
            )

        return evaluate_fitness

    async def start_autonomous_evolution(self):
        """Start the autonomous evolution process."""
        if self.evolution_running:
            logger.warning("Evolution already running")
            return

        self.evolution_running = True
        logger.info("Starting autonomous evolution engine")

        # Initialize population if empty
        if not self.population:
            await self._initialize_population()

        # Main evolution loop
        while self.evolution_running:
            try:
                await self._evolution_cycle()
                await asyncio.sleep(self.evolution_interval)
            except Exception as e:
                logger.error(f"Evolution cycle failed: {e}")
                await asyncio.sleep(60)  # Recovery delay

    async def stop_autonomous_evolution(self):
        """Stop the autonomous evolution process."""
        self.evolution_running = False
        logger.info("Stopping autonomous evolution engine")

    async def _initialize_population(self):
        """Initialize the first generation of models."""
        logger.info("Initializing evolution population")

        # Create diverse initial population
        base_configs = self._generate_diverse_configs()

        for i, config in enumerate(base_configs):
            candidate = EvolutionCandidate(
                model_id=f"gen0_model_{i}",
                performance=await self._evaluate_model_performance(config),
                configuration=config,
                fitness_score=0.0,
                generation=0,
            )
            candidate.fitness_score = self.fitness_evaluator(candidate)
            self.population.append(candidate)

        # Sort by fitness and select elites
        self.population.sort(key=lambda x: x.fitness_score, reverse=True)
        elite_count = int(self.population_size * self.elite_ratio)
        self.elite_models = self.population[:elite_count]

        logger.info(f"Initialized population with {len(self.population)} candidates")

    async def _evolution_cycle(self):
        """Execute one complete evolution cycle."""
        self.current_generation += 1
        logger.info(
            f"Starting evolution cycle for generation {self.current_generation}"
        )

        # Evaluate current population
        await self._evaluate_population()

        # Selection and reproduction
        new_population = await self._reproduce_population()

        # Mutation and crossover
        mutated_population = await self._mutate_population(new_population)

        # Evaluate new candidates
        for candidate in mutated_population:
            candidate.performance = await self._evaluate_model_performance(
                candidate.configuration
            )
            candidate.fitness_score = self.fitness_evaluator(candidate)

        # Survival selection
        combined_population = self.population + mutated_population
        combined_population.sort(key=lambda x: x.fitness_score, reverse=True)

        # Keep best performers
        self.population = combined_population[: self.population_size]
        elite_count = int(self.population_size * self.elite_ratio)
        self.elite_models = self.population[:elite_count]

        # Log evolution progress
        best_fitness = self.population[0].fitness_score
        avg_fitness = np.mean([c.fitness_score for c in self.population])

        logger.info(
            f"Generation {self.current_generation}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}"
        )

        # Check for deployment of improved models
        await self._autonomous_deployment_check()

        # Adaptive parameter adjustment
        self._adapt_evolution_parameters()

    async def _evaluate_population(self):
        """Evaluate performance of all candidates in parallel."""
        tasks = []
        for candidate in self.population:
            if self._needs_reevaluation(candidate):
                task = self._evaluate_model_performance(candidate.configuration)
                tasks.append((candidate, task))

        if tasks:
            results = await asyncio.gather(*[task for _, task in tasks])
            for (candidate, _), performance in zip(tasks, results):
                candidate.performance = performance
                candidate.fitness_score = self.fitness_evaluator(candidate)

    def _needs_reevaluation(self, candidate: EvolutionCandidate) -> bool:
        """Determine if candidate needs performance re-evaluation."""
        if not hasattr(candidate, "last_evaluated"):
            return True

        # Re-evaluate if performance data is old
        if candidate.performance.timestamp:
            eval_time = datetime.fromisoformat(candidate.performance.timestamp)
            return datetime.now() - eval_time > timedelta(hours=6)

        return True

    async def _reproduce_population(self) -> List[EvolutionCandidate]:
        """Create new candidates through reproduction."""
        new_population = []

        # Keep elite models
        new_population.extend(self.elite_models)

        # Generate offspring through crossover
        while len(new_population) < self.population_size:
            if np.random.random() < self.crossover_rate:
                parent1, parent2 = self._select_parents()
                offspring = await self._crossover(parent1, parent2)
                new_population.append(offspring)
            else:
                # Clone elite with slight variation
                parent = np.random.choice(self.elite_models)
                offspring = self._clone_with_variation(parent)
                new_population.append(offspring)

        return new_population

    def _select_parents(self) -> Tuple[EvolutionCandidate, EvolutionCandidate]:
        """Select parents for reproduction using tournament selection."""
        tournament_size = 3

        def tournament_select():
            tournament = np.random.choice(
                self.population, tournament_size, replace=False
            )
            return max(tournament, key=lambda x: x.fitness_score)

        parent1 = tournament_select()
        parent2 = tournament_select()

        # Ensure diversity
        while parent2.model_id == parent1.model_id and len(self.population) > 1:
            parent2 = tournament_select()

        return parent1, parent2

    async def _crossover(
        self, parent1: EvolutionCandidate, parent2: EvolutionCandidate
    ) -> EvolutionCandidate:
        """Create offspring through intelligent crossover."""
        # Merge configurations intelligently
        offspring_config = {}

        for key in parent1.configuration:
            if key in parent2.configuration:
                if np.random.random() < 0.5:
                    offspring_config[key] = parent1.configuration[key]
                else:
                    offspring_config[key] = parent2.configuration[key]

                # Blend numerical parameters
                if isinstance(parent1.configuration[key], (int, float)) and isinstance(
                    parent2.configuration[key], (int, float)
                ):
                    val1, val2 = parent1.configuration[key], parent2.configuration[key]
                    alpha = np.random.uniform(0.3, 0.7)
                    offspring_config[key] = alpha * val1 + (1 - alpha) * val2
            else:
                offspring_config[key] = parent1.configuration[key]

        # Add unique identifier
        model_id = f"gen{self.current_generation}_cross_{len(self.population)}"

        return EvolutionCandidate(
            model_id=model_id,
            performance=PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, "", ""),
            configuration=offspring_config,
            fitness_score=0.0,
            generation=self.current_generation,
            parent_id=f"{parent1.model_id}+{parent2.model_id}",
        )

    def _clone_with_variation(self, parent: EvolutionCandidate) -> EvolutionCandidate:
        """Clone parent with small variations."""
        config_copy = parent.configuration.copy()

        # Add small random variations
        for key, value in config_copy.items():
            if isinstance(value, float):
                config_copy[key] = value * np.random.uniform(0.95, 1.05)
            elif isinstance(value, int) and value > 0:
                config_copy[key] = max(1, int(value * np.random.uniform(0.9, 1.1)))

        model_id = f"gen{self.current_generation}_clone_{len(self.population)}"

        return EvolutionCandidate(
            model_id=model_id,
            performance=PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, "", ""),
            configuration=config_copy,
            fitness_score=0.0,
            generation=self.current_generation,
            parent_id=parent.model_id,
        )

    async def _mutate_population(
        self, population: List[EvolutionCandidate]
    ) -> List[EvolutionCandidate]:
        """Apply various mutation strategies to population."""
        mutated = []

        for candidate in population:
            if np.random.random() < self.mutation_rate:
                # Select random mutation strategy
                strategy = np.random.choice(list(self.mutation_strategies.keys()))
                mutated_candidate = await self.mutation_strategies[strategy](candidate)
                mutated.append(mutated_candidate)
            else:
                mutated.append(candidate)

        return mutated

    async def _mutate_parameters(
        self, candidate: EvolutionCandidate
    ) -> EvolutionCandidate:
        """Standard parameter mutation."""
        mutated_config = candidate.configuration.copy()

        # Mutate numerical parameters
        for key, value in mutated_config.items():
            if isinstance(value, float):
                mutated_config[key] = value * np.random.normal(1.0, 0.1)
            elif isinstance(value, int) and value > 0:
                mutated_config[key] = max(
                    1, int(value + np.random.normal(0, value * 0.1))
                )

        return self._create_mutated_candidate(candidate, mutated_config, "param_mut")

    async def _mutate_architecture(
        self, candidate: EvolutionCandidate
    ) -> EvolutionCandidate:
        """Architectural mutation for model structure."""
        mutated_config = candidate.configuration.copy()

        # Modify architectural parameters
        if "hidden_layers" in mutated_config:
            current_layers = mutated_config["hidden_layers"]
            if np.random.random() < 0.3:  # Add layer
                mutated_config["hidden_layers"] = current_layers + [
                    np.random.randint(32, 256)
                ]
            elif np.random.random() < 0.3 and len(current_layers) > 1:  # Remove layer
                mutated_config["hidden_layers"] = current_layers[:-1]
            else:  # Modify existing layer
                if current_layers:
                    idx = np.random.randint(len(current_layers))
                    mutated_config["hidden_layers"][idx] = np.random.randint(16, 512)

        return self._create_mutated_candidate(candidate, mutated_config, "arch_mut")

    async def _hybrid_mutation(
        self, candidate: EvolutionCandidate
    ) -> EvolutionCandidate:
        """Hybrid mutation combining multiple approaches."""
        mutated_config = candidate.configuration.copy()

        # Apply multiple mutation types
        strategies = ["param", "arch", "optim"]
        num_mutations = np.random.randint(1, len(strategies) + 1)
        selected_strategies = np.random.choice(strategies, num_mutations, replace=False)

        for strategy in selected_strategies:
            if strategy == "param":
                await self._mutate_parameters_in_place(mutated_config)
            elif strategy == "arch":
                await self._mutate_architecture_in_place(mutated_config)
            elif strategy == "optim":
                await self._mutate_optimizer_in_place(mutated_config)

        return self._create_mutated_candidate(candidate, mutated_config, "hybrid_mut")

    async def _quantum_mutation(
        self, candidate: EvolutionCandidate
    ) -> EvolutionCandidate:
        """Quantum-inspired mutation with superposition effects."""
        if not self.config.get("enable_quantum_mutations", True):
            return await self._mutate_parameters(candidate)

        mutated_config = candidate.configuration.copy()

        # Quantum tunneling through local optima
        for key, value in mutated_config.items():
            if isinstance(value, (int, float)):
                # Quantum jump with small probability
                if np.random.random() < 0.05:
                    jump_magnitude = abs(value) * np.random.uniform(0.5, 2.0)
                    direction = np.random.choice([-1, 1])
                    mutated_config[key] = value + direction * jump_magnitude

        return self._create_mutated_candidate(candidate, mutated_config, "quantum_mut")

    async def _neuromorphic_mutation(
        self, candidate: EvolutionCandidate
    ) -> EvolutionCandidate:
        """Neuromorphic-inspired adaptive mutation."""
        if not self.config.get("enable_neuromorphic_adaptation", True):
            return await self._mutate_parameters(candidate)

        mutated_config = candidate.configuration.copy()

        # Spike-based parameter adaptation
        if "learning_rate" in mutated_config:
            current_lr = mutated_config["learning_rate"]
            spike_threshold = 0.1

            # Adaptive learning rate based on performance history
            recent_performance = self._get_recent_performance_trend()
            if recent_performance < spike_threshold:
                mutated_config["learning_rate"] = current_lr * np.random.uniform(
                    0.1, 0.5
                )
            else:
                mutated_config["learning_rate"] = current_lr * np.random.uniform(
                    1.1, 2.0
                )

        return self._create_mutated_candidate(candidate, mutated_config, "neuro_mut")

    async def _photonic_mutation(
        self, candidate: EvolutionCandidate
    ) -> EvolutionCandidate:
        """Photonic-inspired optimization mutation."""
        if not self.config.get("enable_photonic_optimization", True):
            return await self._mutate_parameters(candidate)

        mutated_config = candidate.configuration.copy()

        # Light-based parameter optimization
        for key, value in mutated_config.items():
            if isinstance(value, (int, float)):
                # Interference pattern optimization
                wave_amplitude = abs(value) * 0.1
                phase = np.random.uniform(0, 2 * np.pi)
                interference = wave_amplitude * np.sin(phase)
                mutated_config[key] = value + interference

        return self._create_mutated_candidate(candidate, mutated_config, "photonic_mut")

    def _create_mutated_candidate(
        self, parent: EvolutionCandidate, mutated_config: Dict, mutation_type: str
    ) -> EvolutionCandidate:
        """Create new candidate from mutated configuration."""
        model_id = f"gen{self.current_generation}_{mutation_type}_{hash(str(mutated_config)) % 10000}"

        return EvolutionCandidate(
            model_id=model_id,
            performance=PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, "", ""),
            configuration=mutated_config,
            fitness_score=0.0,
            generation=self.current_generation,
            parent_id=parent.model_id,
        )

    async def _evaluate_model_performance(
        self, config: Dict[str, Any]
    ) -> PerformanceMetrics:
        """Evaluate model performance with given configuration."""
        # Simulate model training and evaluation
        # In practice, this would train and test actual models

        start_time = time.time()

        # Simulate performance based on config
        base_accuracy = 0.8
        config_bonus = (
            sum(1 for v in config.values() if isinstance(v, (int, float))) * 0.01
        )
        accuracy = min(0.95, base_accuracy + config_bonus + np.random.normal(0, 0.05))

        precision = accuracy + np.random.normal(0, 0.02)
        recall = accuracy + np.random.normal(0, 0.02)
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        latency_ms = np.random.uniform(10, 100)
        throughput_rps = 1000 / latency_ms
        memory_mb = np.random.uniform(100, 500)
        cpu_percent = np.random.uniform(20, 80)

        return PerformanceMetrics(
            accuracy=max(0, min(1, accuracy)),
            precision=max(0, min(1, precision)),
            recall=max(0, min(1, recall)),
            f1_score=max(0, min(1, f1)),
            latency_ms=latency_ms,
            throughput_rps=throughput_rps,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            timestamp=datetime.now().isoformat(),
            version=f"gen{self.current_generation}",
        )

    async def _autonomous_deployment_check(self):
        """Check if any evolved models should be deployed autonomously."""
        if not self.elite_models:
            return

        best_candidate = self.elite_models[0]

        # Check if significantly better than current production model
        if self._should_deploy_autonomously(best_candidate):
            await self._deploy_model_autonomously(best_candidate)

    def _should_deploy_autonomously(self, candidate: EvolutionCandidate) -> bool:
        """Determine if model should be deployed autonomously."""
        if not self.performance_history:
            return candidate.fitness_score > 0.8

        # Compare with recent performance
        recent_avg = np.mean([p.accuracy for p in self.performance_history[-10:]])
        improvement = candidate.performance.accuracy - recent_avg

        return (
            improvement > self.improvement_tolerance
            and candidate.performance.accuracy > self.performance_threshold
            and candidate.fitness_score > 0.85
        )

    async def _deploy_model_autonomously(self, candidate: EvolutionCandidate):
        """Deploy evolved model autonomously."""
        logger.info(
            f"Autonomous deployment: {candidate.model_id} with fitness {candidate.fitness_score:.4f}"
        )

        try:
            # Save model configuration
            deployment_path = Path("models/autonomous") / f"{candidate.model_id}.json"
            deployment_path.parent.mkdir(parents=True, exist_ok=True)

            deployment_data = {
                "model_id": candidate.model_id,
                "configuration": candidate.configuration,
                "performance": asdict(candidate.performance),
                "fitness_score": candidate.fitness_score,
                "deployed_at": datetime.now().isoformat(),
            }

            with open(deployment_path, "w") as f:
                json.dump(deployment_data, f, indent=2)

            # Update performance history
            self.performance_history.append(candidate.performance)

            logger.info(f"Successfully deployed {candidate.model_id}")

        except Exception as e:
            logger.error(f"Failed to deploy {candidate.model_id}: {e}")

    def _adapt_evolution_parameters(self):
        """Adaptively adjust evolution parameters based on progress."""
        if len(self.performance_history) < 10:
            return

        # Analyze recent performance trends
        recent_scores = [p.accuracy for p in self.performance_history[-10:]]
        performance_trend = np.diff(recent_scores)

        if np.mean(performance_trend) < 0.001:  # Stagnation
            # Increase mutation rate and diversity
            self.mutation_rate = min(0.3, self.mutation_rate * 1.2)
            logger.info(
                f"Increased mutation rate to {self.mutation_rate:.3f} due to stagnation"
            )
        elif np.mean(performance_trend) > 0.01:  # Good progress
            # Decrease mutation rate for fine-tuning
            self.mutation_rate = max(0.05, self.mutation_rate * 0.9)
            logger.info(
                f"Decreased mutation rate to {self.mutation_rate:.3f} for fine-tuning"
            )

    def _generate_diverse_configs(self) -> List[Dict[str, Any]]:
        """Generate diverse initial configurations."""
        configs = []

        # Base configuration templates
        base_templates = [
            {"learning_rate": 0.001, "batch_size": 32, "hidden_layers": [64, 32]},
            {"learning_rate": 0.01, "batch_size": 64, "hidden_layers": [128, 64, 32]},
            {"learning_rate": 0.0001, "batch_size": 16, "hidden_layers": [256, 128]},
        ]

        # Generate variations
        for template in base_templates:
            for _ in range(self.population_size // len(base_templates)):
                config = template.copy()
                # Add random variations
                config["learning_rate"] *= np.random.uniform(0.5, 2.0)
                config["batch_size"] = int(
                    config["batch_size"] * np.random.uniform(0.5, 2.0)
                )

                # Random architectural variations
                if np.random.random() < 0.3:
                    config["hidden_layers"].append(np.random.randint(16, 128))

                configs.append(config)

        # Fill remaining slots with completely random configs
        while len(configs) < self.population_size:
            configs.append(
                {
                    "learning_rate": 10 ** np.random.uniform(-5, -2),
                    "batch_size": 2 ** np.random.randint(3, 8),
                    "hidden_layers": [
                        2 ** np.random.randint(4, 9)
                        for _ in range(np.random.randint(1, 4))
                    ],
                }
            )

        return configs[: self.population_size]

    def _get_recent_performance_trend(self) -> float:
        """Get recent performance improvement trend."""
        if len(self.performance_history) < 5:
            return 0.0

        recent = [p.accuracy for p in self.performance_history[-5:]]
        return np.mean(np.diff(recent))

    async def _mutate_parameters_in_place(self, config: Dict[str, Any]):
        """Mutate parameters in place."""
        for key, value in config.items():
            if isinstance(value, float):
                config[key] = value * np.random.normal(1.0, 0.05)

    async def _mutate_architecture_in_place(self, config: Dict[str, Any]):
        """Mutate architecture in place."""
        if "hidden_layers" in config and config["hidden_layers"]:
            idx = np.random.randint(len(config["hidden_layers"]))
            config["hidden_layers"][idx] = max(
                16, int(config["hidden_layers"][idx] * np.random.uniform(0.8, 1.2))
            )

    async def _mutate_optimizer_in_place(self, config: Dict[str, Any]):
        """Mutate optimizer parameters in place."""
        if "learning_rate" in config:
            config["learning_rate"] *= np.random.uniform(0.5, 2.0)

    def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution status."""
        if not self.population:
            return {
                "status": "not_initialized",
                "generation": self.current_generation,
                "population_size": 0,
                "best_fitness": 0.0,
                "average_fitness": 0.0,
                "elite_count": 0,
                "mutation_rate": self.mutation_rate,
                "performance_history_length": len(self.performance_history),
            }

        best_fitness = (
            max(c.fitness_score for c in self.population) if self.population else 0
        )
        avg_fitness = (
            np.mean([c.fitness_score for c in self.population])
            if self.population
            else 0
        )

        return {
            "status": "running" if self.evolution_running else "stopped",
            "generation": self.current_generation,
            "population_size": len(self.population),
            "best_fitness": best_fitness,
            "average_fitness": avg_fitness,
            "elite_count": len(self.elite_models),
            "mutation_rate": self.mutation_rate,
            "performance_history_length": len(self.performance_history),
        }

    def save_evolution_state(self, filepath: str):
        """Save current evolution state."""
        state = {
            "current_generation": self.current_generation,
            "population": [asdict(c) for c in self.population],
            "elite_models": [asdict(c) for c in self.elite_models],
            "performance_history": [asdict(p) for p in self.performance_history],
            "config": self.config,
            "mutation_rate": self.mutation_rate,
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Evolution state saved to {filepath}")

    def load_evolution_state(self, filepath: str):
        """Load evolution state from file."""
        with open(filepath, "r") as f:
            state = json.load(f)

        self.current_generation = state["current_generation"]
        self.population = [EvolutionCandidate(**c) for c in state["population"]]
        self.elite_models = [EvolutionCandidate(**c) for c in state["elite_models"]]
        self.performance_history = [
            PerformanceMetrics(**p) for p in state["performance_history"]
        ]
        self.mutation_rate = state.get("mutation_rate", self.mutation_rate)

        logger.info(f"Evolution state loaded from {filepath}")


# Global evolution engine instance
_evolution_engine = None


def get_evolution_engine() -> AutonomousEvolutionEngine:
    """Get global evolution engine instance."""
    global _evolution_engine
    if _evolution_engine is None:
        _evolution_engine = AutonomousEvolutionEngine()
    return _evolution_engine


async def start_autonomous_evolution():
    """Start the autonomous evolution process."""
    engine = get_evolution_engine()
    await engine.start_autonomous_evolution()


async def stop_autonomous_evolution():
    """Stop the autonomous evolution process."""
    engine = get_evolution_engine()
    await engine.stop_autonomous_evolution()


def get_evolution_status() -> Dict[str, Any]:
    """Get current evolution status."""
    engine = get_evolution_engine()
    return engine.get_evolution_status()
