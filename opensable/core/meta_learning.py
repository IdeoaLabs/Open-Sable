"""
Meta-Learning and Self-Improvement - AGI learns how to learn and improve itself.

Features:
- Performance tracking and analysis
- Strategy learning and adaptation
- Self-modification capabilities
- Learning from failures
- Skill acquisition
- Meta-cognitive awareness
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of learning strategies."""
    TRIAL_AND_ERROR = "trial_and_error"
    IMITATION = "imitation"
    REINFORCEMENT = "reinforcement"
    TRANSFER = "transfer"
    ACTIVE_LEARNING = "active_learning"


class PerformanceMetric(Enum):
    """Performance metrics."""
    ACCURACY = "accuracy"
    SPEED = "speed"
    EFFICIENCY = "efficiency"
    QUALITY = "quality"
    SUCCESS_RATE = "success_rate"


@dataclass
class PerformanceRecord:
    """Record of task performance."""
    task_id: str
    task_type: str
    timestamp: datetime
    success: bool
    duration: timedelta
    metrics: Dict[PerformanceMetric, float]
    strategy_used: StrategyType
    feedback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_score(self) -> float:
        """Calculate overall performance score."""
        if not self.metrics:
            return 1.0 if self.success else 0.0
        
        # Weighted average of metrics
        return sum(self.metrics.values()) / len(self.metrics)


@dataclass
class LearningStrategy:
    """A learned strategy for solving tasks."""
    strategy_id: str
    name: str
    strategy_type: StrategyType
    applicable_to: List[str]  # Task types
    steps: List[str]
    success_rate: float = 0.5
    avg_duration: Optional[timedelta] = None
    usage_count: int = 0
    last_used: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_success_rate(self, success: bool):
        """Update success rate with new data point."""
        # Exponential moving average
        alpha = 0.1
        new_value = 1.0 if success else 0.0
        self.success_rate = alpha * new_value + (1 - alpha) * self.success_rate


class PerformanceTracker:
    """
    Tracks agent performance across tasks.
    
    Analyzes patterns and identifies improvement opportunities.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.records: List[PerformanceRecord] = []
        self.storage_path = storage_path or Path("./data/performance.json")
        self._load_records()
    
    def record_performance(
        self,
        task_id: str,
        task_type: str,
        success: bool,
        duration: timedelta,
        metrics: Dict[PerformanceMetric, float],
        strategy_used: StrategyType,
        feedback: Optional[str] = None
    ):
        """Record task performance."""
        record = PerformanceRecord(
            task_id=task_id,
            task_type=task_type,
            timestamp=datetime.utcnow(),
            success=success,
            duration=duration,
            metrics=metrics,
            strategy_used=strategy_used,
            feedback=feedback
        )
        
        self.records.append(record)
        self._save_records()
        
        logger.info(f"Recorded performance: {task_type} - {'Success' if success else 'Failure'}")
    
    def get_task_type_stats(self, task_type: str) -> Dict[str, Any]:
        """Get statistics for a specific task type."""
        relevant = [r for r in self.records if r.task_type == task_type]
        
        if not relevant:
            return {}
        
        success_count = sum(1 for r in relevant if r.success)
        total_count = len(relevant)
        
        avg_duration = sum([r.duration.total_seconds() for r in relevant], 0) / total_count
        
        # Best performing strategy
        strategy_performance = {}
        for record in relevant:
            strategy = record.strategy_used
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {'success': 0, 'total': 0}
            
            strategy_performance[strategy]['total'] += 1
            if record.success:
                strategy_performance[strategy]['success'] += 1
        
        best_strategy = max(
            strategy_performance.items(),
            key=lambda x: x[1]['success'] / x[1]['total'] if x[1]['total'] > 0 else 0
        )[0] if strategy_performance else None
        
        return {
            'task_type': task_type,
            'total_attempts': total_count,
            'success_count': success_count,
            'success_rate': success_count / total_count,
            'avg_duration_seconds': avg_duration,
            'best_strategy': best_strategy.value if best_strategy else None,
            'strategy_performance': {
                k.value: v for k, v in strategy_performance.items()
            }
        }
    
    def identify_weaknesses(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Identify task types with low performance."""
        task_types = set(r.task_type for r in self.records)
        weaknesses = []
        
        for task_type in task_types:
            stats = self.get_task_type_stats(task_type)
            if stats.get('success_rate', 1.0) < threshold:
                weaknesses.append(stats)
        
        weaknesses.sort(key=lambda x: x.get('success_rate', 0))
        return weaknesses
    
    def get_improvement_trends(self, task_type: str, window: int = 10) -> Dict[str, Any]:
        """Analyze improvement trends for a task type."""
        relevant = [r for r in self.records if r.task_type == task_type]
        
        if len(relevant) < window:
            return {'insufficient_data': True}
        
        # Compare recent vs older performance
        recent = relevant[-window:]
        older = relevant[-2*window:-window] if len(relevant) >= 2*window else relevant[:-window]
        
        recent_success_rate = sum(1 for r in recent if r.success) / len(recent)
        older_success_rate = sum(1 for r in older if r.success) / len(older) if older else 0
        
        improvement = recent_success_rate - older_success_rate
        
        return {
            'task_type': task_type,
            'recent_success_rate': recent_success_rate,
            'older_success_rate': older_success_rate,
            'improvement': improvement,
            'trend': 'improving' if improvement > 0.1 else ('stable' if abs(improvement) <= 0.1 else 'declining')
        }
    
    def _save_records(self):
        """Save performance records."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = [
                {
                    'task_id': r.task_id,
                    'task_type': r.task_type,
                    'timestamp': r.timestamp.isoformat(),
                    'success': r.success,
                    'duration': r.duration.total_seconds(),
                    'metrics': {k.value: v for k, v in r.metrics.items()},
                    'strategy_used': r.strategy_used.value,
                    'feedback': r.feedback,
                    'metadata': r.metadata
                }
                for r in self.records
            ]
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save performance records: {e}")
    
    def _load_records(self):
        """Load performance records."""
        try:
            if not self.storage_path.exists():
                return
            
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for item in data:
                record = PerformanceRecord(
                    task_id=item['task_id'],
                    task_type=item['task_type'],
                    timestamp=datetime.fromisoformat(item['timestamp']),
                    success=item['success'],
                    duration=timedelta(seconds=item['duration']),
                    metrics={PerformanceMetric(k): v for k, v in item['metrics'].items()},
                    strategy_used=StrategyType(item['strategy_used']),
                    feedback=item.get('feedback'),
                    metadata=item.get('metadata', {})
                )
                self.records.append(record)
            
            logger.info(f"Loaded {len(self.records)} performance records")
        
        except Exception as e:
            logger.error(f"Failed to load performance records: {e}")


class StrategyLibrary:
    """
    Library of learned strategies.
    
    Stores and retrieves strategies for different task types.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.strategies: Dict[str, LearningStrategy] = {}
        self.storage_path = storage_path or Path("./data/strategies.json")
        self._load_strategies()
    
    def add_strategy(
        self,
        name: str,
        strategy_type: StrategyType,
        applicable_to: List[str],
        steps: List[str]
    ) -> str:
        """Add a new strategy."""
        strategy_id = f"strat_{len(self.strategies)}_{name.replace(' ', '_')}"
        
        strategy = LearningStrategy(
            strategy_id=strategy_id,
            name=name,
            strategy_type=strategy_type,
            applicable_to=applicable_to,
            steps=steps
        )
        
        self.strategies[strategy_id] = strategy
        self._save_strategies()
        
        logger.info(f"Added strategy: {name}")
        return strategy_id
    
    def get_best_strategy(
        self,
        task_type: str,
        min_success_rate: float = 0.6
    ) -> Optional[LearningStrategy]:
        """Get best strategy for a task type."""
        applicable = [
            s for s in self.strategies.values()
            if task_type in s.applicable_to and s.success_rate >= min_success_rate
        ]
        
        if not applicable:
            return None
        
        # Sort by success rate
        applicable.sort(key=lambda x: x.success_rate, reverse=True)
        return applicable[0]
    
    def update_strategy_performance(
        self,
        strategy_id: str,
        success: bool,
        duration: timedelta
    ):
        """Update strategy performance metrics."""
        if strategy_id not in self.strategies:
            return
        
        strategy = self.strategies[strategy_id]
        strategy.update_success_rate(success)
        strategy.usage_count += 1
        strategy.last_used = datetime.utcnow()
        
        # Update average duration
        if strategy.avg_duration:
            # Exponential moving average
            alpha = 0.2
            strategy.avg_duration = timedelta(
                seconds=alpha * duration.total_seconds() + (1 - alpha) * strategy.avg_duration.total_seconds()
            )
        else:
            strategy.avg_duration = duration
        
        self._save_strategies()
    
    def prune_ineffective_strategies(self, threshold: float = 0.3):
        """Remove strategies with low success rates."""
        to_remove = [
            sid for sid, s in self.strategies.items()
            if s.usage_count > 10 and s.success_rate < threshold
        ]
        
        for sid in to_remove:
            del self.strategies[sid]
            logger.info(f"Pruned ineffective strategy: {sid}")
        
        if to_remove:
            self._save_strategies()
    
    def _save_strategies(self):
        """Save strategies to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                sid: {
                    'strategy_id': s.strategy_id,
                    'name': s.name,
                    'strategy_type': s.strategy_type.value,
                    'applicable_to': s.applicable_to,
                    'steps': s.steps,
                    'success_rate': s.success_rate,
                    'avg_duration': s.avg_duration.total_seconds() if s.avg_duration else None,
                    'usage_count': s.usage_count,
                    'last_used': s.last_used.isoformat() if s.last_used else None,
                    'metadata': s.metadata
                }
                for sid, s in self.strategies.items()
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save strategies: {e}")
    
    def _load_strategies(self):
        """Load strategies from disk."""
        try:
            if not self.storage_path.exists():
                return
            
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for sid, item in data.items():
                strategy = LearningStrategy(
                    strategy_id=item['strategy_id'],
                    name=item['name'],
                    strategy_type=StrategyType(item['strategy_type']),
                    applicable_to=item['applicable_to'],
                    steps=item['steps'],
                    success_rate=item['success_rate'],
                    avg_duration=timedelta(seconds=item['avg_duration']) if item['avg_duration'] else None,
                    usage_count=item['usage_count'],
                    last_used=datetime.fromisoformat(item['last_used']) if item.get('last_used') else None,
                    metadata=item.get('metadata', {})
                )
                self.strategies[sid] = strategy
            
            logger.info(f"Loaded {len(self.strategies)} strategies")
        
        except Exception as e:
            logger.error(f"Failed to load strategies: {e}")


class SelfImprover:
    """
    Self-improvement engine.
    
    Analyzes performance and generates improvement plans.
    """
    
    def __init__(
        self,
        performance_tracker: PerformanceTracker,
        strategy_library: StrategyLibrary,
        llm_function: Optional[Callable] = None
    ):
        self.tracker = performance_tracker
        self.library = strategy_library
        self.llm_function = llm_function
    
    async def analyze_and_improve(self) -> Dict[str, Any]:
        """
        Analyze performance and generate improvement plan.
        
        Returns:
            Improvement plan
        """
        # Identify weaknesses
        weaknesses = self.tracker.identify_weaknesses()
        
        if not weaknesses:
            return {'status': 'no_improvements_needed'}
        
        improvements = []
        
        for weakness in weaknesses[:3]:  # Focus on top 3
            task_type = weakness['task_type']
            
            # Check if we have effective strategies
            best_strategy = self.library.get_best_strategy(task_type)
            
            if not best_strategy or best_strategy.success_rate < 0.7:
                # Need to learn new strategy
                new_strategy = await self._learn_new_strategy(task_type, weakness)
                if new_strategy:
                    improvements.append({
                        'task_type': task_type,
                        'action': 'learned_new_strategy',
                        'strategy_id': new_strategy,
                        'previous_success_rate': weakness['success_rate']
                    })
            else:
                # Refine existing strategy
                improvements.append({
                    'task_type': task_type,
                    'action': 'use_best_strategy',
                    'strategy_id': best_strategy.strategy_id,
                    'strategy_success_rate': best_strategy.success_rate
                })
        
        return {
            'status': 'improvements_identified',
            'weaknesses_count': len(weaknesses),
            'improvements': improvements
        }
    
    async def _learn_new_strategy(
        self,
        task_type: str,
        performance_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Learn a new strategy for a task type.
        
        Args:
            task_type: Task type to learn strategy for
            performance_data: Performance statistics
        
        Returns:
            Strategy ID if created
        """
        # Use LLM to generate strategy
        if self.llm_function:
            prompt = f"""Analyze this task performance and create an improved strategy:

Task Type: {task_type}
Current Success Rate: {performance_data.get('success_rate', 0):.2%}
Average Duration: {performance_data.get('avg_duration_seconds', 0):.1f}s

Based on this data, suggest a step-by-step strategy to improve performance.
Provide:
1. Strategy name
2. Clear steps (5-10 steps)
3. Expected improvements

Format as JSON:
{{
  "name": "...",
  "steps": ["step 1", "step 2", ...],
  "expected_improvement": "..."
}}"""
            
            try:
                response = await self.llm_function(prompt)
                # Extract JSON
                import re
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    strategy_data = json.loads(json_match.group())
                    
                    strategy_id = self.library.add_strategy(
                        name=strategy_data['name'],
                        strategy_type=StrategyType.ACTIVE_LEARNING,
                        applicable_to=[task_type],
                        steps=strategy_data['steps']
                    )
                    
                    logger.info(f"Learned new strategy for {task_type}: {strategy_data['name']}")
                    return strategy_id
            
            except Exception as e:
                logger.error(f"Failed to learn new strategy: {e}")
        
        # Fallback: create generic improvement strategy
        strategy_id = self.library.add_strategy(
            name=f"Improved {task_type} Strategy",
            strategy_type=StrategyType.TRIAL_AND_ERROR,
            applicable_to=[task_type],
            steps=[
                "Review previous failures",
                "Identify common patterns",
                "Test alternative approach",
                "Measure results",
                "Iterate if needed"
            ]
        )
        
        return strategy_id


class MetaLearningSystem:
    """
    Complete meta-learning system.
    
    Coordinates performance tracking, strategy learning, and self-improvement.
    """
    
    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        llm_function: Optional[Callable] = None
    ):
        storage_dir = storage_dir or Path("./data/meta_learning")
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.tracker = PerformanceTracker(storage_dir / "performance.json")
        self.library = StrategyLibrary(storage_dir / "strategies.json")
        self.improver = SelfImprover(self.tracker, self.library, llm_function)
        
        self._improvement_task = None
    
    def record_task_performance(
        self,
        task_id: str,
        task_type: str,
        success: bool,
        duration: timedelta,
        metrics: Optional[Dict[PerformanceMetric, float]] = None,
        strategy_id: Optional[str] = None
    ):
        """Record task performance and update strategies."""
        # Determine strategy used
        if strategy_id:
            strategy = self.library.strategies.get(strategy_id)
            strategy_type = strategy.strategy_type if strategy else StrategyType.TRIAL_AND_ERROR
            
            # Update strategy performance
            self.library.update_strategy_performance(strategy_id, success, duration)
        else:
            strategy_type = StrategyType.TRIAL_AND_ERROR
        
        # Record performance
        self.tracker.record_performance(
            task_id=task_id,
            task_type=task_type,
            success=success,
            duration=duration,
            metrics=metrics or {},
            strategy_used=strategy_type
        )
    
    async def get_strategy_for_task(self, task_type: str) -> Optional[LearningStrategy]:
        """Get best strategy for a task type."""
        return self.library.get_best_strategy(task_type)
    
    async def self_improve(self) -> Dict[str, Any]:
        """Run self-improvement analysis."""
        return await self.improver.analyze_and_improve()
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning report."""
        # Task type statistics
        task_types = set(r.task_type for r in self.tracker.records)
        task_stats = {
            tt: self.tracker.get_task_type_stats(tt)
            for tt in task_types
        }
        
        # Overall statistics
        total_tasks = len(self.tracker.records)
        successful_tasks = sum(1 for r in self.tracker.records if r.success)
        overall_success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
        
        # Strategy statistics
        strategy_count = len(self.library.strategies)
        avg_strategy_success = np.mean([s.success_rate for s in self.library.strategies.values()]) if self.library.strategies else 0
        
        return {
            'total_tasks_performed': total_tasks,
            'overall_success_rate': overall_success_rate,
            'task_types_mastered': len([s for s in task_stats.values() if s.get('success_rate', 0) > 0.8]),
            'total_task_types': len(task_types),
            'strategies_learned': strategy_count,
            'avg_strategy_success_rate': avg_strategy_success,
            'task_stats': task_stats,
            'improvement_opportunities': self.tracker.identify_weaknesses()
        }
    
    async def start_continuous_improvement(self, interval_hours: int = 24):
        """Start continuous self-improvement."""
        async def improvement_loop():
            while True:
                await asyncio.sleep(interval_hours * 3600)
                result = await self.self_improve()
                logger.info(f"Self-improvement cycle: {result}")
                
                # Prune ineffective strategies
                self.library.prune_ineffective_strategies()
        
        self._improvement_task = asyncio.create_task(improvement_loop())
        logger.info(f"Started continuous improvement (every {interval_hours}h)")
    
    async def stop_continuous_improvement(self):
        """Stop continuous improvement."""
        if self._improvement_task:
            self._improvement_task.cancel()
            try:
                await self._improvement_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped continuous improvement")


# Example usage
async def main():
    """Example meta-learning system usage."""
    
    print("=" * 50)
    print("Meta-Learning System Example")
    print("=" * 50)
    
    # Initialize system
    meta_learning = MetaLearningSystem()
    
    # Simulate some task performances
    print("\n1. Recording task performances...")
    
    # Good performance on task A
    for i in range(5):
        meta_learning.record_task_performance(
            task_id=f"task_a_{i}",
            task_type="data_analysis",
            success=True,
            duration=timedelta(minutes=10),
            metrics={PerformanceMetric.ACCURACY: 0.9, PerformanceMetric.SPEED: 0.8}
        )
    
    # Poor performance on task B
    for i in range(5):
        meta_learning.record_task_performance(
            task_id=f"task_b_{i}",
            task_type="creative_writing",
            success=i >= 3,  # Only last 2 succeed
            duration=timedelta(minutes=30),
            metrics={PerformanceMetric.QUALITY: 0.4 + i*0.1}
        )
    
    print("  Recorded 10 task performances")
    
    # Add a strategy
    print("\n2. Adding learned strategy...")
    strategy_id = meta_learning.library.add_strategy(
        name="Iterative Analysis",
        strategy_type=StrategyType.ACTIVE_LEARNING,
        applicable_to=["data_analysis"],
        steps=[
            "Load and inspect data",
            "Identify patterns",
            "Generate hypotheses",
            "Test hypotheses",
            "Report findings"
        ]
    )
    print(f"  Added strategy: {strategy_id}")
    
    # Get strategy for task
    print("\n3. Getting best strategy...")
    strategy = await meta_learning.get_strategy_for_task("data_analysis")
    if strategy:
        print(f"  Best strategy: {strategy.name}")
        print(f"  Success rate: {strategy.success_rate:.2%}")
    
    # Self-improve
    print("\n4. Running self-improvement...")
    improvement_result = await meta_learning.self_improve()
    print(f"  Status: {improvement_result.get('status')}")
    if 'improvements' in improvement_result:
        print(f"  Improvements identified: {len(improvement_result['improvements'])}")
        for imp in improvement_result['improvements']:
            print(f"    - {imp['task_type']}: {imp['action']}")
    
    # Generate learning report
    print("\n5. Learning report...")
    report = meta_learning.get_learning_report()
    print(f"  Total tasks: {report['total_tasks_performed']}")
    print(f"  Overall success rate: {report['overall_success_rate']:.2%}")
    print(f"  Task types mastered: {report['task_types_mastered']}/{report['total_task_types']}")
    print(f"  Strategies learned: {report['strategies_learned']}")
    print(f"  Avg strategy success: {report['avg_strategy_success_rate']:.2%}")
    
    if report['improvement_opportunities']:
        print(f"\n  Improvement opportunities:")
        for opp in report['improvement_opportunities']:
            print(f"    - {opp['task_type']}: {opp['success_rate']:.2%} success rate")
    
    print("\n✅ Meta-learning system example completed!")


if __name__ == "__main__":
    asyncio.run(main())
