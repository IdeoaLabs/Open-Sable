"""
Metacognition - Self-awareness, error detection, and adaptive recovery for AGI.

Features:
- Self-monitoring of reasoning processes
- Error detection and classification
- Confidence calibration
- Decision quality assessment
- Adaptive error recovery
- Performance introspection
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors the system can detect."""
    LOGIC_ERROR = "logic_error"
    KNOWLEDGE_GAP = "knowledge_gap"
    AMBIGUITY = "ambiguity"
    CONTRADICTION = "contradiction"
    RESOURCE_LIMIT = "resource_limit"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    UNEXPECTED_OUTPUT = "unexpected_output"


class ConfidenceLevel(Enum):
    """Confidence levels in decisions/outputs."""
    VERY_HIGH = 5  # > 0.9
    HIGH = 4       # 0.7 - 0.9
    MEDIUM = 3     # 0.5 - 0.7
    LOW = 2        # 0.3 - 0.5
    VERY_LOW = 1   # < 0.3


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    BACKTRACK = "backtrack"
    ASK_FOR_HELP = "ask_for_help"
    USE_FALLBACK = "use_fallback"
    SKIP = "skip"
    ABORT = "abort"


@dataclass
class ThoughtTrace:
    """Trace of reasoning steps."""
    trace_id: str
    task: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    final_answer: Optional[Any] = None
    confidence: float = 0.5
    errors_detected: List[str] = field(default_factory=list)
    
    def add_step(
        self,
        step_type: str,
        content: str,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a reasoning step."""
        self.steps.append({
            'step_number': len(self.steps) + 1,
            'type': step_type,
            'content': content,
            'confidence': confidence,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        })
    
    def complete(self, final_answer: Any, confidence: float):
        """Mark trace as complete."""
        self.end_time = datetime.utcnow()
        self.final_answer = final_answer
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trace_id': self.trace_id,
            'task': self.task,
            'steps': self.steps,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'final_answer': str(self.final_answer),
            'confidence': self.confidence,
            'errors_detected': self.errors_detected
        }


@dataclass
class ErrorReport:
    """Report of detected error."""
    error_id: str
    error_type: ErrorType
    description: str
    severity: int  # 1-10
    context: Dict[str, Any]
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolution: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_id': self.error_id,
            'error_type': self.error_type.value,
            'description': self.description,
            'severity': self.severity,
            'context': self.context,
            'detected_at': self.detected_at.isoformat(),
            'resolved': self.resolved,
            'resolution': self.resolution
        }


class SelfMonitor:
    """
    Monitors reasoning processes in real-time.
    
    Tracks confidence, detects anomalies, and flags potential errors.
    """
    
    def __init__(self):
        self.active_traces: Dict[str, ThoughtTrace] = {}
        self.completed_traces: List[ThoughtTrace] = []
        self.monitoring_rules: List[Dict[str, Any]] = []
        
        # Default monitoring rules
        self._load_default_rules()
    
    def start_trace(self, task: str) -> str:
        """Start monitoring a reasoning trace."""
        trace_id = f"trace_{len(self.active_traces)}_{datetime.utcnow().timestamp()}"
        
        trace = ThoughtTrace(
            trace_id=trace_id,
            task=task
        )
        
        self.active_traces[trace_id] = trace
        logger.debug(f"Started trace: {trace_id}")
        return trace_id
    
    def add_step(
        self,
        trace_id: str,
        step_type: str,
        content: str,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add step to active trace."""
        if trace_id not in self.active_traces:
            logger.warning(f"Trace not found: {trace_id}")
            return
        
        trace = self.active_traces[trace_id]
        trace.add_step(step_type, content, confidence, metadata)
        
        # Check monitoring rules
        self._apply_monitoring_rules(trace)
    
    def complete_trace(
        self,
        trace_id: str,
        final_answer: Any,
        confidence: float
    ):
        """Complete a reasoning trace."""
        if trace_id not in self.active_traces:
            return
        
        trace = self.active_traces[trace_id]
        trace.complete(final_answer, confidence)
        
        # Move to completed
        self.completed_traces.append(trace)
        del self.active_traces[trace_id]
        
        # Keep only recent traces
        if len(self.completed_traces) > 100:
            self.completed_traces = self.completed_traces[-100:]
        
        logger.debug(f"Completed trace: {trace_id}")
    
    def _apply_monitoring_rules(self, trace: ThoughtTrace):
        """Apply monitoring rules to detect issues."""
        for rule in self.monitoring_rules:
            rule_type = rule['type']
            
            if rule_type == 'low_confidence':
                # Check for low confidence steps
                if trace.steps:
                    last_step = trace.steps[-1]
                    if last_step['confidence'] < rule['threshold']:
                        trace.errors_detected.append(
                            f"Low confidence detected: {last_step['confidence']:.2f}"
                        )
            
            elif rule_type == 'contradictory_steps':
                # Check for contradictions (simplified)
                if len(trace.steps) >= 2:
                    recent_contents = [s['content'].lower() for s in trace.steps[-2:]]
                    if 'not' in recent_contents[1] and any(
                        word in recent_contents[0] for word in recent_contents[1].split()
                    ):
                        trace.errors_detected.append("Possible contradiction detected")
            
            elif rule_type == 'stuck_in_loop':
                # Check for repeated steps
                if len(trace.steps) >= 3:
                    recent_types = [s['type'] for s in trace.steps[-3:]]
                    if len(set(recent_types)) == 1:
                        trace.errors_detected.append("Possible reasoning loop detected")
    
    def _load_default_rules(self):
        """Load default monitoring rules."""
        self.monitoring_rules = [
            {'type': 'low_confidence', 'threshold': 0.3},
            {'type': 'contradictory_steps'},
            {'type': 'stuck_in_loop'}
        ]
    
    def get_trace(self, trace_id: str) -> Optional[ThoughtTrace]:
        """Get trace by ID."""
        if trace_id in self.active_traces:
            return self.active_traces[trace_id]
        
        for trace in self.completed_traces:
            if trace.trace_id == trace_id:
                return trace
        
        return None
    
    def get_recent_traces(self, n: int = 10) -> List[ThoughtTrace]:
        """Get n most recent traces."""
        return self.completed_traces[-n:]


class ErrorDetector:
    """
    Detects various types of errors in reasoning and execution.
    
    Analyzes outputs, checks consistency, validates logic.
    """
    
    def __init__(self):
        self.error_reports: List[ErrorReport] = []
        self.error_patterns: Dict[ErrorType, List[str]] = {}
        
        self._load_error_patterns()
    
    def detect_errors(
        self,
        output: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ErrorReport]:
        """
        Detect errors in output.
        
        Args:
            output: Output to analyze
            context: Additional context
        
        Returns:
            List of detected errors
        """
        errors = []
        
        # Check for contradictions
        if self._has_contradiction(output):
            errors.append(self._create_error_report(
                ErrorType.CONTRADICTION,
                "Output contains contradictory statements",
                severity=8,
                context={'output': output[:200]}
            ))
        
        # Check for knowledge gaps
        uncertainty_phrases = ['not sure', 'don\'t know', 'unclear', 'uncertain']
        if any(phrase in output.lower() for phrase in uncertainty_phrases):
            errors.append(self._create_error_report(
                ErrorType.KNOWLEDGE_GAP,
                "Uncertainty detected in output",
                severity=5,
                context={'output': output[:200]}
            ))
        
        # Check for ambiguity
        if self._is_ambiguous(output):
            errors.append(self._create_error_report(
                ErrorType.AMBIGUITY,
                "Ambiguous output detected",
                severity=4,
                context={'output': output[:200]}
            ))
        
        # Store error reports
        self.error_reports.extend(errors)
        
        return errors
    
    def detect_logical_error(
        self,
        premise: str,
        conclusion: str
    ) -> Optional[ErrorReport]:
        """Detect logical errors in reasoning."""
        # Simple logical checks (could be enhanced)
        
        # Check for affirming the consequent
        if 'if' in premise.lower() and 'then' in premise.lower():
            # Extract parts
            parts = premise.lower().split('then')
            if len(parts) == 2:
                antecedent = parts[0].replace('if', '').strip()
                consequent = parts[1].strip()
                
                # If conclusion affirms consequent, potential error
                if consequent in conclusion.lower() and antecedent not in conclusion.lower():
                    return self._create_error_report(
                        ErrorType.LOGIC_ERROR,
                        "Possible logical fallacy: affirming the consequent",
                        severity=7,
                        context={'premise': premise, 'conclusion': conclusion}
                    )
        
        return None
    
    def classify_error(self, error_message: str) -> ErrorType:
        """Classify error from error message."""
        error_lower = error_message.lower()
        
        if 'timeout' in error_lower:
            return ErrorType.TIMEOUT
        elif 'invalid' in error_lower or 'validation' in error_lower:
            return ErrorType.INVALID_INPUT
        elif 'memory' in error_lower or 'resource' in error_lower:
            return ErrorType.RESOURCE_LIMIT
        else:
            return ErrorType.UNEXPECTED_OUTPUT
    
    def _has_contradiction(self, text: str) -> bool:
        """Check for contradictions."""
        # Simple check for negation patterns
        sentences = text.split('.')
        
        for i in range(len(sentences) - 1):
            sent1 = sentences[i].lower()
            sent2 = sentences[i + 1].lower()
            
            # Check for negation of previous statement
            if 'not' in sent2 and any(
                word in sent2 for word in sent1.split()
                if len(word) > 4 and word not in ['that', 'this', 'with']
            ):
                return True
        
        return False
    
    def _is_ambiguous(self, text: str) -> bool:
        """Check for ambiguity."""
        ambiguous_phrases = [
            'could be', 'might be', 'possibly', 'perhaps',
            'may', 'it depends', 'various', 'multiple'
        ]
        
        # Count ambiguous phrases
        count = sum(1 for phrase in ambiguous_phrases if phrase in text.lower())
        
        # Ambiguous if multiple uncertainty markers
        return count >= 3
    
    def _create_error_report(
        self,
        error_type: ErrorType,
        description: str,
        severity: int,
        context: Dict[str, Any]
    ) -> ErrorReport:
        """Create error report."""
        error_id = f"err_{len(self.error_reports)}_{datetime.utcnow().timestamp()}"
        
        return ErrorReport(
            error_id=error_id,
            error_type=error_type,
            description=description,
            severity=severity,
            context=context
        )
    
    def _load_error_patterns(self):
        """Load common error patterns."""
        self.error_patterns = {
            ErrorType.CONTRADICTION: ['but', 'however', 'although', 'not'],
            ErrorType.KNOWLEDGE_GAP: ['unsure', 'don\'t know', 'unclear'],
            ErrorType.AMBIGUITY: ['possibly', 'maybe', 'could be']
        }


class ConfidenceCalibrator:
    """
    Calibrates confidence scores based on historical accuracy.
    
    Adjusts overconfidence or underconfidence.
    """
    
    def __init__(self):
        self.predictions: List[Dict[str, Any]] = []
        self.calibration_curve: Dict[float, float] = {}
    
    def record_prediction(
        self,
        predicted_confidence: float,
        actual_correctness: bool
    ):
        """Record a prediction for calibration."""
        self.predictions.append({
            'confidence': predicted_confidence,
            'correct': actual_correctness,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Update calibration curve periodically
        if len(self.predictions) % 10 == 0:
            self._update_calibration_curve()
    
    def calibrate_confidence(self, raw_confidence: float) -> float:
        """Calibrate raw confidence score."""
        if not self.calibration_curve:
            return raw_confidence
        
        # Find closest calibrated value
        closest_raw = min(
            self.calibration_curve.keys(),
            key=lambda x: abs(x - raw_confidence)
        )
        
        return self.calibration_curve[closest_raw]
    
    def get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to level."""
        calibrated = self.calibrate_confidence(confidence)
        
        if calibrated > 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif calibrated > 0.7:
            return ConfidenceLevel.HIGH
        elif calibrated > 0.5:
            return ConfidenceLevel.MEDIUM
        elif calibrated > 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _update_calibration_curve(self):
        """Update calibration curve from predictions."""
        # Group by confidence buckets
        buckets = {}
        bucket_size = 0.1
        
        for pred in self.predictions:
            bucket = int(pred['confidence'] / bucket_size) * bucket_size
            if bucket not in buckets:
                buckets[bucket] = {'correct': 0, 'total': 0}
            
            buckets[bucket]['total'] += 1
            if pred['correct']:
                buckets[bucket]['correct'] += 1
        
        # Calculate actual accuracy for each bucket
        self.calibration_curve = {
            bucket: data['correct'] / data['total']
            for bucket, data in buckets.items()
            if data['total'] >= 5  # Minimum samples for calibration
        }


class ErrorRecovery:
    """
    Implements error recovery strategies.
    
    Attempts to recover from detected errors.
    """
    
    def __init__(self):
        self.recovery_history: List[Dict[str, Any]] = []
    
    async def recover_from_error(
        self,
        error: ErrorReport,
        context: Dict[str, Any],
        retry_function: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Attempt to recover from an error.
        
        Args:
            error: Error to recover from
            context: Execution context
            retry_function: Function to retry
        
        Returns:
            Recovery result
        """
        # Select recovery strategy
        strategy = self._select_strategy(error)
        
        logger.info(f"Attempting recovery for {error.error_type.value} using {strategy.value}")
        
        result = {
            'error_id': error.error_id,
            'strategy': strategy.value,
            'success': False
        }
        
        try:
            if strategy == RecoveryStrategy.RETRY and retry_function:
                # Retry with backoff
                await asyncio.sleep(1)
                retry_result = await retry_function()
                result['success'] = True
                result['output'] = retry_result
            
            elif strategy == RecoveryStrategy.USE_FALLBACK:
                # Use fallback approach
                result['success'] = True
                result['output'] = "Using fallback approach"
            
            elif strategy == RecoveryStrategy.ASK_FOR_HELP:
                # Request human assistance
                result['success'] = False
                result['message'] = "Human assistance required"
            
            elif strategy == RecoveryStrategy.SKIP:
                # Skip this step
                result['success'] = True
                result['output'] = "Skipped problematic step"
            
            elif strategy == RecoveryStrategy.BACKTRACK:
                # Backtrack to earlier state
                result['success'] = True
                result['output'] = "Backtracked to safe state"
            
            else:  # ABORT
                result['success'] = False
                result['message'] = "Recovery failed, aborting"
        
        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
        
        # Record recovery attempt
        self.recovery_history.append({
            'error_type': error.error_type.value,
            'strategy': strategy.value,
            'success': result['success'],
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return result
    
    def _select_strategy(self, error: ErrorReport) -> RecoveryStrategy:
        """Select appropriate recovery strategy."""
        # Strategy selection based on error type and severity
        if error.severity >= 8:
            return RecoveryStrategy.ABORT
        
        strategy_map = {
            ErrorType.TIMEOUT: RecoveryStrategy.RETRY,
            ErrorType.RESOURCE_LIMIT: RecoveryStrategy.USE_FALLBACK,
            ErrorType.KNOWLEDGE_GAP: RecoveryStrategy.ASK_FOR_HELP,
            ErrorType.INVALID_INPUT: RecoveryStrategy.BACKTRACK,
            ErrorType.LOGIC_ERROR: RecoveryStrategy.BACKTRACK,
            ErrorType.CONTRADICTION: RecoveryStrategy.RETRY,
            ErrorType.AMBIGUITY: RecoveryStrategy.SKIP
        }
        
        return strategy_map.get(error.error_type, RecoveryStrategy.USE_FALLBACK)


class MetacognitiveSystem:
    """
    Metacognitive system for self-awareness and adaptive behavior.
    Monitors reasoning, detects errors, and adjusts strategies.
    """
    
    def __init__(self, config):
        self.config = config
        self.traces: Dict[str, ThoughtTrace] = {}
        self.error_history: List[Dict] = []
        self.performance_stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'errors_detected': 0,
            'errors_recovered': 0
        }
    
    async def initialize(self):
        """Initialize metacognitive system"""
        logger.info("Metacognitive system initialized")
    
    async def assess_confidence(self, response: str, task: str) -> float:
        """
        Assess confidence in a response.
        Returns value 0.0-1.0
        
        Heuristics:
        - Length of response (too short = low confidence)
        - Contains actual data vs just URLs
        - Number of information sources
        """
        confidence = 0.5  # baseline
        
        # Check response length
        if len(response) < 100:
            confidence -= 0.2
        elif len(response) > 500:
            confidence += 0.2
        
        # Check if response has actual content vs just URLs
        import re
        urls = re.findall(r'https?://[^\s]+', response)
        text_without_urls = re.sub(r'https?://[^\s]+', '', response)
        
        if len(urls) > 0 and len(text_without_urls.strip()) < 200:
            # Mostly URLs, little actual content
            confidence -= 0.3
            logger.info(f"🧠 Low confidence: Response is mostly URLs without content")
        
        # Check if response actually answers the question
        task_words = set(task.lower().split())
        response_words = set(response.lower().split())
        overlap = len(task_words & response_words)
        
        if overlap < 2:
            confidence -= 0.2
        
        # Clamp to 0-1
        return max(0.0, min(1.0, confidence))
    
    async def evaluate_response_quality(self, task: str, response: str) -> Dict[str, Any]:
        """
        Evaluate if response sufficiently answers the task.
        
        Returns:
            Dict with 'sufficient', 'needs_more', 'confidence'
        """
        confidence = await self.assess_confidence(response, task)
        
        # Determine if we need more info
        needs_scraping = False
        if confidence < 0.7:
            # Check if response has URLs that could be scraped
            import re
            urls = re.findall(r'https?://[^\s<>"]+', response)
            if len(urls) > 0:
                needs_scraping = True
                logger.info(f"🧠 Metacognition suggests scraping {len(urls)} URLs for better response")
        
        return {
            "sufficient": confidence >= 0.7,
            "needs_more": "scrape" if needs_scraping else "none",
            "confidence": int(confidence * 100)
        }


class MetacognitiveEngine:
    """
    Complete metacognitive system.
    
    Integrates self-monitoring, error detection, and recovery.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.monitor = SelfMonitor()
        self.detector = ErrorDetector()
        self.calibrator = ConfidenceCalibrator()
        self.recovery = ErrorRecovery()
        
        self.storage_path = storage_path or Path("./data/metacognition.json")
        self._load_state()
    
    def start_monitoring_task(self, task: str) -> str:
        """Start monitoring a task."""
        return self.monitor.start_trace(task)
    
    def record_thought_step(
        self,
        trace_id: str,
        step_type: str,
        content: str,
        raw_confidence: float
    ):
        """Record a thought/reasoning step."""
        # Calibrate confidence
        calibrated_confidence = self.calibrator.calibrate_confidence(raw_confidence)
        
        # Add to monitor
        self.monitor.add_step(
            trace_id, step_type, content, calibrated_confidence
        )
        
        # Check for errors in content
        errors = self.detector.detect_errors(content)
        
        if errors:
            logger.warning(f"Detected {len(errors)} errors in step")
            
            # Add to trace
            trace = self.monitor.get_trace(trace_id)
            if trace:
                trace.errors_detected.extend([e.description for e in errors])
    
    async def complete_task(
        self,
        trace_id: str,
        final_answer: Any,
        raw_confidence: float,
        actual_correctness: Optional[bool] = None
    ):
        """Complete monitored task."""
        # Calibrate confidence
        calibrated_confidence = self.calibrator.calibrate_confidence(raw_confidence)
        
        # Complete trace
        self.monitor.complete_trace(trace_id, final_answer, calibrated_confidence)
        
        # Record for calibration if we know correctness
        if actual_correctness is not None:
            self.calibrator.record_prediction(raw_confidence, actual_correctness)
        
        # Detect errors in final answer
        errors = self.detector.detect_errors(str(final_answer))
        
        # Attempt recovery if errors found
        if errors:
            for error in errors:
                if error.severity >= 7:
                    recovery_result = await self.recovery.recover_from_error(
                        error, {'trace_id': trace_id}
                    )
                    logger.info(f"Recovery result: {recovery_result}")
        
        self._save_state()
    
    def get_confidence_level(self, raw_confidence: float) -> str:
        """Get confidence level description."""
        level = self.calibrator.get_confidence_level(raw_confidence)
        return level.name
    
    def get_introspection_report(self) -> Dict[str, Any]:
        """Generate introspection report."""
        recent_traces = self.monitor.get_recent_traces(20)
        
        # Analyze recent performance
        total_errors = sum(len(t.errors_detected) for t in recent_traces)
        avg_confidence = sum(t.confidence for t in recent_traces) / len(recent_traces) if recent_traces else 0
        
        # Recovery success rate
        recovery_attempts = len(self.recovery.recovery_history)
        recovery_successes = sum(
            1 for r in self.recovery.recovery_history if r['success']
        )
        recovery_rate = recovery_successes / recovery_attempts if recovery_attempts > 0 else 0
        
        return {
            'recent_traces': len(recent_traces),
            'total_errors_detected': total_errors,
            'avg_confidence': avg_confidence,
            'confidence_calibration': len(self.calibrator.calibration_curve) > 0,
            'recovery_attempts': recovery_attempts,
            'recovery_success_rate': recovery_rate,
            'active_traces': len(self.monitor.active_traces),
            'error_reports': len(self.detector.error_reports),
            'unresolved_errors': sum(1 for e in self.detector.error_reports if not e.resolved)
        }
    
    def _save_state(self):
        """Save metacognitive state."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'completed_traces': [t.to_dict() for t in self.monitor.completed_traces[-50:]],
                'error_reports': [e.to_dict() for e in self.detector.error_reports[-100:]],
                'calibration_predictions': self.calibrator.predictions[-500:],
                'recovery_history': self.recovery.recovery_history[-100:]
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save metacognitive state: {e}")
    
    def _load_state(self):
        """Load metacognitive state."""
        try:
            if not self.storage_path.exists():
                return
            
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            # Load predictions for calibration
            self.calibrator.predictions = data.get('calibration_predictions', [])
            self.calibrator._update_calibration_curve()
            
            # Load recovery history
            self.recovery.recovery_history = data.get('recovery_history', [])
            
            logger.info("Loaded metacognitive state")
        
        except Exception as e:
            logger.error(f"Failed to load metacognitive state: {e}")


# Example usage
async def main():
    """Example metacognitive system usage."""
    
    print("=" * 50)
    print("Metacognitive System Example")
    print("=" * 50)
    
    # Initialize system
    metacog = MetacognitiveSystem()
    
    # Start monitoring a task
    print("\n1. Starting task monitoring...")
    trace_id = metacog.start_monitoring_task("Solve math problem")
    print(f"  Trace ID: {trace_id}")
    
    # Record reasoning steps
    print("\n2. Recording reasoning steps...")
    metacog.record_thought_step(
        trace_id, "understanding", "The problem asks for the sum of 5 and 3", 0.9
    )
    metacog.record_thought_step(
        trace_id, "calculation", "5 + 3 = 8", 0.95
    )
    metacog.record_thought_step(
        trace_id, "verification", "Let me verify: 8 - 3 = 5, correct", 0.85
    )
    print("  Recorded 3 steps")
    
    # Complete task
    print("\n3. Completing task...")
    await metacog.complete_task(trace_id, "The answer is 8", 0.9, actual_correctness=True)
    print("  Task completed")
    
    # Demonstrate error detection
    print("\n4. Testing error detection...")
    trace_id2 = metacog.start_monitoring_task("Analyze statement")
    metacog.record_thought_step(
        trace_id2, "analysis",
        "The sky is blue. However, the sky is not blue.",
        0.5
    )
    await metacog.complete_task(trace_id2, "Inconclusive", 0.3)
    print("  Errors detected in contradictory statement")
    
    # Get confidence level
    print("\n5. Confidence calibration...")
    level = metacog.get_confidence_level(0.75)
    print(f"  Confidence 0.75 → {level}")
    
    # Get introspection report
    print("\n6. Introspection report...")
    report = metacog.get_introspection_report()
    print(f"  Recent traces: {report['recent_traces']}")
    print(f"  Errors detected: {report['total_errors_detected']}")
    print(f"  Avg confidence: {report['avg_confidence']:.2f}")
    print(f"  Recovery success rate: {report['recovery_success_rate']:.2%}")
    print(f"  Unresolved errors: {report['unresolved_errors']}")
    
    print("\n✅ Metacognitive system example completed!")


if __name__ == "__main__":
    asyncio.run(main())
