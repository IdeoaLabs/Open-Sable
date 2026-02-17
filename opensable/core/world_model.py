"""
World Model - Internal model of environment state and predictive capabilities.

Features:
- Environment state tracking
- Entity tracking (objects, agents, events)
- Causal reasoning
- State prediction
- Counterfactual reasoning
- World simulation
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities in the world model."""
    OBJECT = "object"
    AGENT = "agent"
    EVENT = "event"
    LOCATION = "location"
    CONCEPT = "concept"


class RelationType(Enum):
    """Types of relations between entities."""
    IS_A = "is_a"
    HAS = "has"
    LOCATED_AT = "located_at"
    CAUSES = "causes"
    REQUIRES = "requires"
    PRECEDES = "precedes"
    SIMILAR_TO = "similar_to"


@dataclass
class Entity:
    """Entity in the world model."""
    entity_id: str
    entity_type: EntityType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 1.0  # Confidence in entity existence
    
    def update_property(self, key: str, value: Any):
        """Update entity property."""
        self.properties[key] = value
        self.last_updated = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entity_id': self.entity_id,
            'entity_type': self.entity_type.value,
            'name': self.name,
            'properties': self.properties,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'confidence': self.confidence
        }


@dataclass
class Relation:
    """Relation between entities."""
    relation_id: str
    relation_type: RelationType
    source_id: str
    target_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'relation_id': self.relation_id,
            'relation_type': self.relation_type.value,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'properties': self.properties,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence
        }


@dataclass
class WorldState:
    """Snapshot of world state at a point in time."""
    state_id: str
    timestamp: datetime
    entities: Dict[str, Entity]
    relations: Dict[str, Relation]
    global_properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'state_id': self.state_id,
            'timestamp': self.timestamp.isoformat(),
            'entities': {eid: e.to_dict() for eid, e in self.entities.items()},
            'relations': {rid: r.to_dict() for rid, r in self.relations.items()},
            'global_properties': self.global_properties
        }


class StateTracker:
    """
    Tracks the current state of the world.
    
    Maintains entities, relations, and state changes.
    """
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.state_history: List[WorldState] = []
        self.global_properties: Dict[str, Any] = {}
    
    def add_entity(
        self,
        entity_type: EntityType,
        name: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add entity to world model."""
        entity_id = self._generate_id(f"{entity_type.value}_{name}")
        
        entity = Entity(
            entity_id=entity_id,
            entity_type=entity_type,
            name=name,
            properties=properties or {}
        )
        
        self.entities[entity_id] = entity
        logger.debug(f"Added entity: {name} ({entity_type.value})")
        return entity_id
    
    def update_entity(
        self,
        entity_id: str,
        properties: Dict[str, Any]
    ):
        """Update entity properties."""
        if entity_id in self.entities:
            for key, value in properties.items():
                self.entities[entity_id].update_property(key, value)
    
    def add_relation(
        self,
        relation_type: RelationType,
        source_id: str,
        target_id: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add relation between entities."""
        relation_id = f"{source_id}_{relation_type.value}_{target_id}"
        
        relation = Relation(
            relation_id=relation_id,
            relation_type=relation_type,
            source_id=source_id,
            target_id=target_id,
            properties=properties or {}
        )
        
        self.relations[relation_id] = relation
        logger.debug(f"Added relation: {source_id} {relation_type.value} {target_id}")
        return relation_id
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)
    
    def find_entities(
        self,
        entity_type: Optional[EntityType] = None,
        name_pattern: Optional[str] = None,
        property_filter: Optional[Dict[str, Any]] = None
    ) -> List[Entity]:
        """Find entities matching criteria."""
        entities = list(self.entities.values())
        
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]
        
        if name_pattern:
            entities = [e for e in entities if name_pattern.lower() in e.name.lower()]
        
        if property_filter:
            for key, value in property_filter.items():
                entities = [e for e in entities if e.properties.get(key) == value]
        
        return entities
    
    def get_relations(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None,
        as_source: bool = True,
        as_target: bool = True
    ) -> List[Relation]:
        """Get relations involving an entity."""
        relations = []
        
        for relation in self.relations.values():
            if relation_type and relation.relation_type != relation_type:
                continue
            
            if as_source and relation.source_id == entity_id:
                relations.append(relation)
            elif as_target and relation.target_id == entity_id:
                relations.append(relation)
        
        return relations
    
    def snapshot_state(self) -> str:
        """Create snapshot of current state."""
        state_id = f"state_{len(self.state_history)}_{datetime.utcnow().timestamp()}"
        
        # Deep copy current state
        snapshot = WorldState(
            state_id=state_id,
            timestamp=datetime.utcnow(),
            entities={eid: Entity(**e.to_dict()) for eid, e in self.entities.items()},
            relations={rid: Relation(**r.to_dict()) for rid, r in self.relations.items()},
            global_properties=self.global_properties.copy()
        )
        
        self.state_history.append(snapshot)
        
        # Keep only recent snapshots (max 100)
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]
        
        return state_id
    
    def get_state_at_time(self, timestamp: datetime) -> Optional[WorldState]:
        """Get state closest to given timestamp."""
        if not self.state_history:
            return None
        
        # Find closest state
        closest = min(
            self.state_history,
            key=lambda s: abs((s.timestamp - timestamp).total_seconds())
        )
        
        return closest
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID."""
        return hashlib.sha256(f"{content}_{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16]


class CausalReasoner:
    """
    Performs causal reasoning on world model.
    
    Infers cause-effect relationships.
    """
    
    def __init__(self, state_tracker: StateTracker):
        self.tracker = state_tracker
        self.causal_rules: List[Dict[str, Any]] = []
    
    def infer_causality(
        self,
        event1_id: str,
        event2_id: str,
        temporal_window: timedelta = timedelta(seconds=10)
    ) -> float:
        """
        Infer if event1 might cause event2.
        
        Returns confidence score 0-1.
        """
        event1 = self.tracker.get_entity(event1_id)
        event2 = self.tracker.get_entity(event2_id)
        
        if not event1 or not event2:
            return 0.0
        
        # Check temporal ordering
        if event1.created_at > event2.created_at:
            return 0.0  # Can't cause future events
        
        time_diff = event2.created_at - event1.created_at
        if time_diff > temporal_window:
            return 0.0  # Too far apart
        
        # Simple heuristic - closer in time = higher causality
        score = 1.0 - (time_diff.total_seconds() / temporal_window.total_seconds())
        
        # Check for existing causal relations
        causal_relations = self.tracker.get_relations(
            event1_id,
            relation_type=RelationType.CAUSES,
            as_source=True
        )
        
        if any(r.target_id == event2_id for r in causal_relations):
            score = 1.0  # Explicit causal relation exists
        
        return score
    
    def add_causal_rule(
        self,
        condition: Dict[str, Any],
        effect: Dict[str, Any],
        confidence: float = 1.0
    ):
        """Add causal rule."""
        self.causal_rules.append({
            'condition': condition,
            'effect': effect,
            'confidence': confidence
        })
    
    def apply_causal_rules(
        self,
        current_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply causal rules to predict effects."""
        predictions = []
        
        for rule in self.causal_rules:
            # Check if condition matches
            condition_met = all(
                current_state.get(k) == v
                for k, v in rule['condition'].items()
            )
            
            if condition_met:
                predictions.append({
                    'effect': rule['effect'],
                    'confidence': rule['confidence'],
                    'rule': rule
                })
        
        return predictions


class StatePredictor:
    """
    Predicts future states based on current state and dynamics.
    
    Uses learned patterns and causal models.
    """
    
    def __init__(
        self,
        state_tracker: StateTracker,
        causal_reasoner: CausalReasoner
    ):
        self.tracker = state_tracker
        self.reasoner = causal_reasoner
        self.prediction_history: List[Dict[str, Any]] = []
    
    async def predict_next_state(
        self,
        time_delta: timedelta = timedelta(minutes=5)
    ) -> WorldState:
        """
        Predict world state after time_delta.
        
        Args:
            time_delta: Time into future to predict
        
        Returns:
            Predicted world state
        """
        # Start with current state
        current_entities = {eid: Entity(**e.to_dict()) for eid, e in self.tracker.entities.items()}
        current_relations = {rid: Relation(**r.to_dict()) for rid, r in self.tracker.relations.items()}
        
        # Apply state transitions
        # (Simple implementation - could be enhanced with ML)
        
        # Example: agents move, objects change state, etc.
        for entity in current_entities.values():
            if entity.entity_type == EntityType.AGENT:
                # Predict agent movement
                if 'velocity' in entity.properties:
                    # Update position
                    pass
            
            elif entity.entity_type == EntityType.OBJECT:
                # Predict object state changes
                if 'decay_rate' in entity.properties:
                    # Apply decay
                    pass
        
        # Apply causal rules
        current_state_dict = {
            'entities': len(current_entities),
            'time': datetime.utcnow()
        }
        
        predicted_effects = self.reasoner.apply_causal_rules(current_state_dict)
        
        # Create predicted state
        predicted_state = WorldState(
            state_id=f"predicted_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow() + time_delta,
            entities=current_entities,
            relations=current_relations,
            global_properties={'prediction': True, 'time_delta': time_delta.total_seconds()}
        )
        
        # Record prediction
        self.prediction_history.append({
            'predicted_state_id': predicted_state.state_id,
            'prediction_time': datetime.utcnow().isoformat(),
            'target_time': predicted_state.timestamp.isoformat(),
            'effects': predicted_effects
        })
        
        return predicted_state
    
    def evaluate_predictions(
        self,
        actual_state: WorldState,
        predicted_state: WorldState
    ) -> Dict[str, Any]:
        """Evaluate prediction accuracy."""
        # Compare entities
        entity_accuracy = self._compare_entities(
            actual_state.entities,
            predicted_state.entities
        )
        
        # Compare relations
        relation_accuracy = self._compare_relations(
            actual_state.relations,
            predicted_state.relations
        )
        
        return {
            'entity_accuracy': entity_accuracy,
            'relation_accuracy': relation_accuracy,
            'overall_accuracy': (entity_accuracy + relation_accuracy) / 2
        }
    
    def _compare_entities(
        self,
        actual: Dict[str, Entity],
        predicted: Dict[str, Entity]
    ) -> float:
        """Compare entity sets."""
        if not actual:
            return 1.0 if not predicted else 0.0
        
        # Count matches
        matches = sum(
            1 for eid in actual.keys()
            if eid in predicted and self._entities_match(actual[eid], predicted[eid])
        )
        
        return matches / len(actual)
    
    def _compare_relations(
        self,
        actual: Dict[str, Relation],
        predicted: Dict[str, Relation]
    ) -> float:
        """Compare relation sets."""
        if not actual:
            return 1.0 if not predicted else 0.0
        
        matches = sum(
            1 for rid in actual.keys()
            if rid in predicted
        )
        
        return matches / len(actual)
    
    def _entities_match(self, e1: Entity, e2: Entity, threshold: float = 0.8) -> bool:
        """Check if entities match."""
        if e1.entity_type != e2.entity_type:
            return False
        
        if e1.name != e2.name:
            return False
        
        # Compare properties
        common_keys = set(e1.properties.keys()) & set(e2.properties.keys())
        if not common_keys:
            return True
        
        matches = sum(
            1 for k in common_keys
            if e1.properties[k] == e2.properties[k]
        )
        
        return (matches / len(common_keys)) >= threshold


class WorldModel:
    """
    Complete world model system.
    
    Integrates state tracking, causal reasoning, and prediction.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.tracker = StateTracker()
        self.reasoner = CausalReasoner(self.tracker)
        self.predictor = StatePredictor(self.tracker, self.reasoner)
        
        self.storage_path = storage_path or Path("./data/world_model.json")
        self._load_model()
    
    async def initialize(self):
        """Initialize the world model (load persisted state)"""
        self._load_model()
        return self
    
    def add_observation(
        self,
        observation: str,
        entities: List[Dict[str, Any]],
        relations: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Add observation to world model.
        
        Args:
            observation: Description of observation
            entities: List of observed entities
            relations: List of observed relations
        """
        # Add entities
        entity_ids = {}
        for ent in entities:
            eid = self.tracker.add_entity(
                entity_type=EntityType(ent['type']),
                name=ent['name'],
                properties=ent.get('properties', {})
            )
            entity_ids[ent['name']] = eid
        
        # Add relations
        if relations:
            for rel in relations:
                self.tracker.add_relation(
                    relation_type=RelationType(rel['type']),
                    source_id=entity_ids.get(rel['source'], rel['source']),
                    target_id=entity_ids.get(rel['target'], rel['target']),
                    properties=rel.get('properties', {})
                )
        
        # Snapshot state
        self.tracker.snapshot_state()
        self._save_model()
        
        logger.info(f"Added observation: {observation}")
    
    def query_state(
        self,
        entity_name: Optional[str] = None,
        entity_type: Optional[EntityType] = None
    ) -> List[Entity]:
        """Query current world state."""
        return self.tracker.find_entities(
            entity_type=entity_type,
            name_pattern=entity_name
        )
    
    async def predict_future(
        self,
        time_delta: timedelta = timedelta(minutes=5)
    ) -> WorldState:
        """Predict future state."""
        return await self.predictor.predict_next_state(time_delta)
    
    def simulate_action(
        self,
        action: Dict[str, Any]
    ) -> WorldState:
        """
        Simulate the effect of an action.
        
        Returns predicted state after action.
        """
        # Create temporary copy of state
        temp_entities = {eid: Entity(**e.to_dict()) for eid, e in self.tracker.entities.items()}
        
        # Apply action effects (simplified)
        action_type = action.get('type')
        
        if action_type == 'move':
            # Move entity
            entity_id = action.get('entity_id')
            new_location = action.get('location')
            if entity_id in temp_entities:
                temp_entities[entity_id].update_property('location', new_location)
        
        elif action_type == 'modify':
            # Modify entity property
            entity_id = action.get('entity_id')
            property_name = action.get('property')
            new_value = action.get('value')
            if entity_id in temp_entities:
                temp_entities[entity_id].update_property(property_name, new_value)
        
        # Return simulated state
        return WorldState(
            state_id=f"simulated_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            entities=temp_entities,
            relations=self.tracker.relations.copy(),
            global_properties={'simulated': True, 'action': action}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get world model statistics."""
        return {
            'total_entities': len(self.tracker.entities),
            'entities_by_type': {
                et.value: sum(1 for e in self.tracker.entities.values() if e.entity_type == et)
                for et in EntityType
            },
            'total_relations': len(self.tracker.relations),
            'relations_by_type': {
                rt.value: sum(1 for r in self.tracker.relations.values() if r.relation_type == rt)
                for rt in RelationType
            },
            'state_snapshots': len(self.tracker.state_history),
            'causal_rules': len(self.reasoner.causal_rules),
            'predictions_made': len(self.predictor.prediction_history)
        }
    
    def _save_model(self):
        """Save world model to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'entities': {eid: e.to_dict() for eid, e in self.tracker.entities.items()},
                'relations': {rid: r.to_dict() for rid, r in self.tracker.relations.items()},
                'causal_rules': self.reasoner.causal_rules,
                'global_properties': self.tracker.global_properties
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save world model: {e}")
    
    def _load_model(self):
        """Load world model from disk."""
        try:
            if not self.storage_path.exists():
                return
            
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            # Load entities
            for eid, ent_data in data.get('entities', {}).items():
                entity = Entity(
                    entity_id=ent_data['entity_id'],
                    entity_type=EntityType(ent_data['entity_type']),
                    name=ent_data['name'],
                    properties=ent_data['properties'],
                    created_at=datetime.fromisoformat(ent_data['created_at']),
                    last_updated=datetime.fromisoformat(ent_data['last_updated']),
                    confidence=ent_data['confidence']
                )
                self.tracker.entities[eid] = entity
            
            # Load relations
            for rid, rel_data in data.get('relations', {}).items():
                relation = Relation(
                    relation_id=rel_data['relation_id'],
                    relation_type=RelationType(rel_data['relation_type']),
                    source_id=rel_data['source_id'],
                    target_id=rel_data['target_id'],
                    properties=rel_data['properties'],
                    timestamp=datetime.fromisoformat(rel_data['timestamp']),
                    confidence=rel_data['confidence']
                )
                self.tracker.relations[rid] = relation
            
            # Load causal rules
            self.reasoner.causal_rules = data.get('causal_rules', [])
            
            # Load global properties
            self.tracker.global_properties = data.get('global_properties', {})
            
            logger.info(f"Loaded world model: {len(self.tracker.entities)} entities, {len(self.tracker.relations)} relations")
        
        except Exception as e:
            logger.error(f"Failed to load world model: {e}")


# Example usage
async def main():
    """Example world model usage."""
    
    print("=" * 50)
    print("World Model Example")
    print("=" * 50)
    
    # Initialize world model
    world = WorldModel()
    
    # Add observations
    print("\n1. Adding observations...")
    world.add_observation(
        observation="User is working on Python project",
        entities=[
            {'type': 'agent', 'name': 'User', 'properties': {'activity': 'coding'}},
            {'type': 'object', 'name': 'Python Project', 'properties': {'status': 'in_progress'}}
        ],
        relations=[
            {'type': 'has', 'source': 'User', 'target': 'Python Project'}
        ]
    )
    
    world.add_observation(
        observation="Task deadline approaching",
        entities=[
            {'type': 'event', 'name': 'Project Deadline', 'properties': {'date': '2026-03-01'}}
        ],
        relations=[
            {'type': 'requires', 'source': 'Python Project', 'target': 'Project Deadline'}
        ]
    )
    
    print("  Added 2 observations")
    
    # Query state
    print("\n2. Querying current state...")
    agents = world.query_state(entity_type=EntityType.AGENT)
    print(f"  Agents: {len(agents)}")
    for agent in agents:
        print(f"    - {agent.name}: {agent.properties}")
    
    objects = world.query_state(entity_type=EntityType.OBJECT)
    print(f"  Objects: {len(objects)}")
    for obj in objects:
        print(f"    - {obj.name}: {obj.properties}")
    
    # Add causal rule
    print("\n3. Adding causal rule...")
    world.reasoner.add_causal_rule(
        condition={'activity': 'coding'},
        effect={'project_progress': 'increased'},
        confidence=0.9
    )
    print("  Added rule: coding → project progress")
    
    # Predict future
    print("\n4. Predicting future state...")
    future_state = await world.predict_future(timedelta(hours=1))
    print(f"  Predicted {len(future_state.entities)} entities in 1 hour")
    
    # Simulate action
    print("\n5. Simulating action...")
    # Get user entity ID
    user_entities = world.query_state(name_pattern="User")
    if user_entities:
        user_id = user_entities[0].entity_id
        simulated = world.simulate_action({
            'type': 'modify',
            'entity_id': user_id,
            'property': 'activity',
            'value': 'testing'
        })
        print(f"  Simulated state after action")
    
    # Get statistics
    print("\n6. World model statistics...")
    stats = world.get_stats()
    print(f"  Total entities: {stats['total_entities']}")
    print(f"  Entities by type: {stats['entities_by_type']}")
    print(f"  Total relations: {stats['total_relations']}")
    print(f"  Causal rules: {stats['causal_rules']}")
    print(f"  State snapshots: {stats['state_snapshots']}")
    
    print("\n✅ World model example completed!")


if __name__ == "__main__":
    asyncio.run(main())
