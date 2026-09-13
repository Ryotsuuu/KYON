"""
agents/Learning_System/persistent_knowledge.py
=============================================
Persistent Bidirectional Knowledge Store for GA and Master Agent Systems.

Maintains cross-session state of:
- Discovered high-fitness deck schemata and building blocks
- Empirical pairwise card synergies extracted from evolutionary search
- Threat-specific counter strategies
- Co-evolutionary ecosystem dynamics (evolutionary velocity, adaptation latency)
"""
import os
import json
import time
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

ROOT = Path(__file__).resolve().parents[2]
SYSTEM_DIR = ROOT / "ptcg-system"
KNOWLEDGE_FILE = SYSTEM_DIR / "ga_master_knowledge.json"


class PersistentKnowledgeManager:
    """Manages persistent bidirectional knowledge transfer between GA and Master Agents."""

    def __init__(self, file_path: Optional[Path] = None, store_path: Optional[Path] = None):
        target = store_path or file_path or KNOWLEDGE_FILE
        self.file_path = Path(target)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = {
            'version': '1.0',
            'last_updated': datetime.now().isoformat(),
            'total_evolved_cycles': 0,
            'discovered_schemata': [],
            'synergy_pairs': [],
            'counter_strategies': {},
            'ecosystem_metrics': {
                'generation_history': [],
                'evolutionary_velocity': 0.0,
                'niche_filling_speed': 0.0,
                'counter_latency_turns': 0.0
            }
        }
        self.load_knowledge()

    def load_knowledge(self) -> Dict[str, Any]:
        """Load persistent knowledge from disk if available."""
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        self.data.update(loaded)
            except Exception:
                pass
        return self.data

    def save_knowledge(self):
        """Save persistent knowledge atomically to disk."""
        self.data['last_updated'] = datetime.now().isoformat()
        try:
            tmp_file = self.file_path.with_suffix('.tmp')
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
            if tmp_file.exists():
                tmp_file.replace(self.file_path)
        except Exception:
            pass

    def record_ga_discovery(
        self,
        deck: List[int],
        fitness: float,
        archetype: str,
        win_rate: float = 0.50,
        key_cards: Optional[List[int]] = None
    ):
        """Record an elite evolved deck and its key building block schemata."""
        if not deck:
            return

        schema_entry = {
            'timestamp': datetime.now().isoformat(),
            'archetype': archetype,
            'fitness': round(float(fitness), 2),
            'win_rate': round(float(win_rate), 4),
            'deck_sample': deck[:15],
            'key_cards': key_cards or list(dict.fromkeys(deck))[:6],
            'cards': key_cards or list(dict.fromkeys(deck))[:6],
            'deck_hash': hash(tuple(sorted(deck)))
        }

        # Keep top 50 highest fitness schemata to bound storage (<5MB)
        schemata = self.data.setdefault('discovered_schemata', [])
        if not any(s.get('deck_hash') == schema_entry['deck_hash'] for s in schemata):
            schemata.append(schema_entry)
            schemata.sort(key=lambda x: x.get('fitness', 0.0), reverse=True)
            self.data['discovered_schemata'] = schemata[:50]

        self.data['total_evolved_cycles'] = self.data.get('total_evolved_cycles', 0) + 1
        self.save_knowledge()

    def record_counter_strategy(
        self,
        target_id_or_archetype: Optional[str] = None,
        counter_deck: Optional[List[int]] = None,
        win_rate: float = 0.50,
        notes: str = '',
        target_agent_id: Optional[str] = None
    ):
        """Record a successful counter-strategy against a specific threat."""
        target = target_agent_id or target_id_or_archetype or 'unknown'
        counters = self.data.setdefault('counter_strategies', {})
        counters[target] = {
            'timestamp': datetime.now().isoformat(),
            'target': target,
            'counter_deck': list(counter_deck or []),
            'win_rate': round(float(win_rate), 4),
            'notes': notes
        }
        self.save_knowledge()

    def get_counter_strategy(self, target_id_or_archetype: str) -> Optional[dict]:
        """Retrieve the best recorded counter-strategy for a target threat."""
        counters = self.data.get('counter_strategies', {})
        return counters.get(target_id_or_archetype)

    def get_top_schemata(self, archetype: Optional[str] = None, limit: int = 5) -> List[dict]:
        """Retrieve highest fitness schemata, optionally filtered by archetype."""
        schemata = self.data.get('discovered_schemata', [])
        if archetype:
            filtered = [s for s in schemata if s.get('archetype') == archetype]
            return filtered[:limit]
        return schemata[:limit]

    def record_generation_metrics(
        self,
        generation: int,
        best_fitness: float = 0.0,
        avg_fitness: float = 0.0,
        diversity_score: float = 0.0,
        max_fitness: Optional[float] = None,
        active_agents: Optional[int] = None,
    ):
        """Track evolutionary velocity and diversity dynamics across generations."""
        if max_fitness is not None:
            best_fitness = max_fitness
        metrics = self.data.setdefault('ecosystem_metrics', {})
        history = metrics.setdefault('generation_history', [])
        history.append({
            'generation': generation,
            'best_fitness': round(float(best_fitness), 2),
            'avg_fitness': round(float(avg_fitness), 2),
            'diversity': round(float(diversity_score), 4),
            'active_agents': active_agents or 0,
            'timestamp': datetime.now().isoformat()
        })

        # Keep last 100 entries to maintain bounded storage
        metrics['generation_history'] = history[-100:]

        # Calculate evolutionary velocity (average fitness gain per gen over last 5 gens)
        if len(history) >= 2:
            recent = history[-5:]
            delta = recent[-1]['avg_fitness'] - recent[0]['avg_fitness']
            gens = max(1, len(recent) - 1)
            metrics['evolutionary_velocity'] = round(delta / gens, 3)

        self.save_knowledge()

    def get_ecosystem_telemetry(self) -> Dict[str, Any]:
        """Return executive summary of persistent knowledge store."""
        metrics = self.data.get('ecosystem_metrics', {})
        history = metrics.get('generation_history', [])
        return {
            'total_evolved_cycles': self.data.get('total_evolved_cycles', 0),
            'discovered_schemata_count': len(self.data.get('discovered_schemata', [])),
            'total_schemata': len(self.data.get('discovered_schemata', [])),
            'recorded_counter_strategies': len(self.data.get('counter_strategies', {})),
            'total_counter_strategies': len(self.data.get('counter_strategies', {})),
            'generation_records': len(history),
            'evolutionary_velocity': metrics.get('evolutionary_velocity', 0.0),
            'store_size_bytes': self.file_path.stat().st_size if self.file_path.exists() else 0,
            'last_updated': self.data.get('last_updated', 'Never')
        }


# Singleton accessor
_KNOWLEDGE_MANAGER: Optional[PersistentKnowledgeManager] = None

def get_persistent_knowledge() -> PersistentKnowledgeManager:
    global _KNOWLEDGE_MANAGER
    if _KNOWLEDGE_MANAGER is None:
        _KNOWLEDGE_MANAGER = PersistentKnowledgeManager()
    return _KNOWLEDGE_MANAGER