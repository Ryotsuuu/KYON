"""
agents/Evaluation_System/champion_tracker.py
=============================================
Automated Champion Benchmark Tracker & Kaggle Hot-Swap Engine.

Monitors agent win rates across tournaments, matchups-simulations, and GA evolution.
When an agent sets a new all-time high meta win rate, automatically:
1. Backs up timestamped archive in ptcg-system/checkpoints/
2. Hot-swaps submission.tar.gz with the new champion
3. Updates persistent champion telemetry in ptcg-system/champion_benchmark.json
"""
import os
import sys
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SYSTEM_DIR = ROOT_DIR / "ptcg-system"
CHECKPOINTS_DIR = SYSTEM_DIR / "checkpoints"
BENCHMARK_FILE = SYSTEM_DIR / "champion_benchmark.json"


class ChampionTracker:
    """Tracks the highest performing meta champion and automates Kaggle export hot-swapping."""

    def __init__(self):
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        self.benchmark_file = BENCHMARK_FILE
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.benchmark_file.exists():
            try:
                with open(self.benchmark_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load champion benchmark: {e}")
        return {
            "current_champion": None,
            "champion_win_rate": 0.0,
            "champion_archetype": None,
            "games_tested": 0,
            "recorded_at": None,
            "history": []
        }

    def _save(self) -> None:
        try:
            with open(self.benchmark_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save champion benchmark: {e}")

    def check_and_promote(
        self,
        agent_id: str,
        win_rate: float,
        games_tested: int,
        context: str = "matchups-simulation",
        deck: Optional[list] = None
    ) -> bool:
        """
        Evaluates candidate win rate. If it surpasses current champion (min 60% WR, min 10 games),
        promotes candidate, exports submission archive, and hot-swaps submission.tar.gz.
        """
        if games_tested < 10 or win_rate < 0.60:
            return False

        prev_best = self.data.get("champion_win_rate", 0.0)
        if win_rate <= prev_best and self.data.get("current_champion") is not None:
            return False

        # New Champion Detected!
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        record = {
            "agent_id": agent_id,
            "win_rate": round(win_rate, 4),
            "games_tested": games_tested,
            "context": context,
            "promoted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "previous_champion": self.data.get("current_champion"),
            "previous_win_rate": prev_best
        }

        self.data["current_champion"] = agent_id
        self.data["champion_win_rate"] = round(win_rate, 4)
        self.data["games_tested"] = games_tested
        self.data["recorded_at"] = record["promoted_at"]
        self.data["history"].append(record)
        self._save()

        # Generate Champion Submission Archive
        try:
            from ptcg import export_agent_tarball, _get_agent_deck
            agent_deck = deck or _get_agent_deck(agent_id)
            if agent_deck and len(agent_deck) == 60:
                # 1. Save timestamped checkpoint archive
                checkpoint_tar = CHECKPOINTS_DIR / f"champion_{agent_id}_{timestamp_str}.tar.gz"
                export_agent_tarball(agent_id, checkpoint_tar)

                # 2. Hot-swap root submission.tar.gz
                root_submission = ROOT_DIR / "submission.tar.gz"
                export_agent_tarball(agent_id, root_submission)
                logger.info(f"🏆 NEW CHAMPION PROMOTED: {agent_id} ({win_rate*100:.1f}%) -> Hot-swapped submission.tar.gz")
                return True
        except Exception as e:
            logger.warning(f"Failed to auto-export champion archive: {e}")

        return True

    def get_status(self) -> Dict[str, Any]:
        return {
            "current_champion": self.data.get("current_champion"),
            "champion_win_rate": self.data.get("champion_win_rate", 0.0),
            "games_tested": self.data.get("games_tested", 0),
            "recorded_at": self.data.get("recorded_at"),
            "total_promotions": len(self.data.get("history", []))
        }


_champion_tracker: Optional[ChampionTracker] = None


def get_champion_tracker() -> ChampionTracker:
    global _champion_tracker
    if _champion_tracker is None:
        _champion_tracker = ChampionTracker()
    return _champion_tracker
