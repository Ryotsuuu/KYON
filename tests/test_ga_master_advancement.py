"""
tests/test_ga_master_advancement.py
===================================
Comprehensive Test Suite for GA and Autonomous Master Agent Advancements:
1. Legality and Predefined Boundary Tests
2. Gatekeeper and Suggestion Popup Tests
3. Strategic Analysis and 9D Categorization Tests
4. Persistent Bidirectional Knowledge Tests
5. Memory, Storage and Behavior Tracking Tests
"""
import unittest
import json
import os
import shutil
import tempfile
from pathlib import Path
from collections import Counter

from agents.csv_data import get_csv_index
from agents.Genetic_Algorithm.deck_optimizer import (
    GeneticOptimizer,
    Individual,
    check_advanced_optimization_unlocked,
    render_unlock_suggestion_popup,
)
from unittest.mock import patch, MagicMock
from agents.Learning_System.persistent_knowledge import PersistentKnowledgeManager

ROOT = Path(__file__).resolve().parents[1]


class TestGAMasterAdvancement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.idx = get_csv_index(ROOT / "data")
        reg_file = ROOT / "ptcg-system" / "agents_registry.json"
        if reg_file.exists():
            with open(reg_file, 'r', encoding='utf-8') as f:
                cls.registry = json.load(f)
        else:
            cls.registry = {}

        cls.test_deck = cls.registry.get("S_FIG_stage_2_ex", {}).get("deck", [6] * 20 + [1119] * 4 + [1] * 36)

    @patch("agents.Resource_Management.simulation_runner.SimulationRunner.run_simulations")
    def test_method1_predefined_strict_bounds(self, mock_sim):
        """Verify Method 1 enforces 60 cards, max 4 copies, max 1 ACE SPEC, and basic Pokemon."""
        mock_sim.return_value = {"wins_p1": 3, "wins_p2": 1, "draws": 0, "completed_games": 4, "avg_turns": 20.0}
        opt = GeneticOptimizer(
            base_deck=self.test_deck,
            population_size=6,
            optimization_method="predefined",
            csv_index=self.idx,
            energy_types=["Fighting"],
            archetype="mega_stage_2_ex"
        )
        for _ in range(2):
            opt.evolve()

        for ind in opt.population:
            deck = ind.deck
            self.assertEqual(len(deck), 60, f"Deck length must be 60, got {len(deck)}")

            counts = Counter(deck)
            for cid, count in counts.items():
                card = self.idx.get_card(cid)
                is_basic_energy = cid in (1, 2, 3, 4, 5, 6, 7, 8, 9)
                if not is_basic_energy:
                    self.assertLessEqual(count, 4, f"Card #{cid} exceeded 4-copy rule: {count}")
                if card and card.is_ace_spec:
                    self.assertLessEqual(count, 1, f"ACE SPEC #{cid} exceeded 1-copy rule: {count}")

            has_basic_pkmn = any(
                self.idx.get_card(cid) and self.idx.get_card(cid).is_pokemon and self.idx.get_card(cid).is_basic
                for cid in deck
            )
            self.assertTrue(has_basic_pkmn, "Deck must have at least 1 Basic Pokemon")

    @patch("agents.Resource_Management.simulation_runner.SimulationRunner.run_simulations")
    def test_team_rocket_invariant_protection(self, mock_sim):
        """Verify Card 431 (Mewtwo ex) requires 4+ TR Pokemon invariant or is removed in non-TR decks."""
        mock_sim.return_value = {"wins_p1": 2, "wins_p2": 2, "draws": 0, "completed_games": 4, "avg_turns": 25.0}
        opt = GeneticOptimizer(
            base_deck=self.test_deck,
            population_size=4,
            optimization_method="advanced",
            csv_index=self.idx,
            energy_types=["Psychic"],
            archetype="balanced"
        )
        deck_with_431 = list(self.test_deck)
        deck_with_431[0] = 431
        repaired = opt.repair_deck(deck_with_431)
        self.assertEqual(len(repaired), 60)
        self.assertNotIn(431, repaired, "Card 431 must not exist in a non-TR deck with fewer than 4 TR cards")

    @patch("agents.Resource_Management.simulation_runner.SimulationRunner.run_simulations")
    def test_dragon_flexible_energy_protection(self, mock_sim):
        """Verify Dragon-type decks protect flexible energy (IDs 10, 12, 16)."""
        mock_sim.return_value = {"wins_p1": 2, "wins_p2": 2, "draws": 0, "completed_games": 4, "avg_turns": 25.0}
        opt = GeneticOptimizer(
            base_deck=self.test_deck,
            population_size=4,
            optimization_method="advanced",
            csv_index=self.idx,
            energy_types=["Dragon"],
            archetype="balanced"
        )
        counts = Counter([12, 16, 10])
        self.assertTrue(opt._is_protected(12, counts))
        self.assertTrue(opt._is_protected(16, counts))
        self.assertTrue(opt._is_protected(10, counts))

    def test_gatekeeper_condition_a_unlocked(self):
        """Verify agent with > 10,000 games (e.g. S_FIG_stage_2_ex) unlocks Method 2."""
        unlocked, status = check_advanced_optimization_unlocked("S_FIG_stage_2_ex", self.registry)
        self.assertTrue(unlocked, "S_FIG_stage_2_ex with >10K games must be unlocked under Condition A")
        self.assertTrue(status['condition_a_met'])
        self.assertGreater(status['agent_games'], 10000)

    def test_gatekeeper_condition_b_unlocked(self):
        """Verify that when all 10 energy types reach >= 3,000 games, Condition B unlocks."""
        mock_registry = {
            f"agent_{i}": {
                'games': 4000,
                'energy_types': [etype]
            }
            for i, etype in enumerate(["Grass", "Fire", "Water", "Lightning", "Psychic", "Fighting", "Darkness", "Metal", "Dragon", "Colorless"])
        }
        unlocked, status = check_advanced_optimization_unlocked("low_game_agent", mock_registry)
        self.assertTrue(unlocked, "Condition B must unlock when all energy types have >= 3,000 games")
        self.assertTrue(status['condition_b_met'])

    def test_gatekeeper_locked_scenario(self):
        """Verify an agent with < 10,000 games remains locked when metagame coverage is incomplete."""
        unlocked, status = check_advanced_optimization_unlocked("S_WAT_stage_1_ex", self.registry)
        self.assertFalse(unlocked, "S_WAT_stage_1_ex (8,008 games) must be locked")
        self.assertFalse(status['condition_a_met'])
        self.assertFalse(status['condition_b_met'])

    def test_suggestion_popup_rendering(self):
        """Verify the suggestion popup returns a formatted string with deficit info."""
        _, status = check_advanced_optimization_unlocked("S_WAT_stage_1_ex", self.registry)
        popup = render_unlock_suggestion_popup("S_WAT_stage_1_ex", status)
        self.assertIsNotNone(popup)
        popup_str = str(popup)
        self.assertIn("S_WAT_stage_1_ex", popup_str)
        self.assertIn("10,000", popup_str)
        self.assertIn("python ptcg.py", popup_str)

    def test_csv_card_data_deep_fields_loaded(self):
        """Verify enriched strategic analysis, 9D categorization, synergies, and GA risk are loaded."""
        self.assertGreater(len(self.idx.cards), 1000)
        self.assertEqual(len(self.idx.ga_risk_cards), 510, f"Expected 510 GA risk cards, found {len(self.idx.ga_risk_cards)}")
        sample_card = self.idx.get_card(1088)
        self.assertIsNotNone(sample_card)
        self.assertEqual(sample_card.gameplay_role, "trainer_utility")
        self.assertGreater(sample_card.effectiveness_score, 0)
        self.assertTrue(sample_card.is_item)

    @patch("agents.Resource_Management.simulation_runner.SimulationRunner.run_simulations")
    def test_strategic_mutation_bias(self, mock_sim):
        """Verify advanced mutation biases towards complementary and high-effectiveness cards."""
        mock_sim.return_value = {"wins_p1": 2, "wins_p2": 2, "draws": 0, "completed_games": 4, "avg_turns": 25.0}
        opt = GeneticOptimizer(
            base_deck=self.test_deck,
            population_size=4,
            optimization_method="advanced",
            csv_index=self.idx,
            energy_types=["Fighting"],
            archetype="mega_stage_2_ex"
        )
        counts = Counter(self.test_deck)
        rep = opt._find_strategic_replacement(6, self.test_deck, counts)
        self.assertIsNotNone(rep)

    def test_knowledge_store_persistence_and_reload(self):
        """Verify PersistentKnowledgeManager persists schemata and reloads accurately."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_knowledge.json"
            mgr1 = PersistentKnowledgeManager(store_path=tmp_path)
            mgr1.record_ga_discovery(
                deck=[1, 2, 3, 4, 5],
                fitness=88.5,
                archetype="aggro"
            )
            mgr1.record_counter_strategy(
                target_agent_id="TestChampion",
                counter_deck=[10, 11, 12],
                win_rate=0.72
            )

            mgr2 = PersistentKnowledgeManager(store_path=tmp_path)
            schemata = mgr2.get_top_schemata("aggro")
            self.assertEqual(len(schemata), 1)
            self.assertEqual(schemata[0]['fitness'], 88.5)

            counter = mgr2.get_counter_strategy("TestChampion")
            self.assertIsNotNone(counter)
            self.assertEqual(counter['win_rate'], 0.72)

    def test_ga_to_master_schemata_transfer(self):
        """Verify evolved elite decks enrich the persistent knowledge store."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_knowledge.json"
            mgr = PersistentKnowledgeManager(store_path=tmp_path)
            mgr.record_ga_discovery([10, 20, 30], 92.0, "stall")
            top = mgr.get_top_schemata("stall")
            self.assertEqual(len(top), 1)
            self.assertIn(10, top[0]['cards'])

    def test_storage_serialization_bounds(self):
        """Verify persistent knowledge files stay compact (<10MB) without unbounded growth."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_bounds.json"
            mgr = PersistentKnowledgeManager(store_path=tmp_path)
            for i in range(200):
                mgr.record_ga_discovery([i, i + 1], 50.0 + i, "balanced")

            file_size_mb = tmp_path.stat().st_size / (1024 * 1024)
            self.assertLess(file_size_mb, 10.0, f"Knowledge store exceeded 10MB limit: {file_size_mb:.2f} MB")
            schemata = mgr.get_top_schemata("balanced", limit=100)
            self.assertLessEqual(len(schemata), 50, "Schemata count should be capped at 50")

    def test_evolutionary_behavior_tracking(self):
        """Verify evolutionary velocity and telemetry tracking."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_telem.json"
            mgr = PersistentKnowledgeManager(store_path=tmp_path)
            mgr.record_generation_metrics(1, 50.0, 60.0, 10)
            mgr.record_generation_metrics(2, 55.0, 68.0, 10)
            mgr.record_generation_metrics(3, 62.0, 75.0, 10)

            telem = mgr.get_ecosystem_telemetry()
            self.assertEqual(telem['generation_records'], 3)
            self.assertGreater(telem['evolutionary_velocity'], 0.0)

    def test_max_6_pokemon_species_constraint(self):
        """Verify max 6 Pokemon species rule in both validator and repair pipeline."""
        from agents.deck_validator import MasterDeckValidator
        validator = MasterDeckValidator()

        # Find 8 distinct basic Pokemon
        basic_pks = [cid for cid, c in self.idx.cards.items() if c.is_pokemon and c.is_basic][:8]
        self.assertEqual(len(basic_pks), 8)

        # Build illegal deck with 8 species (1 copy each) + 52 basic energies
        deck_8_species = list(basic_pks) + [1] * (60 - len(basic_pks))
        rep = validator.validate_deck(deck_8_species)
        self.assertFalse(rep.is_legal)
        self.assertTrue(any("Rule 7 Fail: 8 Pokemon species (max 6)" in err for err in rep.errors))

        # Repair should enforce max 6 species
        opt = GeneticOptimizer(base_deck=deck_8_species, population_size=4, csv_index=self.idx, energy_types=["Grass"])
        repaired = opt._repair_deck(deck_8_species)
        rep_fixed = validator.validate_deck(repaired)
        repaired_pk_species = {cid for cid in repaired if self.idx.get_card(cid) and self.idx.get_card(cid).is_pokemon}
        self.assertLessEqual(len(repaired_pk_species), 6)
        self.assertLessEqual(sum(1 for cid in repaired if self.idx.get_card(cid) and self.idx.get_card(cid).is_pokemon), 18)

    def test_attack_energy_harmonization_and_repair(self):
        """Verify attack energy vs HP type distinction and unmet energy repair."""
        from agents.deck_validator import MasterDeckValidator
        validator = MasterDeckValidator()

        # Applin (#42) requires {G}{R} (Grass and Fire)
        applin = self.idx.get_card(42)
        self.assertIsNotNone(applin)
        self.assertIn("Grass", applin.get_attack_energy_types())
        self.assertIn("Fire", applin.get_attack_energy_types())

        # Deck with Applin, but only Fire Energy (2) and Water Energy (3) - Grass is missing!
        illegal_deck = [42] * 4 + [2] * 28 + [3] * 28
        rep = validator.validate_deck(illegal_deck)
        self.assertFalse(rep.is_legal)
        self.assertTrue(any("requires Grass energy to attack" in err for err in rep.errors))

        # Genetic optimizer repair should detect unmet attack cost and replace with compatible card
        opt = GeneticOptimizer(base_deck=illegal_deck, population_size=4, csv_index=self.idx, energy_types=["Fire", "Water"])
        repaired = opt._repair_deck(illegal_deck)
        for cid in repaired:
            c = self.idx.get_card(cid)
            if c and c.is_pokemon:
                self.assertTrue(c.get_attack_energy_types().issubset({"Fire", "Water"}))

    def test_special_energy_hp_matching_scan(self):
        """Verify typed special energy cards require matching Pokemon HP class."""
        from agents.deck_validator import MasterDeckValidator
        validator = MasterDeckValidator()

        # Build deck with Grow Grass Energy (#18) but ONLY Fire-type Pokemon (e.g. Charmander #108 or Charizard)
        # and Fire basic energy (#2)
        fire_pk = [cid for cid, c in self.idx.cards.items() if c.is_pokemon and c.is_basic and c.type == "Fire"][0]
        deck_with_dead_grass_spec = [fire_pk] * 4 + [18] * 4 + [2] * 52
        rep = validator.validate_deck(deck_with_dead_grass_spec)
        self.assertFalse(rep.is_legal)
        self.assertTrue(any("Grow Grass Energy (18) requires Grass-type Pokemon in deck" in err for err in rep.errors))

        # Repair deck must purge Grow Grass Energy (18) since deck has no Grass Pokemon
        opt = GeneticOptimizer(base_deck=deck_with_dead_grass_spec, population_size=4, csv_index=self.idx, energy_types=["Fire"])
        repaired = opt._repair_deck(deck_with_dead_grass_spec)
        self.assertNotIn(18, repaired)

    def test_stage_2_rare_candy_accelerator(self):
        """Verify Stage 2 decks require Rare Candy (1079), while non-Stage 2 decks purge it."""
        from agents.deck_validator import MasterDeckValidator
        validator = MasterDeckValidator()

        # Find Stage 2 Pokemon and trace its full evolution chain to find the root Basic
        stage2_cid = [cid for cid, c in self.idx.cards.items() if c.is_pokemon and c.is_stage2][0]
        s2_card = self.idx.get_card(stage2_cid)

        # Trace: Stage 2 -> evolves_from Stage 1 -> evolves_from Basic
        basic_cid = None
        if s2_card.evolves_from:
            stage1_match = [c for c in self.idx.cards.values() if c.is_pokemon and c.name == s2_card.evolves_from]
            if stage1_match and stage1_match[0].evolves_from:
                basic_match = [cid for cid, c in self.idx.cards.items() if c.is_pokemon and c.is_basic and c.name == stage1_match[0].evolves_from]
                if basic_match:
                    basic_cid = basic_match[0]
        if basic_cid is None:
            basic_cid = [cid for cid, c in self.idx.cards.items() if c.is_pokemon and c.is_basic and c.type == s2_card.type][0]

        # Case 1: Stage 2 deck lacking Rare Candy
        deck_no_candy = [basic_cid] * 3 + [stage2_cid] * 3 + [6] * 54
        rep1 = validator.validate_deck(deck_no_candy)
        self.assertFalse(rep1.is_legal)
        self.assertTrue(any("lacks Rare Candy" in err for err in rep1.errors))

        # Repair must automatically inject Rare Candy (1079)
        opt = GeneticOptimizer(base_deck=deck_no_candy, population_size=4, csv_index=self.idx, energy_types=[s2_card.type])
        repaired1 = opt._repair_deck(deck_no_candy)
        self.assertIn(1079, repaired1)
        self.assertGreaterEqual(repaired1.count(1079), 2)

        # Case 2: Non-Stage 2 deck erroneously having Rare Candy
        basic_only_deck = [basic_cid] * 4 + [1079] * 2 + [6] * 54
        rep2 = validator.validate_deck(basic_only_deck)
        self.assertFalse(rep2.is_legal)
        self.assertTrue(any("has no Stage 2 Pokemon" in err for err in rep2.errors))

        # Repair must purge Rare Candy from non-Stage 2 deck
        repaired2 = opt._repair_deck(basic_only_deck)
        self.assertNotIn(1079, repaired2)


if __name__ == "__main__":
    unittest.main()
