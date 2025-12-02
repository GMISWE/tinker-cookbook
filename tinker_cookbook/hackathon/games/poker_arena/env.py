"""
Poker Arena: LLM Texas Hold'em Environment

A multi-agent poker environment where 6 LLM agents play No-Limit Texas Hold'em.
Agents must make strategic betting decisions with imperfect information.

Game Rules:
- Standard No-Limit Texas Hold'em rules
- 6 players, 100 big blinds starting stack
- Small blind 1, Big blind 2
- One hand = one episode

Rewards:
- Chips won/lost normalized by big blind
- Winning a hand gives positive reward
- Losing chips gives negative reward
"""

import asyncio
import json
import random
from dataclasses import dataclass, field
from typing import Sequence
from itertools import combinations

import chz
from tinker import types
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Action parsing stop condition
STOP_CONDITION = ["}\n", "}\n\n"]

# Card constants
SUITS = ["h", "d", "c", "s"]  # hearts, diamonds, clubs, spades
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
RANK_VALUES = {r: i for i, r in enumerate(RANKS)}

# Hand rankings
HAND_RANKS = [
    "High Card",
    "Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
    "Royal Flush",
]

# Betting rounds
BETTING_ROUNDS = ["preflop", "flop", "turn", "river"]


class Card:
    """Represents a playing card."""

    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit

    def __repr__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __eq__(self, other) -> bool:
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        return False

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))

    @property
    def value(self) -> int:
        return RANK_VALUES[self.rank]


class Deck:
    """A standard 52-card deck."""

    def __init__(self, rng: random.Random):
        self.cards = [Card(r, s) for r in RANKS for s in SUITS]
        rng.shuffle(self.cards)
        self.index = 0

    def deal(self, count: int = 1) -> list[Card]:
        dealt = self.cards[self.index:self.index + count]
        self.index += count
        return dealt


def evaluate_hand(cards: list[Card]) -> tuple[int, list[int]]:
    """
    Evaluate a 5-card poker hand.
    Returns (hand_rank, tiebreaker_values) where higher is better.
    """
    ranks = sorted([c.value for c in cards], reverse=True)
    suits = [c.suit for c in cards]

    is_flush = len(set(suits)) == 1
    is_straight = (
        len(set(ranks)) == 5 and ranks[0] - ranks[4] == 4
    ) or ranks == [12, 3, 2, 1, 0]  # Ace-low straight

    # Count rank occurrences
    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    counts = sorted(rank_counts.values(), reverse=True)

    # Determine hand rank
    if is_straight and is_flush:
        if ranks[0] == 12:  # Ace high
            return (9, ranks)  # Royal Flush
        return (8, ranks)  # Straight Flush
    elif counts == [4, 1]:
        return (7, ranks)  # Four of a Kind
    elif counts == [3, 2]:
        return (6, ranks)  # Full House
    elif is_flush:
        return (5, ranks)  # Flush
    elif is_straight:
        return (4, ranks)  # Straight
    elif counts == [3, 1, 1]:
        return (3, ranks)  # Three of a Kind
    elif counts == [2, 2, 1]:
        return (2, ranks)  # Two Pair
    elif counts == [2, 1, 1, 1]:
        return (1, ranks)  # Pair
    else:
        return (0, ranks)  # High Card


def best_hand(hole_cards: list[Card], community: list[Card]) -> tuple[int, list[int]]:
    """Find the best 5-card hand from 7 cards."""
    all_cards = hole_cards + community
    best = (0, [0, 0, 0, 0, 0])
    for combo in combinations(all_cards, 5):
        hand_value = evaluate_hand(list(combo))
        if hand_value > best:
            best = hand_value
    return best


class PokerGameState:
    """Manages the state of a single poker hand."""

    def __init__(
        self,
        num_players: int = 6,
        starting_stack: int = 200,
        small_blind: int = 1,
        big_blind: int = 2,
        rng: random.Random | None = None,
    ):
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.rng = rng or random.Random()

        # Initialize deck and deal
        self.deck = Deck(self.rng)
        self.hole_cards: dict[int, list[Card]] = {}
        for i in range(num_players):
            self.hole_cards[i] = self.deck.deal(2)

        self.community_cards: list[Card] = []

        # Player state
        self.stacks: dict[int, int] = {i: starting_stack for i in range(num_players)}
        self.folded: dict[int, bool] = {i: False for i in range(num_players)}
        self.all_in: dict[int, bool] = {i: False for i in range(num_players)}
        self.bets_this_round: dict[int, int] = {i: 0 for i in range(num_players)}

        # Pot and betting
        self.pot = 0
        self.current_bet = 0
        self.min_raise = big_blind
        self.last_raiser: int | None = None

        # Betting round
        self.betting_round = "preflop"
        self.dealer_position = 0
        self.current_player = (self.dealer_position + 3) % num_players  # UTG

        # Post blinds
        sb_pos = (self.dealer_position + 1) % num_players
        bb_pos = (self.dealer_position + 2) % num_players
        self._post_blind(sb_pos, small_blind)
        self._post_blind(bb_pos, big_blind)
        self.current_bet = big_blind
        self.last_raiser = bb_pos

        self.done = False
        self.winners: list[int] = []

    def _post_blind(self, player_id: int, amount: int) -> None:
        """Post a blind bet."""
        actual = min(amount, self.stacks[player_id])
        self.stacks[player_id] -= actual
        self.bets_this_round[player_id] = actual
        self.pot += actual
        if self.stacks[player_id] == 0:
            self.all_in[player_id] = True

    def get_active_players(self) -> list[int]:
        """Get players still in the hand."""
        return [i for i in range(self.num_players) if not self.folded[i]]

    def get_players_who_can_act(self) -> list[int]:
        """Get players who can still make betting decisions."""
        return [
            i for i in range(self.num_players)
            if not self.folded[i] and not self.all_in[i]
        ]

    def get_valid_actions(self, player_id: int) -> list[str]:
        """Get valid actions for a player."""
        if self.folded[player_id] or self.all_in[player_id]:
            return []

        actions = ["fold"]
        to_call = self.current_bet - self.bets_this_round[player_id]

        if to_call == 0:
            actions.append("check")
        else:
            actions.append("call")

        # Can raise if has enough chips
        if self.stacks[player_id] > to_call:
            actions.append("raise")

        return actions

    def get_min_raise_amount(self, player_id: int) -> int:
        """Get minimum raise amount."""
        to_call = self.current_bet - self.bets_this_round[player_id]
        return to_call + self.min_raise

    def get_max_raise_amount(self, player_id: int) -> int:
        """Get maximum raise amount (all-in)."""
        return self.stacks[player_id]

    def get_observation(self, player_id: int) -> dict:
        """Get observation for a player."""
        players_info = []
        for i in range(self.num_players):
            info = {
                "id": i,
                "stack": self.stacks[i],
                "bet": self.bets_this_round[i],
                "folded": self.folded[i],
                "all_in": self.all_in[i],
            }
            if i == player_id:
                info["is_you"] = True
            if i == self.current_player:
                info["to_act"] = True
            players_info.append(info)

        valid_actions = self.get_valid_actions(player_id)

        return {
            "your_id": player_id,
            "your_hole_cards": [str(c) for c in self.hole_cards[player_id]],
            "community_cards": [str(c) for c in self.community_cards],
            "pot": self.pot,
            "current_bet": self.current_bet,
            "your_stack": self.stacks[player_id],
            "your_bet_this_round": self.bets_this_round[player_id],
            "to_call": self.current_bet - self.bets_this_round[player_id],
            "players": players_info,
            "betting_round": self.betting_round,
            "dealer_position": self.dealer_position,
            "min_raise": self.get_min_raise_amount(player_id) if "raise" in valid_actions else 0,
            "max_raise": self.get_max_raise_amount(player_id) if "raise" in valid_actions else 0,
            "valid_actions": valid_actions,
            "active_players": len(self.get_active_players()),
        }

    def apply_action(self, player_id: int, action: str, amount: int = 0) -> None:
        """Apply a player's action."""
        if player_id != self.current_player:
            raise ValueError(f"Not player {player_id}'s turn")

        valid_actions = self.get_valid_actions(player_id)
        if action not in valid_actions:
            # Invalid action - auto-fold
            action = "fold"

        if action == "fold":
            self.folded[player_id] = True
        elif action == "check":
            pass  # No action needed
        elif action == "call":
            to_call = self.current_bet - self.bets_this_round[player_id]
            actual = min(to_call, self.stacks[player_id])
            self.stacks[player_id] -= actual
            self.bets_this_round[player_id] += actual
            self.pot += actual
            if self.stacks[player_id] == 0:
                self.all_in[player_id] = True
        elif action == "raise":
            to_call = self.current_bet - self.bets_this_round[player_id]
            min_raise = self.get_min_raise_amount(player_id)
            max_raise = self.get_max_raise_amount(player_id)

            # Clamp raise amount
            raise_amount = max(min_raise, min(amount, max_raise))
            self.stacks[player_id] -= raise_amount
            self.bets_this_round[player_id] += raise_amount
            self.pot += raise_amount

            new_bet = self.bets_this_round[player_id]
            raise_size = new_bet - self.current_bet
            self.min_raise = max(self.min_raise, raise_size)
            self.current_bet = new_bet
            self.last_raiser = player_id

            if self.stacks[player_id] == 0:
                self.all_in[player_id] = True

        # Move to next player
        self._advance_action()

    def _advance_action(self) -> None:
        """Advance to the next player or betting round."""
        active = self.get_active_players()

        # Check if only one player left
        if len(active) == 1:
            self._end_hand()
            return

        # Check if betting round is complete
        can_act = self.get_players_who_can_act()
        if not can_act:
            # Everyone is all-in or folded
            self._deal_remaining_and_showdown()
            return

        # Find next player who can act
        for i in range(1, self.num_players + 1):
            next_player = (self.current_player + i) % self.num_players
            if next_player in can_act:
                # Check if this player has had a chance to act on current bet
                if next_player == self.last_raiser:
                    # Betting round complete
                    self._next_betting_round()
                    return
                # Check if player has matched the current bet (and had a chance to act)
                if (
                    self.bets_this_round[next_player] == self.current_bet
                    and self.last_raiser is not None
                    and next_player != self.last_raiser
                ):
                    # This player has matched the bet after the raise
                    # Check if we've gone around
                    continue
                self.current_player = next_player
                return

        # If we get here, betting round is complete
        self._next_betting_round()

    def _next_betting_round(self) -> None:
        """Move to the next betting round."""
        # Reset betting state
        self.bets_this_round = {i: 0 for i in range(self.num_players)}
        self.current_bet = 0
        self.min_raise = self.big_blind
        self.last_raiser = None

        round_idx = BETTING_ROUNDS.index(self.betting_round)
        if round_idx >= 3:  # After river
            self._showdown()
            return

        # Deal community cards
        if self.betting_round == "preflop":
            self.community_cards = self.deck.deal(3)  # Flop
            self.betting_round = "flop"
        elif self.betting_round == "flop":
            self.community_cards.extend(self.deck.deal(1))  # Turn
            self.betting_round = "turn"
        elif self.betting_round == "turn":
            self.community_cards.extend(self.deck.deal(1))  # River
            self.betting_round = "river"

        # Set first to act (after dealer)
        can_act = self.get_players_who_can_act()
        if not can_act:
            self._deal_remaining_and_showdown()
            return

        for i in range(1, self.num_players + 1):
            next_player = (self.dealer_position + i) % self.num_players
            if next_player in can_act:
                self.current_player = next_player
                self.last_raiser = next_player  # First to act can be "raiser"
                return

    def _deal_remaining_and_showdown(self) -> None:
        """Deal remaining community cards and go to showdown."""
        while len(self.community_cards) < 5:
            self.community_cards.extend(self.deck.deal(1))
        self._showdown()

    def _showdown(self) -> None:
        """Determine winner at showdown."""
        active = self.get_active_players()

        if len(active) == 1:
            self.winners = active
        else:
            # Evaluate hands
            hand_values = {}
            for player_id in active:
                hand_values[player_id] = best_hand(
                    self.hole_cards[player_id],
                    self.community_cards
                )

            # Find best hand
            best_value = max(hand_values.values())
            self.winners = [
                pid for pid, val in hand_values.items()
                if val == best_value
            ]

        self._end_hand()

    def _end_hand(self) -> None:
        """End the hand and distribute pot."""
        self.done = True
        if not self.winners:
            self.winners = self.get_active_players()

        # Split pot among winners
        win_amount = self.pot // len(self.winners)
        for winner in self.winners:
            self.stacks[winner] += win_amount

    def get_rewards(self) -> dict[int, float]:
        """Get rewards for all players (chips won/lost normalized by BB)."""
        rewards = {}
        for i in range(self.num_players):
            chip_diff = self.stacks[i] - self.starting_stack
            rewards[i] = chip_diff / self.big_blind
        return rewards


class PokerRoundCoordinator:
    """Coordinates sequential betting actions among players."""

    def __init__(self, game_state: PokerGameState):
        self.game_state = game_state
        self.condition = asyncio.Condition()

    @property
    def done(self) -> bool:
        return self.game_state.done

    @property
    def current_player(self) -> int:
        return self.game_state.current_player

    async def wait_for_turn(self, player_id: int) -> None:
        """Wait until it's this player's turn."""
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.current_player == player_id or self.done
            )

    async def take_action(self, player_id: int, action: str, amount: int = 0) -> None:
        """Take an action and notify other players."""
        async with self.condition:
            if not self.done and self.current_player == player_id:
                self.game_state.apply_action(player_id, action, amount)
                self.condition.notify_all()


@dataclass
class PokerEnv(Env):
    """Environment for a single poker player."""

    player_id: int
    coordinator: PokerRoundCoordinator
    renderer: Renderer

    @property
    def stop_condition(self) -> StopCondition:
        return STOP_CONDITION

    def _format_observation(self) -> str:
        """Format game state as a prompt for the LLM."""
        obs = self.coordinator.game_state.get_observation(self.player_id)

        prompt = f"""You are Player #{obs['your_id']} in a 6-player No-Limit Texas Hold'em game.

YOUR HAND: {' '.join(obs['your_hole_cards'])}
COMMUNITY CARDS: {' '.join(obs['community_cards']) if obs['community_cards'] else '(none yet)'}
BETTING ROUND: {obs['betting_round'].upper()}

POT: {obs['pot']} chips
CURRENT BET: {obs['current_bet']} chips
YOUR STACK: {obs['your_stack']} chips
YOUR BET THIS ROUND: {obs['your_bet_this_round']} chips
TO CALL: {obs['to_call']} chips

PLAYERS:
"""
        for p in obs['players']:
            status = []
            if p.get('folded'):
                status.append("FOLDED")
            if p.get('all_in'):
                status.append("ALL-IN")
            if p.get('is_you'):
                status.append("YOU")
            if p.get('to_act'):
                status.append("TO ACT")
            status_str = f" ({', '.join(status)})" if status else ""
            prompt += f"  Player #{p['id']}: Stack={p['stack']}, Bet={p['bet']}{status_str}\n"

        prompt += f"""
VALID ACTIONS: {obs['valid_actions']}
"""
        if "raise" in obs['valid_actions']:
            prompt += f"MIN RAISE: {obs['min_raise']} chips, MAX RAISE (all-in): {obs['max_raise']} chips\n"

        prompt += """
Respond with a JSON object containing your action:
{"action": "fold"}
{"action": "check"}
{"action": "call"}
{"action": "raise", "amount": 50}

Make your decision:
"""
        return prompt

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Get initial observation, waiting for turn if needed."""
        await self.coordinator.wait_for_turn(self.player_id)

        if self.coordinator.done:
            return types.ModelInput.empty(), self.stop_condition

        prompt = self._format_observation()
        return self.renderer.build_generation_prompt([{"role": "user", "content": prompt}]), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        """Process player's action."""
        if self.coordinator.done:
            return self._get_done_step()

        # Parse action
        action_type, amount = self._parse_action(action)

        # Take action
        await self.coordinator.take_action(self.player_id, action_type, amount)

        # Check if hand is over
        if self.coordinator.done:
            rewards = self.coordinator.game_state.get_rewards()
            return StepResult(
                reward=rewards[self.player_id],
                episode_done=True,
                next_observation=types.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics=self._get_metrics(),
            )

        # Wait for next turn
        await self.coordinator.wait_for_turn(self.player_id)

        if self.coordinator.done:
            rewards = self.coordinator.game_state.get_rewards()
            return StepResult(
                reward=rewards[self.player_id],
                episode_done=True,
                next_observation=types.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics=self._get_metrics(),
            )

        # Get next observation
        prompt = self._format_observation()
        next_obs = self.renderer.build_generation_prompt([{"role": "user", "content": prompt}])

        return StepResult(
            reward=0.0,  # Intermediate rewards are 0
            episode_done=False,
            next_observation=next_obs,
            next_stop_condition=self.stop_condition,
            metrics={},
        )

    def _parse_action(self, action: Action) -> tuple[str, int]:
        """Parse LLM output to extract action and amount."""
        try:
            message = self.renderer.parse_response(action)[0]
            text = message["content"].strip()

            # Find JSON in text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                data = json.loads(json_str)
                action_type = data.get("action", "fold").lower()
                amount = int(data.get("amount", 0))
                return action_type, amount
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        return "fold", 0

    def _get_done_step(self) -> StepResult:
        """Return done step result."""
        rewards = self.coordinator.game_state.get_rewards()
        return StepResult(
            reward=rewards.get(self.player_id, 0.0),
            episode_done=True,
            next_observation=types.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=self._get_metrics(),
        )

    def _get_metrics(self) -> dict[str, float]:
        """Get metrics for this player."""
        gs = self.coordinator.game_state
        return {
            "poker/stack": float(gs.stacks[self.player_id]),
            "poker/won": float(self.player_id in gs.winners),
            "poker/folded": float(gs.folded[self.player_id]),
            "poker/pot": float(gs.pot),
        }


@dataclass
class PokerEnvGroupBuilder(EnvGroupBuilder):
    """Builder for poker environments sharing one hand."""

    renderer: Renderer
    num_players: int = 6
    starting_stack: int = 200
    small_blind: int = 1
    big_blind: int = 2
    seed: int | None = None

    async def make_envs(self) -> Sequence[Env]:
        """Create all player environments for one hand."""
        rng = random.Random(self.seed) if self.seed else random.Random()
        game_state = PokerGameState(
            num_players=self.num_players,
            starting_stack=self.starting_stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            rng=rng,
        )
        coordinator = PokerRoundCoordinator(game_state)

        return [
            PokerEnv(
                player_id=player_id,
                coordinator=coordinator,
                renderer=self.renderer,
            )
            for player_id in range(self.num_players)
        ]


class PokerArenaDataset(RLDataset):
    """Dataset for Poker Arena hands."""

    def __init__(
        self,
        batch_size: int,
        builder: PokerEnvGroupBuilder,
        num_hands: int,
    ):
        self.batch_size = batch_size
        self.builder = builder
        self.num_hands = num_hands
        self.num_players = builder.num_players

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get a batch of hand builders."""
        hands_per_batch = self.batch_size // self.num_players
        return [self.builder for _ in range(hands_per_batch)]

    def __len__(self) -> int:
        return self.num_hands


@chz.chz
class PokerArenaDatasetBuilder(RLDatasetBuilder):
    """Dataset builder for Poker Arena with CLI configuration."""

    # Game settings
    num_players: int = 6
    starting_stack: int = 200
    small_blind: int = 1
    big_blind: int = 2

    # Training settings
    batch_size: int = 12  # Should be divisible by num_players
    num_train_hands: int = 1000
    num_test_hands: int = 100

    # Model settings
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    renderer_name: str = "qwen3"
    base_url: str | None = None

    async def __call__(self) -> tuple[PokerArenaDataset, PokerArenaDataset | None]:
        """Build training and test datasets."""
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))

        train_builder = PokerEnvGroupBuilder(
            renderer=renderer,
            num_players=self.num_players,
            starting_stack=self.starting_stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
        )

        train_dataset = PokerArenaDataset(
            batch_size=self.batch_size,
            builder=train_builder,
            num_hands=self.num_train_hands,
        )

        test_builder = PokerEnvGroupBuilder(
            renderer=renderer,
            num_players=self.num_players,
            starting_stack=self.starting_stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            seed=42,
        )

        test_dataset = PokerArenaDataset(
            batch_size=self.batch_size,
            builder=test_builder,
            num_hands=self.num_test_hands,
        )

        return train_dataset, test_dataset
