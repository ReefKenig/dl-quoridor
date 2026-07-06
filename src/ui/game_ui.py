import argparse
import sys
from typing import Optional

import numpy as np
import pygame

from src.env.quoridor_env_mp import QuoridorEnvMP, QuoridorStateMP, ACTION_TO_MOVE
from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig
from src.model.network_mp import QuoridorModelMP
from src.utils.config import DIFFICULTY_SETTINGS, DEFAULT_DIFFICULTY

# Constants
FPS = 30
WINDOW_SIZE = 700
MARGIN = 50

# Colors — up to 4 players
PLAYER_COLORS = [
    (200, 50, 50),    # P0: red (human)
    (50, 100, 200),   # P1: blue
    (50, 180, 50),    # P2: green
    (200, 160, 40),   # P3: gold
]

COLORS = {
    "background": (40, 40, 40),
    "board": (139, 69, 19),
    "cell": (205, 133, 63),
    "groove": (100, 50, 10),
    "wall": (220, 220, 200),
    "text": (255, 255, 255),
    "hover_move": (100, 255, 100, 100),
    "hover_wall": (255, 255, 255, 150),
}


def decode_action(action: int, board_size: int):
    W = board_size - 1
    v_offset = 12 + W ** 2
    if action < 12:
        return "pawn", ACTION_TO_MOVE[action]
    elif action < v_offset:
        w = action - 12
        return "h_wall", (w // W, w % W)
    else:
        w = action - v_offset
        return "v_wall", (w // W, w % W)


class GameUI:
    def __init__(self, env: QuoridorEnvMP):
        self.env = env
        self.board_size = env.board_size
        self.num_players = env.num_players

        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption(
            f"Quoridor {self.num_players}P ({self.board_size}x{self.board_size})")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 28)
        self.small_font = pygame.font.SysFont(None, 22)

        # Pixel Math
        self.playable_size = WINDOW_SIZE - 2 * MARGIN
        total_units = 4 * self.board_size + (self.board_size - 1)
        self.unit_size = self.playable_size / total_units
        self.cell_size = self.unit_size * 4
        self.groove_size = self.unit_size

    def _get_pixel_coords(self, row: int, col: int) -> tuple[float, float]:
        x = MARGIN + col * (self.cell_size + self.groove_size)
        y = MARGIN + row * (self.cell_size + self.groove_size)
        return x, y

    def _get_action_hitboxes(self, state: QuoridorStateMP) -> dict:
        valid_actions = self.env.get_valid_actions(state)
        hitboxes = {}
        current_pos = state.positions[state.current_player]

        for action in valid_actions:
            action_type, data = decode_action(action, self.board_size)

            if action_type == "pawn":
                dr, dc = data
                nr, nc = current_pos[0] + dr, current_pos[1] + dc
                x, y = self._get_pixel_coords(nr, nc)
                hitboxes[action] = pygame.Rect(
                    x, y, self.cell_size, self.cell_size)

            elif action_type == "h_wall":
                r, c = data
                x, y = self._get_pixel_coords(r, c)
                w_width = 2 * self.cell_size + self.groove_size
                w_height = self.groove_size
                hitboxes[action] = pygame.Rect(
                    x, y + self.cell_size, w_width, w_height)

            elif action_type == "v_wall":
                r, c = data
                x, y = self._get_pixel_coords(r, c)
                w_width = self.groove_size
                w_height = 2 * self.cell_size + self.groove_size
                hitboxes[action] = pygame.Rect(
                    x + self.cell_size, y, w_width, w_height)

        return hitboxes

    def draw(
        self,
        state: QuoridorStateMP,
        hitboxes: Optional[dict] = None,
        selected_hover_action: Optional[int] = None,
    ):
        self.screen.fill(COLORS["background"])

        # Draw base board background
        board_rect = pygame.Rect(
            MARGIN, MARGIN, self.playable_size, self.playable_size)
        pygame.draw.rect(self.screen, COLORS["groove"], board_rect)
        pygame.draw.rect(self.screen, COLORS["board"], board_rect, 5)

        # Draw cells
        for r in range(self.board_size):
            for c in range(self.board_size):
                x, y = self._get_pixel_coords(r, c)
                cell_rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, COLORS["cell"], cell_rect)

        # Draw placed walls (shared)
        for r, c in state.h_walls:
            x, y = self._get_pixel_coords(r, c)
            w_width = 2 * self.cell_size + self.groove_size
            w_height = self.groove_size
            wall_rect = pygame.Rect(x, y + self.cell_size, w_width, w_height)
            pygame.draw.rect(self.screen, COLORS["wall"], wall_rect)

        for r, c in state.v_walls:
            x, y = self._get_pixel_coords(r, c)
            w_width = self.groove_size
            w_height = 2 * self.cell_size + self.groove_size
            wall_rect = pygame.Rect(x + self.cell_size, y, w_width, w_height)
            pygame.draw.rect(self.screen, COLORS["wall"], wall_rect)

        # Hover highlight
        if selected_hover_action is not None and hitboxes and selected_hover_action in hitboxes:
            rect = hitboxes[selected_hover_action]
            highlight = pygame.Surface(
                (rect.width, rect.height), pygame.SRCALPHA)
            action_type, _ = decode_action(
                selected_hover_action, self.board_size)
            color = (
                COLORS["hover_move"] if action_type == "pawn" else COLORS["hover_wall"]
            )
            highlight.fill(color)
            self.screen.blit(highlight, rect.topleft)

        # Draw pawns
        for i in range(self.num_players):
            r, c = state.positions[i]
            x, y = self._get_pixel_coords(r, c)
            center_x = x + self.cell_size / 2
            center_y = y + self.cell_size / 2
            pygame.draw.circle(
                self.screen, PLAYER_COLORS[i],
                (center_x, center_y), self.cell_size * 0.35
            )
            # Draw player number on pawn
            label = self.small_font.render(str(i + 1), True, (255, 255, 255))
            self.screen.blit(label, (center_x - label.get_width() // 2,
                                     center_y - label.get_height() // 2))

        # Draw UI text — walls remaining for each player
        y_offset = 5
        for i in range(self.num_players):
            txt = self.small_font.render(
                f"P{i + 1}: {state.walls_remaining[i]}w", True, PLAYER_COLORS[i])
            self.screen.blit(txt, (MARGIN + i * 150, y_offset))

        # Current turn indicator
        cp = state.current_player
        turn_text = self.font.render(
            f"Turn: P{cp + 1}" + (" (YOU)" if cp == 0 else " (AI)"),
            True, PLAYER_COLORS[cp])
        self.screen.blit(turn_text,
                         (WINDOW_SIZE // 2 - turn_text.get_width() // 2, WINDOW_SIZE - 40))

        if state.game_over:
            if state.winner is None:
                win_text = "DRAW!"
                color = COLORS["text"]
            else:
                win_text = f"PLAYER {state.winner + 1} WINS!" + (
                    " (YOU!)" if state.winner == 0 else "")
                color = PLAYER_COLORS[state.winner]
            go_surf = self.font.render(win_text, True, color)
            self.screen.blit(
                go_surf, (WINDOW_SIZE // 2 - go_surf.get_width() // 2, WINDOW_SIZE - 70))

        pygame.display.flip()

    def play_vs_ai(self, mcts: MCTSMaxN, temperature: float = 0.3):
        """Play Human (P0) vs AI (all others)."""
        state = self.env.reset()
        running = True

        while running:
            is_human_turn = not state.game_over and state.current_player == 0
            hitboxes = self._get_action_hitboxes(
                state) if is_human_turn else {}

            # Hover detection
            selected_hover_action = None
            if is_human_turn:
                mouse_pos = pygame.mouse.get_pos()
                min_dist_sq = float("inf")
                for action, rect in hitboxes.items():
                    if rect.collidepoint(mouse_pos):
                        dx = mouse_pos[0] - rect.centerx
                        dy = mouse_pos[1] - rect.centery
                        dist_sq = dx * dx + dy * dy
                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                            selected_hover_action = action

            # Event Loop
            action_taken = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if is_human_turn and not action_taken:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if (selected_hover_action is not None
                                and selected_hover_action in hitboxes):
                            state, _, _, _ = self.env.step(
                                state, selected_hover_action)
                            action_taken = True

            self.draw(state, hitboxes, selected_hover_action)

            # AI turns (any player != 0)
            if not state.game_over and state.current_player != 0:
                cp = state.current_player
                thinking_text = self.font.render(
                    f"P{cp + 1} (AI) thinking...", True, PLAYER_COLORS[cp])
                self.screen.blit(thinking_text, (MARGIN, WINDOW_SIZE - 70))
                pygame.display.flip()

                action_probs = mcts.search(self.env, state, temperature=temperature)
                if temperature < 0.1:
                    best_action = int(np.argmax(action_probs))
                else:
                    best_action = int(np.random.choice(len(action_probs), p=action_probs))
                state, _, _, _ = self.env.step(state, best_action)

            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


def load_ai_and_run(num_players: int = 4, num_simulations: int = 100,
                    temperature: float = 0.3, c_puct: float = 1.41,
                    dirichlet_epsilon: float = 0.25):
    board_size = 5
    print(
        f"Setting up {num_players}-player Quoridor on {board_size}x{board_size} board...")

    env = QuoridorEnvMP(board_size=board_size, num_players=num_players)
    in_channels = 3 * num_players + 3

    print("Loading model...")
    model = QuoridorModelMP(
        board_size=board_size,
        action_space_size=env.action_space_size,
        num_channels=64,
        num_res_blocks=4,
        in_channels=in_channels,
        num_players=num_players,
    )

    # Try to load the best checkpoint for this player count
    import os
    ckpt_dir = f"checkpoints_mp_n{num_players}"
    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(best_path):
        model.load(best_path)
        print(f"Loaded model from {best_path}")
    else:
        print(
            f"WARNING: No model found at {best_path}. AI will play randomly.")

    def nn_evaluate(state):
        tensor = env.state_to_tensor(state)
        return model.predict(tensor)

    mcts_cfg = MCTSConfig(num_simulations=num_simulations, c_puct=c_puct,
                          dirichlet_epsilon=dirichlet_epsilon)
    mcts = MCTSMaxN(config=mcts_cfg, evaluate_fn=nn_evaluate,
                    num_players=num_players)

    print("Starting GUI...")
    gui = GameUI(env)
    gui.play_vs_ai(mcts, temperature=temperature)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quoridor Human vs AI (2-4 players)")
    parser.add_argument(
        "--players", "-p",
        type=int,
        default=4,
        choices=[2, 4],
        help="Number of players (2 or 4)",
    )
    parser.add_argument(
        "--difficulty", "-d",
        type=str,
        default=DEFAULT_DIFFICULTY,
        choices=list(DIFFICULTY_SETTINGS.keys()),
        help="AI difficulty level",
    )
    args = parser.parse_args()

    settings = DIFFICULTY_SETTINGS[args.difficulty]

    load_ai_and_run(
        num_players=args.players,
        num_simulations=settings["num_simulations"],
        temperature=settings["temperature"],
        c_puct=settings["c_puct"],
        dirichlet_epsilon=settings["dirichlet_epsilon"],
    )
