import sys
import pygame
import numpy as np
from typing import Optional, Tuple, Any

from src.env.quoridor_env import QuoridorEnv, QuoridorState, decode_action
from src.model.network import QuoridorModel
from src.mcts.mcts import MCTS, MCTSConfig
from src.utils.checkpoint import CheckpointManager
from src.utils.config import load_config

# Constants
FPS = 30
WINDOW_SIZE = 700
MARGIN = 50

# Colors
COLORS = {
    "background": (40, 40, 40),
    "board": (139, 69, 19),
    "cell": (205, 133, 63),
    "groove": (100, 50, 10),
    "wall": (220, 220, 200),
    "player0": (200, 50, 50),
    "player1": (50, 100, 200),
    "text": (255, 255, 255),
    "hover_move": (100, 255, 100, 100),
    "hover_wall": (255, 255, 255, 150),
}


class GameUI:
    def __init__(self, env: QuoridorEnv):
        self.env = env
        self.board_size = env.board_size

        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption(
            f"Quoridor AI({self.board_size}x{self.board_size})")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)

        # Pixel Math
        self.playable_size = WINDOW_SIZE - 2 * MARGIN
        total_units = 4 * self.board_size + (self.board_size - 1)
        self.unit_size = self.playable_size / total_units
        self.cell_size = self.unit_size * 4
        self.groove_size = self.unit_size

    def _get_pixel_coords(self, row: int, col: int) -> tuple[float, float]:
        """Convert board (row, col) to top-left (x, y) pixel coordinates of the cell."""
        x = MARGIN + col * (self.cell_size + self.groove_size)
        y = MARGIN + row * (self.cell_size + self.groove_size)
        return x, y

    def _get_action_hitboxes(self, state: QuoridorState) -> dict:
        valid_actions = self.env.get_valid_actions(state)
        hitboxes = {}
        current_pos = state.p0_pos if state.current_player == 0 else state.p1_pos

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

    def draw(self, state: QuoridorState, hitboxes: Optional[dict] = None):
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

        # Draw placed walls
        all_h_walls = state.p0_h_walls | state.p1_h_walls
        all_v_walls = state.p0_v_walls | state.p1_v_walls

        for r, c in all_h_walls:
            x, y = self._get_pixel_coords(r, c)
            w_width = 2 * self.cell_size + self.groove_size
            w_height = self.groove_size
            wall_rect = pygame.Rect(x, y + self.cell_size, w_width, w_height)
            pygame.draw.rect(self.screen, COLORS["wall"], wall_rect)

        for r, c in all_v_walls:
            x, y = self._get_pixel_coords(r, c)
            w_width = self.groove_size
            w_height = 2 * self.cell_size + self.groove_size
            wall_rect = pygame.Rect(x + self.cell_size, y, w_width, w_height)
            pygame.draw.rect(self.screen, COLORS["wall"], wall_rect)

        # Hover Highlights (Human turn only)
        if hitboxes and not state.game_over and state.current_player == 0:
            mouse_pos = pygame.mouse.get_pos()
            for action, rect in hitboxes.items():
                if rect.collidepoint(mouse_pos):
                    highlight = pygame.Surface(
                        (rect.width, rect.height), pygame.SRCALPHA
                    )
                    color = (
                        COLORS["hover_move"] if action < 12 else COLORS["hover_wall"]
                    )
                    highlight.fill(color)
                    self.screen.blit(highlight, rect.topleft)

        # Draw pawns
        for _, pos, color in [
            (0, state.p0_pos, COLORS["player0"]),
            (1, state.p1_pos, COLORS["player1"]),
        ]:
            r, c = pos
            x, y = self._get_pixel_coords(r, c)
            center_x = x + self.cell_size / 2
            center_y = y + self.cell_size / 2
            pygame.draw.circle(
                self.screen, color, (center_x, center_y), self.cell_size * 0.35
            )

        # Draw UI text (Walls remaining, Current Turn)
        p0_text = self.font.render(
            f"P0 Walls: {state.p0_walls}", True, COLORS["player0"]
        )
        p1_text = self.font.render(
            f"P1 Walls: {state.p1_walls}", True, COLORS["player1"]
        )
        turn_text = self.font.render(
            f"Turn: {'P0' if state.current_player == 0 else 'P1'}",
            True,
            COLORS["text"],
        )

        self.screen.blit(p0_text, (MARGIN, 10))
        self.screen.blit(
            p1_text, (WINDOW_SIZE - MARGIN - p1_text.get_width(), 10))
        self.screen.blit(turn_text, (WINDOW_SIZE // 2 -
                         turn_text.get_width() // 2, 10))

        if state.game_over:
            win_text = (
                "DRAW!" if state.winner is None else f"PLAYER {state.winner} WINS!"
            )
            color = (
                COLORS["text"]
                if state.winner is None
                else (COLORS["player0"] if state.winner == 0 else COLORS["player1"])
            )
            go_surf = self.font.render(win_text, True, color)
            self.screen.blit(
                go_surf, (WINDOW_SIZE // 2 - go_surf.get_width() //
                          2, WINDOW_SIZE - 40)
            )

        pygame.display.flip()

    def play_vs_ai(self, mcts: MCTS):
        """Play Human (P0) vs AI (P1)."""
        state = self.env.reset()
        running = True

        while running:
            hitboxes = (
                self._get_action_hitboxes(state)
                if not state.game_over and state.current_player == 0
                else {}
            )

            # Keep PyGame responsive during AI's turn
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # Human Turn (P0)
                if state.current_player == 0 and not state.game_over:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        for action, rect in hitboxes.items():
                            if rect.collidepoint(event.pos):
                                state, _, _, _ = self.env.step(state, action)
                                break

            # Draw the current state before AI thinks
            self.draw(state, hitboxes)

            # AI Turn (P1)
            if state.current_player == 1 and not state.game_over:
                # Provide visual feedback while MCTS blocks the thread
                thinking_text = self.font.render(
                    "AI is thinking...", True, COLORS["player1"]
                )
                self.screen.blit(thinking_text, (MARGIN, WINDOW_SIZE - 40))
                pygame.display.flip()

                # Temperature 0.0 means AI plays greedily (best move)
                action_probs = mcts.search(self.env, state, temperature=0.0)
                best_action = int(np.argmax(action_probs))
                state, _, _, _ = self.env.step(state, best_action)

            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


def load_ai_and_run(config_path: str = "configs/config_5x5.json"):
    print("Loading config...")
    cfg = load_config(config_path)

    print("Loading environment...")
    env = QuoridorEnv(is_poc=cfg.is_poc)

    net_cfg = cfg.network_config()
    print("Loading model and checkpoints...")
    model = QuoridorModel(
        board_size=cfg.board_size,
        action_space_size=env.action_space_size,
        num_channels=net_cfg.get("num_channels", 64),
        num_res_blocks=net_cfg.get("num_res_blocks", 4),
    )

    ckpt = CheckpointManager(base_dir="checkpoints")
    latest_state = ckpt.load_latest()

    if latest_state:
        model.load(latest_state["model_path"])
        print(f"Loaded AI from iteration {latest_state['iteration']}!")
    else:
        print("WARNING: No checkpoints found! AI will play completely randomly.")

    def nn_evaluate(state):
        tensor = env.state_to_tensor(state)
        return model.predict(tensor)

    mcts_cfg = cfg.mcts_config()
    mcts = MCTS(config=mcts_cfg, evaluate_fn=nn_evaluate)

    print("Starting GUI...")
    gui = GameUI(env)
    gui.play_vs_ai(mcts)


if __name__ == "__main__":
    load_ai_and_run()
