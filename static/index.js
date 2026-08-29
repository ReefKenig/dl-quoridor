// --- Global state (shared across modules) ---

const mainMenu = document.getElementById("main-menu");
const gameScreen = document.getElementById("game-screen");
const canvas = document.getElementById("game-board");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("game-status");
const restartBtn = document.getElementById("restart-btn");

// Game State Variables
let currentGridSize = parseInt(document.getElementById("board-size").value);
let numPlayers = parseInt(document.getElementById("num-players").value);
let currentDifficulty = document.getElementById("difficulty").value;
let cellSize = 0;
let isPlayerTurn = true;
let userSeat = 0;
let hoverState = null;

let gameState = {
  players: [],
  h_walls: [],
  v_walls: [],
  valid_moves: [],
  walls_remaining: [],
  num_players: 4,
  current_player: 0,
};

let lastCanvasSize = 0;

function resizeCanvas() {
  const gameContainer = document.getElementById("game-screen");
  if (!gameContainer.clientWidth) return; // hidden: nothing to size against
  const style = getComputedStyle(gameContainer);
  const padLeft = parseFloat(style.paddingLeft);
  const padRight = parseFloat(style.paddingRight);
  const containerWidth = gameContainer.clientWidth - padLeft - padRight;
  const panelWidth = window.innerWidth > 480 ? 126 : 0;
  const available = containerWidth - panelWidth;
  const layout = document.querySelector(".game-layout");

  // Measure the space left with the board collapsed. A short viewport (a phone
  // held sideways) must shrink the board rather than push the panel and buttons
  // below the fold: body is overflow:hidden, so anything past it is unreachable.
  const panel = document.querySelector(".walls-panel");
  const sidePanel = panelWidth > 0; // panel sits beside the board, not below it
  const prevCanvasHeight = canvas.style.height;
  const prevLayoutHeight = layout ? layout.style.height : "";
  const prevPanelDisplay = panel ? panel.style.display : "";
  canvas.style.height = "0px";
  if (layout) layout.style.height = "";
  // Beside the board the panel costs no extra height, so take it out of the
  // measurement; stacked below it, it does, so leave it in.
  if (sidePanel && panel) panel.style.display = "none";
  const roomBelow =
    window.innerHeight - gameContainer.getBoundingClientRect().bottom - 8;
  canvas.style.height = prevCanvasHeight;
  if (layout) layout.style.height = prevLayoutHeight;
  if (panel) panel.style.display = prevPanelDisplay;

  const size = Math.max(
    180,
    Math.min(Math.floor(available), Math.floor(roomBelow), 420),
  );
  if (size > 0 && Math.abs(size - lastCanvasSize) > 2) {
    lastCanvasSize = size;
    canvas.width = size;
    canvas.height = size;
    canvas.style.width = size + "px";
    canvas.style.height = size + "px";
    if (window.innerWidth > 480) {
      layout.style.height = size + "px";
    } else {
      layout.style.height = "";
    }
    if (gameState.players && gameState.players.length > 0) {
      drawBoard();
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }
}

resizeCanvas();
window.addEventListener("resize", resizeCanvas);

document.addEventListener("DOMContentLoaded", () => {
  const playerCountSelect = document.getElementById('num-players');
  const seatSelect = document.getElementById('seat-select');

  if (playerCountSelect && seatSelect) {
    // Kept in document order so they go back where they came from.
    const fourPlayerSeats = Array.from(seatSelect.options).filter(
      option => option.value === '2' || option.value === '3'
    );

    const updateSeatOptions = () => {
      const is4p = playerCountSelect.value === '4';

      // If switching to 2-player while holding seat 3 or 4, reset to Random
      if (!is4p && parseInt(seatSelect.value) > 1) {
        seatSelect.value = '-1';
      }

      // Detach rather than hide: Safari ignores display:none on <option>.
      fourPlayerSeats.forEach(option => {
        if (is4p && !option.parentNode) {
          seatSelect.appendChild(option);
        } else if (!is4p && option.parentNode) {
          option.remove();
        }
      });
    };

    playerCountSelect.addEventListener('change', updateSeatOptions);
    updateSeatOptions();
  }
});
