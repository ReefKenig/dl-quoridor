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
  const style = getComputedStyle(gameContainer);
  const padLeft = parseFloat(style.paddingLeft);
  const padRight = parseFloat(style.paddingRight);
  const containerWidth = gameContainer.clientWidth - padLeft - padRight;
  const panelWidth = window.innerWidth > 480 ? 126 : 0;
  const available = containerWidth - panelWidth;
  const size = Math.min(Math.floor(available), 420);
  if (size > 0 && Math.abs(size - lastCanvasSize) > 2) {
    lastCanvasSize = size;
    canvas.width = size;
    canvas.height = size;
    canvas.style.width = size + "px";
    canvas.style.height = size + "px";
    const layout = document.querySelector(".game-layout");
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
