// UI Elements
const mainMenu = document.getElementById("main-menu");
const gameScreen = document.getElementById("game-screen");
const canvas = document.getElementById("game-board");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("game-status");
const restartBtn = document.getElementById("restart-btn");

// Game State Variables
let currentGridSize = parseInt(document.getElementById("board-size").value);
let numPlayers = parseInt(document.getElementById("num-players").value);
let cellSize = 0;
let isPlayerTurn = true;
let hoverState = null;

// Player colors
const PLAYER_COLORS = ["#00b4d8", "#e63946", "#2ecc71", "#f1c40f"];
const PLAYER_LABELS = ["You", "AI-1", "AI-2", "AI-3"];

// Track game state from server
let gameState = {
  players: [],
  h_walls: [],
  v_walls: [],
  valid_moves: [],
  walls_remaining: [],
  num_players: 4,
  current_player: 0,
};

// --- NAVIGATION FUNCTIONS ---

async function startGame() {
  currentGridSize = parseInt(document.getElementById("board-size").value);
  numPlayers = parseInt(document.getElementById("num-players").value);
  const difficulty = document.getElementById("difficulty").value;

  mainMenu.classList.add("hidden");
  gameScreen.classList.remove("hidden");
  statusText.innerText = "Connecting to server...";

  try {
    const response = await fetch("/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ num_players: numPlayers }),
    });

    const data = await response.json();
    isPlayerTurn = true;
    gameState = data;

    statusText.innerText = `Game Started! ${numPlayers}P ${currentGridSize}x${currentGridSize} - Your turn`;
    drawBoard();
  } catch (error) {
    console.log("Failed to start game:", error);
    statusText.innerText = "Server connection lost.";
  }
}

async function restartGame() {
  statusText.innerText = "Restarting game...";
  restartBtn.classList.add("hidden");

  try {
    const response = await fetch("/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ num_players: numPlayers }),
    });
    const data = await response.json();

    isPlayerTurn = true;
    gameState = data;

    statusText.innerText = `Game Started! (${numPlayers}P) - Your turn.`;
    drawBoard();
  } catch (error) {
    console.error("Failed to restart:", error);
    statusText.innerText = "Server connection lost.";
  }
}

function quitToMenu() {
  gameScreen.classList.add("hidden");
  mainMenu.classList.remove("hidden");
  restartBtn.classList.add("hidden");
}

// --- DRAWING LOGIC ---

function drawBoard() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  cellSize = canvas.width / currentGridSize;

  // Draw grid squares
  ctx.fillStyle = "#1b263b";
  for (let row = 0; row < currentGridSize; row++) {
    for (let col = 0; col < currentGridSize; col++) {
      ctx.fillRect(
        col * cellSize + 5,
        row * cellSize + 5,
        cellSize - 10,
        cellSize - 10,
      );
    }
  }

  // Draw walls
  ctx.fillStyle = "#e94560";
  if (gameState.h_walls) {
    gameState.h_walls.forEach((wall) => {
      ctx.fillRect(
        wall.col * cellSize + 5,
        (wall.row + 1) * cellSize - 4,
        cellSize * 2 - 10,
        8,
      );
    });
  }

  if (gameState.v_walls) {
    gameState.v_walls.forEach((wall) => {
      ctx.fillRect(
        (wall.col + 1) * cellSize - 4,
        wall.row * cellSize + 5,
        8,
        cellSize * 2 - 10,
      );
    });
  }

  // Draw all player pawns
  if (gameState.players) {
    gameState.players.forEach((pos, i) => {
      drawPawn(pos.row, pos.col, PLAYER_COLORS[i], i);
    });
  }

  // Draw hover preview
  if (hoverState && isPlayerTurn) {
    ctx.globalAlpha = 0.4;

    if (hoverState.type === "pawn") {
      drawPawn(hoverState.row, hoverState.col, PLAYER_COLORS[0], null);
    } else if (hoverState.type === "h_wall") {
      ctx.fillStyle = "#e94560";
      ctx.fillRect(
        hoverState.col * cellSize + 5,
        (hoverState.row + 1) * cellSize - 4,
        cellSize * 2 - 10,
        8,
      );
    } else if (hoverState.type === "v_wall") {
      ctx.fillStyle = "#e94560";
      ctx.fillRect(
        (hoverState.col + 1) * cellSize - 4,
        hoverState.row * cellSize + 5,
        8,
        cellSize * 2 - 10,
      );
    }

    ctx.globalAlpha = 1.0;
  }

  // Draw walls-remaining info
  drawWallsInfo();
}

function drawPawn(row, col, color, playerIndex) {
  const centerX = col * cellSize + cellSize / 2;
  const centerY = row * cellSize + cellSize / 2;
  const radius = cellSize / 2 - 15;

  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI, false);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#fff";
  ctx.stroke();

  // Draw player number
  if (playerIndex !== null) {
    ctx.fillStyle = "#fff";
    ctx.font = "bold 14px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(playerIndex), centerX, centerY);
  }
}

function drawWallsInfo() {
  if (!gameState.walls_remaining) return;
  const infoY = canvas.height - 5;
  ctx.font = "12px sans-serif";
  ctx.textBaseline = "bottom";
  ctx.textAlign = "left";
  const np = gameState.num_players || numPlayers;
  for (let i = 0; i < np; i++) {
    ctx.fillStyle = PLAYER_COLORS[i];
    const label = i === 0 ? "You" : `AI${i}`;
    ctx.fillText(
      `${label}: ${gameState.walls_remaining[i]}w`,
      10 + i * 120,
      infoY,
    );
  }
}

// --- INTERACTION LOGIC ---

function getGridActionFromPixels(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (clientX - rect.left) * scaleX;
  const y = (clientY - rect.top) * scaleY;

  const gridX = x / cellSize;
  const gridY = y / cellSize;

  const distToHLine = Math.abs(gridY - Math.round(gridY));
  const distToVLine = Math.abs(gridX - Math.round(gridX));
  const W = currentGridSize - 1;

  let action = null;

  if (distToHLine < 0.25 && distToHLine < distToVLine) {
    const r = Math.round(gridY) - 1;
    const c = Math.floor(gridX - 0.5);
    if (r >= 0 && r < W && c >= 0 && c < W)
      action = { type: "h_wall", row: r, col: c };
  } else if (distToVLine < 0.25 && distToVLine < distToHLine) {
    const c = Math.round(gridX) - 1;
    const r = Math.floor(gridY - 0.5);
    if (r >= 0 && r < W && c >= 0 && c < W)
      action = { type: "v_wall", row: r, col: c };
  } else {
    const r = Math.floor(gridY);
    const c = Math.floor(gridX);
    if (r >= 0 && r < currentGridSize && c >= 0 && c < currentGridSize) {
      action = { type: "pawn", row: r, col: c };
    }
  }

  // Server-side validation filter
  if (action && gameState.valid_moves) {
    const isLegal = gameState.valid_moves.some(
      (move) =>
        move.type === action.type &&
        move.row === action.row &&
        move.col === action.col,
    );
    return isLegal ? action : null;
  }

  return null;
}

canvas.addEventListener("mousemove", function (event) {
  if (!isPlayerTurn) return;
  hoverState = getGridActionFromPixels(event.clientX, event.clientY);
  drawBoard();
});

canvas.addEventListener("mouseout", function () {
  hoverState = null;
  drawBoard();
});

canvas.addEventListener("click", function (event) {
  if (!isPlayerTurn) return;

  const action = getGridActionFromPixels(event.clientX, event.clientY);
  if (action) {
    sendMoveToServer(action.type, action.row, action.col);
    hoverState = null;
    drawBoard();
  }
});

async function sendMoveToServer(type, targetRow, targetCol) {
  statusText.innerText = "AI is thinking...";
  isPlayerTurn = false;

  try {
    const response = await fetch("/api/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        type: type,
        target: { row: targetRow, col: targetCol },
      }),
    });

    const data = await response.json();

    if (data.error) {
      alert("Invalid move: " + data.error);
      statusText.innerText = "Your turn";
      isPlayerTurn = true;
      return;
    }

    gameState = data.newState;
    drawBoard();

    if (data.status === "game_over") {
      const winner = data.winner;
      if (winner === 0) {
        statusText.innerText = "Game Over! You win!";
      } else if (winner === null) {
        statusText.innerText = "Game Over! Draw!";
      } else {
        statusText.innerText = `Game Over! Player ${winner} (AI) wins!`;
      }
      isPlayerTurn = false;
      restartBtn.classList.remove("hidden");
      return;
    }

    statusText.innerText = "Your turn";
    isPlayerTurn = true;
  } catch (error) {
    console.error("Error communicating with AI:", error);
    statusText.innerText = "Server connection lost.";
  }
}
