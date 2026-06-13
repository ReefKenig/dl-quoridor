// UI Elements
const mainMenu = document.getElementById("main-menu");
const gameScreen = document.getElementById("game-screen");
const canvas = document.getElementById("game-board");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("game-status");
const restartBtn = document.getElementById("restart-btn");

// Game State Variables
let currentGridSize = parseInt(document.getElementById("board-size").value);
let cellSize = 0;
let isPlayerTurn = true; // Prevents clicking while waiting for AI
let hoverState = null;

// Track position
let gameState = {
  player1: { row: 4, col: 2 },
  player2: { row: 0, col: 2 },
  h_walls: [],
  v_walls: [],
};

// --- NAVIGATION FUNCTIONS ---

async function startGame() {
  // Read settings from the menu
  currentGridSize = parseInt(document.getElementById("board-size").value);
  const difficulty = document.getElementById("difficulty").value;

  // Hide menu, show game
  mainMenu.classList.add("hidden");
  gameScreen.classList.remove("hidden");
  statusText.innerText = "Connecting to server...";

  // Ask Python for the official starting positions
  try {
    const response = await fetch(
      `/api/${currentGridSize}x${currentGridSize}/reset`,
      { method: "POST" },
    );

    const data = await response.json();

    isPlayerTurn = true;

    // Overwrite the local JS state with the true Python state
    gameState = {
      player1: data.player1,
      player2: data.player2,
      h_walls: data.h_walls,
      v_walls: data.v_walls,
      valid_moves: data.valid_moves,
    };

    statusText.innerText = `Game Started! ${currentGridSize}x${currentGridSize} - Player 1's turn`;
    drawBoard();
  } catch (error) {
    console.log("Failed to start game:", error);
    statusText.innerText = "Server connection lost.";
    return;
  }
}

async function restartGame() {
  statusText.innerText = "Restarting game...";
  restartBtn.classList.add("hidden");

  try {
    const response = await fetch(
      `/api/${currentGridSize}x${currentGridSize}/reset`,
      { method: "POST" },
    );
    const data = await response.json();

    isPlayerTurn = true;
    gameState = data;

    statusText.innerText = `Game Started! (${currentGridSize}x${currentGridSize}) - Your turn.`;
  } catch (error) {
    console.error("Failed to restart:", error);
    (statusText, (innerText = "Server connection lost."));
  }
}

function quitToMenu() {
  // Hide game, show menu
  gameScreen.classList.add("hidden");
  mainMenu.classList.remove("hidden");
  restartBtn.classList.add("hidden");
}

// --- DRAWING LOGIC ---

function drawBoard() {
  // Clear the canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Calculate how big each square should be based on the grid size
  // We leave a little math room for the walls between squares
  cellSize = canvas.width / currentGridSize;

  // Draw the grid squares
  ctx.fillStyle = "#1b263b";
  for (let row = 0; row < currentGridSize; row++) {
    for (let col = 0; col < currentGridSize; col++) {
      // Draw a square with a slight gap for the walls
      ctx.fillRect(
        col * cellSize + 5,
        row * cellSize + 5,
        cellSize - 10,
        cellSize - 10,
      );
    }
  }

  // Draw walls (bright red)
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
        wall.row * cellSize - 5,
        8,
        cellSize * 2 - 10,
      );
    });
  }

  // Draw Player 1 (Human - Cyan)
  if (gameState.player1)
    drawPawn(gameState.player1.row, gameState.player1.col, "#00b4d8");

  // Draw Player 2 (AI - Red)
  if (gameState.player2)
    drawPawn(gameState.player2.row, gameState.player2.col, "#e63946");

  // Draw hover Preview (Semi-transparent)
  if (hoverState && isPlayerTurn) {
    ctx.globalAlpha = 0.4;

    if (hoverState.type === "pawn") {
      drawPawn(hoverState.row, hoverState.col, "#00b4d8");
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

    ctx.globalAlpha = 1.0; // Reset transparancy
  }
}

function drawPawn(row, col, color) {
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

  // 1. Declare the action variable (Do NOT return yet!)
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

  // 2. --- SERVER-SIDE VALIDATION FILTER ---
  // Only process if an action was detected AND the server gave us the cheat-sheet
  if (action && gameState.valid_moves) {
    const isLegal = gameState.valid_moves.some(
      (move) =>
        move.type === action.type &&
        move.row === action.row &&
        move.col === action.col,
    );

    if (isLegal) {
      return action; // Safely return ONLY if Python explicitly approved it
    } else {
      return null; // It's illegal. Return null to hide the ghost hover.
    }
  }

  return null; // Fallback
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

// Listen for clicks on the canvas
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

  // Send the player's requested move to Flask
  try {
    const response = await fetch(
      `/api/${currentGridSize}x${currentGridSize}/move`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: type,
          target: { row: targetRow, col: targetCol },
        }),
      },
    );

    const data = await response.json();

    if (data.error) {
      // Invalid move
      alert("Invalid move: " + data.error);
      statusText.innerText = "Player 1's turn";
      isPlayerTurn = true;
      return;
    }

    // Update the board with the new state from the server
    gameState = data.newState;
    drawBoard();

    // Game Over lock
    if (data.status === "game_over") {
      const winner = data.winner === "ai" ? "AI" : "Player 1";
      statusText.innerText = `Game Over! Winner: ${data.winner}`;
      isPlayerTurn = false;
      restartBtn.classList.remove("hidden");
      return;
    }

    // Give control back to the player
    statusText.innerText = "Player 1's turn";
    isPlayerTurn = true;
  } catch (error) {
    console.error("Error communicating with AI:", error);
    statusText.innerText = "Server connection lost.";
  }
}
