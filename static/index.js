// UI Elements
const mainMenu = document.getElementById("main-menu");
const gameScreen = document.getElementById("game-screen");
const canvas = document.getElementById("game-board");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("game-status");

// Game State Variables
let currentGridSize = parseInt(document.getElementById("board-size").value);
let cellSize = 0;
let isPlayerTurn = true; // Prevents clicking while waiting for AI

// Track position
let gameState = {
  player1: {},
  player2: {},
  walls: [], // Wall coordinates will be stored here
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
    };

    statusText.innerText = `Game Started! ${currentGridSize}x${currentGridSize} - Player 1's turn`;
    drawBoard();
  } catch (error) {
    console.log("Failed to start game:", error);
    statusText.innerText = "Server connection lost.";
    return;
  }
}

function quitToMenu() {
  // Hide game, show menu
  gameScreen.classList.add("hidden");
  mainMenu.classList.remove("hidden");
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

  // Draw Player 1 (Human - Cyan)
  drawPawn(gameState.player1.row, gameState.player1.col, "#00b4d8");

  // Draw Player 2 (AI - Red)
  drawPawn(gameState.player2.row, gameState.player2.col, "#e63946");

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

// Listen for clicks on the canvas
canvas.addEventListener("click", function (event) {
  // Calculate where the user clicked relative to the canvas
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  // Map the pixel coordinate to a grid coordinate
  const col = Math.floor(x / cellSize);
  const row = Math.floor(y / cellSize);

  // Find exactly where inside the cell the user clicked
  const offsetX = x % cellSize;
  const offsetY = y & cellSize;

  // The visual gap between cells is on the bottom and right edges
  if (offsetX > cellSize - 10 && col < currentGridSize - 1) {
    sendMoveToServer("v_wall", row, col);
  } else if (offsetY > cellSize - 10 && row < currentGridSize - 1) {
    sendMoveToServer("h_wall", row, col);
  } else {
    sendMoveToServer("pawn", row, col);
  }
});

async function sendMoveToServer(type, targetRow, targetCol) {
  if (!isPlayerTurn) return; // Don't receive clicks if AI is thinking

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
