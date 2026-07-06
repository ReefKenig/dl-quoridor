// --- API & GAME LOGIC ---

let moveController = null;

async function startGame() {
  if (moveController) {
    moveController.abort();
    moveController = null;
  }
  const newGridSize = parseInt(document.getElementById("board-size").value);
  numPlayers = parseInt(document.getElementById("num-players").value);
  currentDifficulty = document.getElementById("difficulty").value;

  if (newGridSize !== currentGridSize) {
    currentGridSize = newGridSize;
    lastCanvasSize = 0;
  }

  mainMenu.classList.add("hidden");
  gameScreen.classList.remove("hidden");
  updateDifficultySwitcher();
  resetWallsInfoCache();
  gameState = { players: [], h_walls: [], v_walls: [], valid_moves: [], walls_remaining: [], num_players: numPlayers, current_player: 0 };
  drawWallsInfo();
  resizeCanvas();

  try {
    const response = await fetch(
      `/api/${currentGridSize}x${currentGridSize}/reset`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ num_players: numPlayers, difficulty: currentDifficulty }),
      },
    );

    const data = await response.json();
    isPlayerTurn = true;
    gameState = data;

    statusText.innerText = `🎮 ${numPlayers}-player game on ${currentGridSize}×${currentGridSize} — Your turn!`;
    drawBoard();
  } catch (error) {
    console.log("Failed to start game:", error);
    statusText.innerText = "❌ Server connection lost.";
  }
}

async function restartGame() {
  if (moveController) {
    moveController.abort();
    moveController = null;
  }
  restartBtn.classList.add("btn-hidden");
  resetWallsInfoCache();
  statusText.innerText = "Resetting...";
  gameState = { players: [], h_walls: [], v_walls: [], valid_moves: [], walls_remaining: [], num_players: numPlayers, current_player: 0 };
  drawWallsInfo();
  drawBoard();

  try {
    const response = await fetch(
      `/api/${currentGridSize}x${currentGridSize}/reset`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ num_players: numPlayers, difficulty: currentDifficulty }),
      },
    );
    const data = await response.json();

    isPlayerTurn = true;
    gameState = data;

    statusText.innerText = `🎮 New game! Your turn (Player 1).`;
    drawBoard();
  } catch (error) {
    console.error("Failed to restart:", error);
    statusText.innerText = "❌ Server connection lost.";
  }
}

function quitToMenu() {
  if (moveController) {
    moveController.abort();
    moveController = null;
  }
  gameScreen.classList.add("hidden");
  mainMenu.classList.remove("hidden");
  restartBtn.classList.add("btn-hidden");
}

function showInstructions() {
  document.getElementById("instructions-modal").classList.remove("hidden");
}

function hideInstructions(event) {
  if (event.target === document.getElementById("instructions-modal") || event.target.tagName === "BUTTON") {
    document.getElementById("instructions-modal").classList.add("hidden");
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function showToast(message, duration = 2500) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toast._timeout);
  toast._timeout = setTimeout(() => toast.classList.remove("show"), duration);
}

async function sendMoveToServer(type, targetRow, targetCol) {
  isPlayerTurn = false;

  // Optimistically show the player's move immediately
  if (type === "pawn" && gameState.players && gameState.players.length > 0) {
    gameState.players[0] = { row: targetRow, col: targetCol };
  } else if (type === "h_wall") {
    if (!gameState.h_walls) gameState.h_walls = [];
    gameState.h_walls.push({ row: targetRow, col: targetCol });
    if (gameState.walls_remaining && gameState.walls_remaining[0] > 0) {
      gameState.walls_remaining[0]--;
    }
  } else if (type === "v_wall") {
    if (!gameState.v_walls) gameState.v_walls = [];
    gameState.v_walls.push({ row: targetRow, col: targetCol });
    if (gameState.walls_remaining && gameState.walls_remaining[0] > 0) {
      gameState.walls_remaining[0]--;
    }
  }
  drawBoard();

  if (type === "pawn") {
    statusText.innerText = numPlayers === 2
      ? "P2 thinking..."
      : "P2, P3, P4 thinking...";
  } else {
    showToast("Wall placed", 1500);
    statusText.innerText = numPlayers === 2
      ? "P2 thinking..."
      : "P2, P3, P4 thinking...";
  }

  if (moveController) moveController.abort();
  moveController = new AbortController();
  const signal = moveController.signal;

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
        signal,
      },
    );

    if (signal.aborted) return;
    const data = await response.json();

    if (data.error) {
      showToast("⚠️ " + data.error);
      statusText.innerText = "🎯 Your turn (Player 1)";
      isPlayerTurn = true;
      // Re-sync state from server in case of drift
      try {
        const sync = await fetch(`/api/${currentGridSize}x${currentGridSize}/state`);
        if (sync.ok) {
          gameState = await sync.json();
          drawBoard();
        }
      } catch (_) {}
      return;
    }

    // Animate AI moves one by one
    if (data.ai_steps && data.ai_steps.length > 0) {
      let prevWallCount = (gameState.h_walls ? gameState.h_walls.length : 0)
        + (gameState.v_walls ? gameState.v_walls.length : 0);

      for (let i = 0; i < data.ai_steps.length; i++) {
        if (signal.aborted) return;
        const step = data.ai_steps[i];
        const playerNum = i + 2;
        const newWallCount = (step.h_walls ? step.h_walls.length : 0)
          + (step.v_walls ? step.v_walls.length : 0);
        const placedWall = newWallCount > prevWallCount;
        prevWallCount = newWallCount;

        if (numPlayers > 2) {
          statusText.innerText = placedWall
            ? `P${playerNum} placed a wall`
            : `P${playerNum} moved`;
        }
        await sleep(400);
        if (signal.aborted) return;
        gameState = step;
        drawBoard();
        if (i < data.ai_steps.length - 1) {
          await sleep(300);
        }
      }
    }

    // Always use newState as the authoritative final state
    if (!data.ai_steps || data.ai_steps.length === 0) {
      gameState = data.newState;
      drawBoard();
    } else {
      gameState = data.newState;
    }

    if (data.status === "game_over") {
      const winner = data.winner;
      if (winner === "human" || winner === 0) {
        statusText.innerText = "🏆 You win! Well played!";
      } else if (winner === null || winner === undefined) {
        statusText.innerText = "🤝 It's a draw!";
      } else {
        const pNum = data.ai_steps ? data.ai_steps.length + 1 : 2;
        statusText.innerText = `😤 Player ${pNum} (AI) beat you this time!`;
      }
      isPlayerTurn = false;
      restartBtn.classList.remove("btn-hidden");
      return;
    }

    statusText.innerText = "🎯 Your turn (Player 1)";
    isPlayerTurn = true;
  } catch (error) {
    if (error.name === "AbortError") return;
    console.error("Error communicating with AI:", error);
    statusText.innerText = "❌ Server connection lost.";
  }
}

function updateDifficultySwitcher() {
  document.querySelectorAll(".diff-opt").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.diff === currentDifficulty);
  });
}

function switchDifficulty(diff) {
  if (diff === currentDifficulty) return;
  currentDifficulty = diff;
  document.getElementById("difficulty").value = diff;
  updateDifficultySwitcher();
  restartGame();
}
