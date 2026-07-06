// --- API & GAME LOGIC ---

async function startGame() {
  currentGridSize = parseInt(document.getElementById("board-size").value);
  numPlayers = parseInt(document.getElementById("num-players").value);
  currentDifficulty = document.getElementById("difficulty").value;

  mainMenu.classList.add("hidden");
  gameScreen.classList.remove("hidden");
  updateDifficultySwitcher();
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
  restartBtn.classList.add("hidden");

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
  gameScreen.classList.add("hidden");
  mainMenu.classList.remove("hidden");
  restartBtn.classList.add("hidden");
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

  statusText.innerText = numPlayers === 2
    ? "P2 thinking..."
    : "P2, P3, P4 thinking...";

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
      for (let i = 0; i < data.ai_steps.length; i++) {
        const step = data.ai_steps[i];
        const playerNum = i + 2;
        statusText.innerText = `P${playerNum} ${getThinkingMessage()}`;
        await sleep(600);
        gameState = step;
        drawBoard();
        await sleep(300);
      }
    }

    // Always use newState as the authoritative final state
    gameState = data.newState;
    drawBoard();

    if (data.status === "game_over") {
      const winner = data.winner;
      if (winner === "human" || winner === 0) {
        statusText.innerText = "🏆 You win! Well played!";
      } else if (winner === null || winner === undefined) {
        statusText.innerText = "🤝 It's a draw!";
      } else {
        const pNum = typeof winner === "number" ? winner + 1 : 2;
        statusText.innerText = `😤 Player ${pNum} (AI) beat you this time!`;
      }
      isPlayerTurn = false;
      restartBtn.classList.remove("hidden");
      return;
    }

    statusText.innerText = "🎯 Your turn (Player 1)";
    isPlayerTurn = true;
  } catch (error) {
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
