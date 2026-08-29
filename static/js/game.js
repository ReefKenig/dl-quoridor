// --- API & GAME LOGIC ---

let moveController = null;
let currentGameId = null;

function createInitialGameState() {
  const mid = Math.floor(currentGridSize / 2);
  const walls = currentGridSize === 5
    ? (numPlayers === 2 ? 3 : 4)
    : (numPlayers === 2 ? 10 : 5);
  const players = [
    { row: currentGridSize - 1, col: mid },
    { row: 0, col: mid },
  ];
  if (numPlayers > 2) {
    players.push(
      { row: mid, col: 0 },
      { row: mid, col: currentGridSize - 1 },
    );
  }
  return {
    players,
    h_walls: [],
    v_walls: [],
    valid_moves: [],
    walls_remaining: Array(numPlayers).fill(walls),
    num_players: numPlayers,
    current_player: 0,
  };
}

async function readResponseData(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.error) {
    throw new Error(data.error || `Request failed (${response.status})`);
  }
  return data;
}

// Shared body of startGame/restartGame: paint the placeholder board, ask the
// server for a fresh game and play the AI opening up to the human's seat.
async function launchGame(successText) {
  isPlayerTurn = false; // Lock board while loading
  gameState = createInitialGameState();
  statusText.innerText = `🎮 ${numPlayers}-player game. You are Player ${userSeat + 1}. ${userSeat === 0 ? "Your turn." : "Waiting for your turn..."}`;
  moveController = new AbortController();
  const signal = moveController.signal;

  drawWallsInfo();
  resizeCanvas();
  drawBoard();

  try {
    const response = await fetch(
      `/api/${currentGridSize}x${currentGridSize}/reset`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          {
            num_players: numPlayers,
            difficulty: currentDifficulty,
            game_id: currentGameId,
            human_seat: userSeat
          }
        ),
        signal,
      }
    );

    if (signal.aborted) return;
    const data = await readResponseData(response);
    currentGameId = data.game_id || currentGameId;
    gameState = data;
    userSeat = Number.isInteger(data.human_seat) ? data.human_seat : userSeat;

    drawBoard();

    // Play AI opening moves if human is not Seat 0
    if (data.initial_ai_steps && data.initial_ai_steps.length > 0) {
      await playAiSequence(data.initial_ai_steps, signal);
    }
    if (signal.aborted) return;

    statusText.innerText = successText(userSeat);
    isPlayerTurn = true; // Unlock board

  } catch (error) {
    if (error.name === "AbortError") return;
    console.error("Failed to start game:", error);
    isPlayerTurn = false;
    statusText.innerText = "❌ Server connection lost.";
    // Without this a failed reset leaves a locked board and no way to retry.
    restartBtn.classList.remove("btn-hidden");
  }
}

async function startGame() {
  if (moveController) {
    moveController.abort();
    moveController = null;
  }

  const newGridSize = parseInt(document.getElementById("board-size").value);
  numPlayers = parseInt(document.getElementById("num-players").value);
  currentDifficulty = document.getElementById("difficulty").value;

  const seatSelect = document.getElementById("seat-select");
  const rawSeat = seatSelect ? Number.parseInt(seatSelect.value, 10) : -1;
  userSeat = Number.isInteger(rawSeat) && rawSeat >= 0
    ? rawSeat % numPlayers
    : Math.floor(Math.random() * numPlayers);

  if (newGridSize !== currentGridSize) {
    currentGridSize = newGridSize;
    lastCanvasSize = 0;
  }

  mainMenu.classList.add("hidden");
  gameScreen.classList.remove("hidden");
  updateDifficultySwitcher();
  resetWallsInfoCache();

  await launchGame(
    seat => `🎮 ${numPlayers}-player game. You are Player ${seat + 1}! Your turn.`
  );
}

async function restartGame() {
  if (moveController) {
    moveController.abort();
    moveController = null;
  }
  restartBtn.classList.add("btn-hidden");
  resetWallsInfoCache();

  // Preserve the current human seat across a mid-game restart or difficulty
  // switch. Re-reading the menu selector here re-randomizes random-seat games.
  const seatSelect = document.getElementById("seat-select");
  if (seatSelect && Number.isInteger(parseInt(seatSelect.value, 10)) && parseInt(seatSelect.value, 10) >= 0) {
    userSeat = parseInt(seatSelect.value, 10) % numPlayers;
  }

  await launchGame(seat => `🎮 New game! You are Player ${seat + 1}. Your turn.`);
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
    gameState.players[userSeat] = { row: targetRow, col: targetCol };
  } else if (type === "h_wall") {
    if (!gameState.h_walls) gameState.h_walls = [];
    gameState.h_walls.push({ row: targetRow, col: targetCol });
    if (gameState.walls_remaining && gameState.walls_remaining[userSeat] > 0) {
      gameState.walls_remaining[userSeat]--;
    }
  } else if (type === "v_wall") {
    if (!gameState.v_walls) gameState.v_walls = [];
    gameState.v_walls.push({ row: targetRow, col: targetCol });
    if (gameState.walls_remaining && gameState.walls_remaining[userSeat] > 0) {
      gameState.walls_remaining[userSeat]--;
    }
  }

  gameState.current_player = (gameState.current_player + 1) % numPlayers;

  drawBoard();

  const thinkingText = numPlayers === 2 ? "🤖 Opponent thinking..." : "🤖 Opponents thinking...";

  if (type === "pawn") {
    statusText.innerText = thinkingText;
  } else {
    showToast("Wall placed", 1500);
    statusText.innerText = thinkingText;
  }

  if (moveController) moveController.abort();
  moveController = new AbortController();
  const signal = moveController.signal;

  document.querySelectorAll(".diff-opt").forEach(b => b.style.pointerEvents = "none");
  const diffSelect = document.getElementById("difficulty");
  if (diffSelect) diffSelect.disabled = true;

  try {
    const response = await fetch(
      `/api/${currentGridSize}x${currentGridSize}/move`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Game-Id": currentGameId || "" },
        body: JSON.stringify({
          type: type,
          target: { row: targetRow, col: targetCol },
          game_id: currentGameId,
        }),
        signal,
      },
    );

    if (signal.aborted) return;
    const data = await response.json();

    if (data.error) {
      showToast("⚠️ " + data.error);

      document.querySelectorAll(".diff-opt").forEach(b => b.style.pointerEvents = "auto");
      if (diffSelect) diffSelect.disabled = false;

      // A rejected move (400) leaves it our turn, but a 409 means the AI is
      // still to play - unlocking there lets the human move an AI pawn.
      let resumeTurn = response.status !== 409;

      // Re-sync state from server in case of drift
      try {
        const sync = await fetch(`/api/${currentGridSize}x${currentGridSize}/state?game_id=${encodeURIComponent(currentGameId || "")}`, { signal });
        if (sync.ok) {
          const syncState = await sync.json();
          if (signal.aborted) return;
          currentGameId = syncState.game_id || currentGameId;
          gameState = syncState;
          if (Number.isInteger(syncState.human_seat)) {
            userSeat = syncState.human_seat;
          }
          if (Number.isInteger(syncState.current_player)) {
            resumeTurn = syncState.current_player === userSeat;
          }
          drawBoard();
        }
      } catch (_) {}
      if (signal.aborted) return;

      isPlayerTurn = resumeTurn;
      statusText.innerText = resumeTurn
        ? `🎯 Your turn (Player ${userSeat + 1})`
        : "🤖 AI is thinking...";
      return;
    }

    if (data.ai_steps && data.ai_steps.length > 0) {
      await playAiSequence(data.ai_steps, signal);
      if (signal.aborted) return;
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
      if (winner === "human" || winner === userSeat) {
        statusText.innerText = "🏆 You win! Well played!";
      } else if (winner === null || winner === undefined) {
        statusText.innerText = "🤝 It's a draw!";
      } else {
        statusText.innerText = `😤 An AI beat you this time!`;
      }
      isPlayerTurn = false;
      restartBtn.classList.remove("btn-hidden");

      document.querySelectorAll(".diff-opt").forEach(b => b.style.pointerEvents = "auto");
      if (diffSelect) diffSelect.disabled = false;
      return;
    }

    statusText.innerText = `🎯 Your turn (Player ${userSeat + 1})`;
    isPlayerTurn = true;

    document.querySelectorAll(".diff-opt").forEach(b => b.style.pointerEvents = "auto");
    if (diffSelect) diffSelect.disabled = false;
  } catch (error) {
    if (error.name === "AbortError") return;
    console.error("Error communicating with AI:", error);
    statusText.innerText = "❌ Server connection lost.";

    document.querySelectorAll(".diff-opt").forEach(b => b.style.pointerEvents = "auto");
    if (diffSelect) diffSelect.disabled = false;
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
  const difficultySelect = document.getElementById("difficulty");
  if (difficultySelect) {
    difficultySelect.value = diff;
  }
  updateDifficultySwitcher();
  restartGame();
}

async function playAiSequence(steps, signal = null) {
  let prevWallCount = (gameState.h_walls ? gameState.h_walls.length : 0) + (gameState.v_walls ? gameState.v_walls.length : 0);

  for (let i = 0; i < steps.length; i++) {
    if (signal && signal.aborted) return;

    const step = steps[i];
    const pNum = Number.isInteger(step.moved_player)
      ? step.moved_player + 1
      : (step.current_player === 0 ? numPlayers : step.current_player);

    const newWallCount = (step.h_walls ? step.h_walls.length : 0) + (step.v_walls ? step.v_walls.length : 0);
    const placedWall = newWallCount > prevWallCount;
    prevWallCount = newWallCount;

    statusText.innerText = placedWall ? `P${pNum} placed a wall` : `P${pNum} moved`;

    await sleep(400)
    if (signal && signal.aborted) return;

    gameState = step;
    drawBoard();

    if (i < steps.length - 1) {
      await sleep(300);
    }
  }
}
