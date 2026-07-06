// --- API & GAME LOGIC ---

async function startGame() {
  currentGridSize = parseInt(document.getElementById("board-size").value);
  numPlayers = parseInt(document.getElementById("num-players").value);

  mainMenu.classList.add("hidden");
  gameScreen.classList.remove("hidden");
  resizeCanvas();
  statusText.innerText = "Connecting to server...";

  try {
    const response = await fetch(
      `/api/${currentGridSize}x${currentGridSize}/reset`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ num_players: numPlayers }),
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
  statusText.innerText = "Restarting game...";
  restartBtn.classList.add("hidden");
  resizeCanvas();

  try {
    const response = await fetch(
      `/api/${currentGridSize}x${currentGridSize}/reset`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ num_players: numPlayers }),
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

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function sendMoveToServer(type, targetRow, targetCol) {
  isPlayerTurn = false;

  // Show thinking messages for AI opponents
  if (numPlayers === 2) {
    statusText.innerText = getThinkingMessage("2");
  } else {
    const names = [];
    for (let i = 1; i < numPlayers; i++) names.push(i + 1);
    statusText.innerText = `🤔 Players ${names.join(", ")} are thinking...`;
  }

  // Small delay so the user sees the thinking message
  await sleep(300);

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
      alert("Invalid move: " + data.error);
      statusText.innerText = "🎯 Your turn (Player 1)";
      isPlayerTurn = true;
      return;
    }

    // Add a small "reveal" delay after AI responds
    await sleep(400);

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
