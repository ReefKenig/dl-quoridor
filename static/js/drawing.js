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
      drawPawn(pos.row, pos.col, PLAYER_COLORS[i], i + 1);
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

function drawPawn(row, col, color, displayNumber) {
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

  // Draw player number (1-based)
  if (displayNumber !== null) {
    ctx.fillStyle = "#fff";
    ctx.font = "bold 14px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(displayNumber), centerX, centerY);
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
    const label = i === 0 ? "P1 (You)" : `P${i + 1} (AI)`;
    ctx.fillText(
      `${label}: ${gameState.walls_remaining[i]}w`,
      10 + i * 130,
      infoY,
    );
  }
}
