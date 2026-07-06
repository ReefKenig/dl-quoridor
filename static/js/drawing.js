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
  const radius = cellSize * 0.35;

  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI, false);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#fff";
  ctx.stroke();

  if (displayNumber !== null) {
    ctx.fillStyle = "#fff";
    const fontSize = Math.max(10, Math.round(cellSize * 0.28));
    ctx.font = `bold ${fontSize}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(displayNumber), centerX, centerY);
  }
}

let lastWallsInfoKey = "";

function drawWallsInfo() {
  const panel = document.getElementById("walls-panel");
  if (!gameState.walls_remaining) {
    if (lastWallsInfoKey !== "empty") {
      panel.innerHTML = "";
      lastWallsInfoKey = "empty";
    }
    return;
  }
  const np = gameState.num_players || numPlayers;
  const infoKey = gameState.walls_remaining.join(",") + ":" + gameState.current_player;
  if (infoKey === lastWallsInfoKey) return;
  lastWallsInfoKey = infoKey;

  let html = "";
  for (let i = 0; i < np; i++) {
    const color = PLAYER_COLORS[i];
    const label = i === 0 ? "You" : `P${i + 1}`;
    const count = gameState.walls_remaining[i];
    const maxWalls = gameState.walls_remaining.reduce((a, b) => Math.max(a, b), 1);
    const bricks = [];
    for (let w = 0; w < maxWalls; w++) {
      bricks.push(
        `<span class="wall-brick ${w < count ? "active" : "used"}" style="--color: ${color}"></span>`,
      );
    }
    html += `<div class="wall-player${gameState.current_player === i ? " current-turn" : ""}">
      <span class="wall-dot" style="background: ${color}"></span>
      <span class="wall-label">${label}</span>
      <span class="wall-bricks">${bricks.join("")}</span>
      <span class="wall-count">${count}</span>
    </div>`;
  }
  panel.innerHTML = html;
}
