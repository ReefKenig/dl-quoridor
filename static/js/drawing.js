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

  // Draw walls, rotated into the human's view
  ctx.fillStyle = "#e94560";
  ["h_walls", "v_walls"].forEach((key) => {
    const type = key === "h_walls" ? "h_wall" : "v_wall";
    (gameState[key] || []).forEach((wall) => {
      const view = wallToView(type, wall.row, wall.col);
      drawWall(view.type, view.row, view.col);
    });
  });

  // Draw all player pawns
  if (gameState.players) {
    gameState.players.forEach((pos, i) => {
      const cell = cellToView(pos.row, pos.col);
      drawPawn(cell.row, cell.col, PLAYER_COLORS[i], i + 1);
    });
  }

  // Draw hover preview
  if (hoverState && isPlayerTurn) {
    ctx.globalAlpha = 0.4;

    const preview = actionToView(hoverState);
    if (preview.type === "pawn") {
      drawPawn(preview.row, preview.col, PLAYER_COLORS[userSeat], null);
    } else {
      ctx.fillStyle = "#e94560";
      drawWall(preview.type, preview.row, preview.col);
    }

    ctx.globalAlpha = 1.0;
  }

  // Draw walls-remaining info
  drawWallsInfo();
}

// Wall rects in view space: row/col are slot-grid indices, not cells.
function drawWall(type, row, col) {
  if (type === "h_wall") {
    ctx.fillRect(col * cellSize + 5, (row + 1) * cellSize - 4, cellSize * 2 - 10, 8);
  } else {
    ctx.fillRect((col + 1) * cellSize - 4, row * cellSize + 5, 8, cellSize * 2 - 10);
  }
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

function resetWallsInfoCache() {
  lastWallsInfoKey = "";
}

function drawWallsInfo() {
  const panel = document.getElementById("walls-panel");
  const np = gameState.num_players || numPlayers;
  const walls = gameState.walls_remaining && gameState.walls_remaining.length === np
    ? gameState.walls_remaining
    : Array(np).fill(0);
  const cp = gameState.current_player || 0;

  const infoKey = walls.join(",") + ":" + cp + ":" + userSeat;
  if (infoKey === lastWallsInfoKey) return;
  lastWallsInfoKey = infoKey;

  const maxWalls = walls.reduce((a, b) => Math.max(a, b), 0);
  let html = "";
  for (let i = 0; i < np; i++) {
    const color = PLAYER_COLORS[i];
    const label = i === userSeat ? "You" : `P${i + 1}`;
    const count = walls[i];
    const bricks = [];
    for (let w = 0; w < maxWalls; w++) {
      bricks.push(
        `<span class="wall-brick ${w < count ? "active" : "used"}" style="--color: ${color}"></span>`,
      );
    }
    html += `<div class="wall-player${cp === i ? " current-turn" : ""}">
      <span class="wall-dot" style="background: ${color}"></span>
      <span class="wall-label">${label}</span>
      <span class="wall-bricks">${bricks.join("")}</span>
      <span class="wall-count">${count}</span>
    </div>`;
  }
  panel.innerHTML = html;
}
