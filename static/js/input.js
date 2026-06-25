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
