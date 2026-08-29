// --- BOARD VIEW ROTATION ---
// The server always speaks board coordinates with seat 0 at the bottom. The
// human sits at the bottom facing up whatever their seat, so the canvas draws
// a rotated view and input rotates back. Seat colors and numbers are unchanged.

// Quarter turns clockwise that bring each seat's home edge to the bottom.
const SEAT_VIEW_TURNS = [0, 2, 3, 1];

function viewTurns() {
  return SEAT_VIEW_TURNS[userSeat] || 0;
}

function cellToView(row, col) {
  const N = currentGridSize;
  switch (viewTurns()) {
    case 1: return { row: col, col: N - 1 - row };
    case 2: return { row: N - 1 - row, col: N - 1 - col };
    case 3: return { row: N - 1 - col, col: row };
    default: return { row, col };
  }
}

function cellFromView(row, col) {
  const N = currentGridSize;
  switch (viewTurns()) {
    case 1: return { row: N - 1 - col, col: row };
    case 2: return { row: N - 1 - row, col: N - 1 - col };
    case 3: return { row: col, col: N - 1 - row };
    default: return { row, col };
  }
}

function flipWall(type) {
  return type === "h_wall" ? "v_wall" : "h_wall";
}

// Wall anchors live on the (N-1)x(N-1) slot grid, and a quarter turn swaps
// horizontal and vertical walls.
function wallToView(type, row, col) {
  const N = currentGridSize;
  switch (viewTurns()) {
    case 1: return { type: flipWall(type), row: col, col: N - 2 - row };
    case 2: return { type, row: N - 2 - row, col: N - 2 - col };
    case 3: return { type: flipWall(type), row: N - 2 - col, col: row };
    default: return { type, row, col };
  }
}

function wallFromView(type, row, col) {
  const N = currentGridSize;
  switch (viewTurns()) {
    case 1: return { type: flipWall(type), row: N - 2 - col, col: row };
    case 2: return { type, row: N - 2 - row, col: N - 2 - col };
    case 3: return { type: flipWall(type), row: col, col: N - 2 - row };
    default: return { type, row, col };
  }
}

// Convert a whole action between board and view space.
function actionToView(action) {
  if (action.type === "pawn") {
    const cell = cellToView(action.row, action.col);
    return { type: "pawn", row: cell.row, col: cell.col };
  }
  return wallToView(action.type, action.row, action.col);
}

function actionFromView(action) {
  if (action.type === "pawn") {
    const cell = cellFromView(action.row, action.col);
    return { type: "pawn", row: cell.row, col: cell.col };
  }
  return wallFromView(action.type, action.row, action.col);
}
