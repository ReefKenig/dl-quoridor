const PLAYER_COLORS = ["#00b4d8", "#e63946", "#2ecc71", "#f1c40f"];

const AI_THINKING_MESSAGES = [
  "Thinking...",
  "Strategizing...",
  "Plotting...",
  "Calculating...",
  "Reading the board...",
  "Weighing options...",
  "Finding a path...",
];

function getThinkingMessage() {
  return AI_THINKING_MESSAGES[Math.floor(Math.random() * AI_THINKING_MESSAGES.length)];
}

function getPlayerLabel(index) {
  if (index === 0) return "Player 1 (You)";
  return `Player ${index + 1} (AI)`;
}
