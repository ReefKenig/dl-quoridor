// Player colors (indexed 0-3 internally, displayed 1-4)
const PLAYER_COLORS = ["#00b4d8", "#e63946", "#2ecc71", "#f1c40f"];

const AI_THINKING_MESSAGES = [
  "🤔 Player {n} is pondering...",
  "🧠 Player {n} is strategizing...",
  "💭 Player {n} is plotting...",
  "⚡ Player {n} is calculating...",
  "🎯 Player {n} is choosing wisely...",
  "🔮 Player {n} is reading the board...",
  "🎲 Player {n} is weighing options...",
  "🏃 Player {n} is finding a path...",
];

function getThinkingMessage(playerNum) {
  const msg =
    AI_THINKING_MESSAGES[Math.floor(Math.random() * AI_THINKING_MESSAGES.length)];
  return msg.replace("{n}", playerNum);
}

function getPlayerLabel(index, numPlayers) {
  // Display as 1-based
  if (index === 0) return "Player 1 (You)";
  return `Player ${index + 1} (AI)`;
}
