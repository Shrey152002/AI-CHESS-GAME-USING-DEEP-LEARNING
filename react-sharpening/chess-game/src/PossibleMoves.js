// Algorithm for Possible moves of all pieces
export function isValidMove(x, y) {
  // Check if the move is within the board boundaries
  if (x >= 0 && x < 8 && y >= 0 && y < 8) {
    return true;
  }
  return false;
}
function getAllOppositionMoves( color, board) {

    const oppositionMoves = [];

    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        // Check if the color is different
        if (board[y][x] && board[y][x][0] !== color) {
          // Add the piece's moves to oppositionMoves
          if (board[y][x][1] === "p") {
            oppositionMoves.push(
              ...getPawnMoves([x, y], board[y][x][0], board)
            );
          } else if (board[y][x][1] === "r") {
            oppositionMoves.push(...getRookMoves([x, y], board));
          } else if (board[y][x][1] === "n") {
            oppositionMoves.push(...getKnightMoves([x, y], board));
          } else if (board[y][x][1] === "b") {
            oppositionMoves.push(...getBishopMoves([x, y], board));
          } else if (board[y][x][1] === "q") {
            oppositionMoves.push(...getQueenMoves([x, y], board));
          } else if (board[y][x][1] === "k") {
            oppositionMoves.push(...getKingMoves([x, y], board));
          }
        }
      }
    }

    return oppositionMoves;
  }


//Function for Pawns
export function getPawnMoves(position, color, board) {
  const moves = [];
  const [x, y] = position;

  // White pawn moves
  if (color === "b") {
    // Move one step forward
    if (isValidMove(x, y + 1)) {
      moves.push({ X: x, Y: y + 1 });
    }

    // Move two steps forward from starting position
    if (y === 1 && isValidMove(x, y + 2)) {
      moves.push({ X: x, Y: y + 2 });
    }

    // Capture diagonally
    if (isValidMove(x - 1, y + 1) && board[y + 1][x - 1] !== null) {
      moves.push({ X: x - 1, Y: y + 1 });
    }
    if (isValidMove(x + 1, y + 1) && board[y + 1][x + 1] !== null) {
      moves.push({ X: x + 1, Y: y + 1 });
    }
  }

  // Black pawn moves
  if (color === "w") {
    // Move one step forward
    if (isValidMove(x, y - 1)) {
      moves.push({ X: x, Y: y - 1 });
    }

    // Move two steps forward from starting position
    if (y === 6 && isValidMove(x, y - 2)) {
      moves.push({ X: x, Y: y - 2 });
    }

    // Capture diagonally
    if (isValidMove(x - 1, y - 1) && board[y - 1][x - 1] !== null) {
      moves.push({ X: x - 1, Y: y - 1 });
    }
    if (isValidMove(x + 1, y - 1) && board[y - 1][x + 1] !== null) {
      moves.push({ X: x + 1, Y: y - 1 });
    }
  }

  return moves;
}
// Function for rook
export function getRookMoves(position, board) {
  const moves = [];
  const [x, y] = position;

  // Move vertically
  for (let i = y + 1; i < 8; i++) {
    if (board[i][x]) {
      if (board[i][x][0] !== board[y][x][0]) {
        moves.push({ X: x, Y: i });
      }
      break;
    }
    moves.push({ X: x, Y: i });
  }
  for (let i = y - 1; i >= 0; i--) {
    if (board[i][x]) {
      if (board[i][x][0] !== board[y][x][0]) {
        moves.push({ X: x, Y: i });
      }
      break;
    }
    moves.push({ X: x, Y: i });
  }

  // Move horizontally
  for (let i = x + 1; i < 8; i++) {
    if (board[y][i]) {
      if (board[y][i][0] !== board[y][x][0]) {
        moves.push({ X: i, Y: y });
      }
      break;
    }
    moves.push({ X: i, Y: y });
  }
  for (let i = x - 1; i >= 0; i--) {
    if (board[y][i]) {
      if (board[y][i][0] !== board[y][x][0]) {
        moves.push({ X: i, Y: y });
      }
      break;
    }
    moves.push({ X: i, Y: y });
  }

  return moves;
}
// Function for knight
export function getKnightMoves(position) {
  const moves = [];
  const [x, y] = position;

  const possibleMoves = [
    [x + 2, y + 1],
    [x + 2, y - 1],
    [x - 2, y + 1],
    [x - 2, y - 1],
    [x + 1, y + 2],
    [x + 1, y - 2],
    [x - 1, y + 2],
    [x - 1, y - 2],
  ];

  for (const move of possibleMoves) {
    const [moveX, moveY] = move;
    if (isValidMove(moveX, moveY)) {
      moves.push({ X: moveX, Y: moveY });
    }
  }

  return moves;
}
// Function for bishop
export function getBishopMoves(position, board) {
  const moves = [];
  const [x, y] = position;

  // Diagonal moves
  const diagonalMoves = [
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1],
  ];

  for (const [dx, dy] of diagonalMoves) {
    let [moveX, moveY] = [x + dx, y + dy];
    while (isValidMove(moveX, moveY)) {
      if (board[moveY][moveX]) {
        if (board[moveY][moveX][0] !== board[y][x][0]) {
          moves.push({ X: moveX, Y: moveY });
        }
        break;
      }
      moves.push({ X: moveX, Y: moveY });
      moveX += dx;
      moveY += dy;
    }
  }

  return moves;
}
// Function for queen
export function getQueenMoves(position, board) {
  const moves = [];
  const [x, y] = position;

  // Diagonal moves
  const diagonalMoves = [
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1],
  ];

  // Horizontal and vertical moves
  const horizontalVerticalMoves = [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
  ];

  // Check diagonal moves
  for (const [dx, dy] of diagonalMoves) {
    let [moveX, moveY] = [x + dx, y + dy];
    while (isValidMove(moveX, moveY)) {
      if (board[moveY][moveX]) {
        if (board[moveY][moveX][0] !== board[y][x][0]) {
          moves.push({ X: moveX, Y: moveY });
        }
        break;
      }
      moves.push({ X: moveX, Y: moveY });
      moveX += dx;
      moveY += dy;
    }
  }

  // Check horizontal and vertical moves
  for (const [dx, dy] of horizontalVerticalMoves) {
    let [moveX, moveY] = [x + dx, y + dy];
    while (isValidMove(moveX, moveY)) {
      if (board[moveY][moveX]) {
        if (board[moveY][moveX][0] !== board[y][x][0]) {
          moves.push({ X: moveX, Y: moveY });
        }
        break;
      }
      moves.push({ X: moveX, Y: moveY });
      moveX += dx;
      moveY += dy;
    }
  }

  return moves;
}
//Function for king
export function getKingMoves(position) {
  const moves = [];
  const [x, y] = position;

  // Possible moves for king
  const possibleMoves = [
    [1, 0],
    [1, 1],
    [0, 1],
    [-1, 1],
    [-1, 0],
    [-1, -1],
    [0, -1],
    [1, -1],
  ];
  // Check each possible move
  for (const [dx, dy] of possibleMoves) {
    const moveX = x + dx;
    const moveY = y + dy;

    if (isValidMove(moveX, moveY)) {
      moves.push({ X: moveX, Y: moveY });
    }
  }

  return moves;
}
