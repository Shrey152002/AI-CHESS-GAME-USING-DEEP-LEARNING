let str =
  "   b-rook b-knight b-bishop b-queen b-king b-bishop b-knight b-rook " +
  "  b-pawn b-pawn b-pawn b-pawn b-pawn  b-pawn b-pawn b-pawn " +
  "   null   null null null null null null null " +
  "null   null null null null null null null " +
  "null null  null null null null null null " +
  "null null null     null null null null null " +
  "w-pawn   w-pawn    w-pawn w-pawn w-pawn w-pawn w-pawn w-pawn " +
  "w-rook   w-knight   w-bishop w-queen w-king w-bishop w-knight w-rook";

function stringToChessboard(str) {
  // Split the string into an array of words
  str = str.trim();
  let words = [];
  let word = "";
  for (let i = 0; i < str.length; i++) {
    // Add each word to the words array
    if (str[i] != " ") {
      word += str[i];
    } else if (str[i - 1] == str[i]) {
    } else {
      words.push(word);
      word = "";
    }
  }
  words.push(word);
  // Check that there are exactly 64 words

  // Create an 8x8 2D array from the words
  let chessboard = [];
  for (let i = 0; i < 8; i++) {
    let row = [];
    for (let j = 0; j < 8; j++) {
      // If the word is "null", push null; otherwise, push the word
      let word = words[i * 8 + j];
      row.push(word === "null" ? null : word);
    }
    chessboard.push(row);
  }

  return chessboard;
}
function chessboardToNullString(chessboard) {
  // Create a new chessboard to avoid mutating the original one
  let newChessboard = [];

  for (let i = 0; i < chessboard.length; i++) {
    let row = [];
    for (let j = 0; j < chessboard[i].length; j++) {
      // If the cell is null, push "null" instead of the cell
      let cell = chessboard[i][j];
      row.push(cell === null ? "null" : cell);
    }
    newChessboard.push(row);
  }

  return newChessboard;
}
let chessboard = stringToChessboard(str);
console.log(chessboard);
chessboard = chessboardToNullString(chessboard)
console.log(chessboard)