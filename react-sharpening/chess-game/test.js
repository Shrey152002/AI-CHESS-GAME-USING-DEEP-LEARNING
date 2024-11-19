import fetch from "node-fetch";
// The chessboard you want to send to the server
function stringToChessboard(str) {
  // Remove all newline characters from the string
  str = str.replace(/\n/g, "");

  // Split the string into an array of words
  let words = [];
  let word = "";
  for (let i = 0; i < str.length; i++) {
    // Add each word to the words array
    if (str[i] != " ") {
      word += str[i];
    } else if (str[i] == "") {
      continue;
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
      row.push(words[i * 8 + j]);
    }
    chessboard.push(row);
  }

  return chessboard;
}

let chessboard = [
  [
    "b-rook",
    "b-knight",
    "b-bishop",
    "b-queen",
    "b-king",
    "b-bishop",
    "b-knight",
    "null",
  ],
  [
    "b-pawn",
    "b-pawn",
    "b-pawn",
    "b-pawn",
    "b-pawn",
    "b-pawn",
    "b-pawn",
    "b-pawn",
  ],
  ["null", "b-rook", "null", "null", "null", "null", "null", "null"],
  ["null", "null", "null", "null", "null", "null", "null", "null"],
  ["null", "null", "null", "null", "w-pawn", "null", "null", "null"],
  ["null", "null", "null", "null", "null", "null", "null", "null"],
  [
    "w-pawn",
    "w-pawn",
    "w-pawn",
    "w-pawn",
    "null",
    "w-pawn",
    "w-pawn",
    "w-pawn",
  ],
  [
    "w-rook",
    "w-knight",
    "w-bishop",
    "w-queen",
    "w-king",
    "w-bishop",
    "w-knight",
    "w-rook",
  ],
];

//console.log(JSON.stringify({ chessboard: chessboard }))
// Make a POST request to the '/play' endpoint
fetch("http://localhost:5000/play", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ chessboard: chessboard }),
})
  .then((response) => response.text())
  .then((text) => {
    try {
      return JSON.parse(text);
    } catch (err) {
      console.error("Error parsing JSON:", text);
      throw err;
    }
  })
  .then((newBoard) => {
    console.log(stringToChessboard(newBoard));
  })
  .catch((error) => console.error("Error:", error));
