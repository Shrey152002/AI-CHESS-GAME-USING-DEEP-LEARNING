let symbols = {
  "w-pawn": "♙",
  "w-knight": "♘",
  "w-bishop": "♗",
  "w-rook": "♖",
  "w-queen": "♕",
  "w-king": "♔",
  "b-pawn": "♟",
  "b-knight": "♞",
  "b-bishop": "♝",
  "b-rook": "♜",
  "b-queen": "♛",
  "b-king": "♚",
};

export const initialBoardState = [
  [
    "b-rook",
    "b-knight",
    "b-bishop",
    "b-queen",
    "b-king",
    "b-bishop",
    "b-knight",
    "b-rook",
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
  [null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null],
  [
    "w-pawn",
    "w-pawn",
    "w-pawn",
    "w-pawn",
    "w-pawn",
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

function Square({ isLight, piece, index_col, index_row, handleClick }) {
  return (
    <div
      className={`square ${isLight ? "light" : "dark"} ${
        piece ? "has-piece" : ""
      } `}
      onClick={() => handleClick(index_row, index_col,piece)}
    >
      {symbols[piece]}
    </div>
  );
}

export const Row = ({ isEvenRow, rowIndex, board, handleClick }) => {
  const squares = [];
  for (let i = 0; i < 8; i++) {
    const isLight = isEvenRow ? i % 2 === 0 : i % 2 !== 0;
    squares.push(
      <Square
        isLight={isLight}
        piece={board[rowIndex][i]}
        key={i}
        index_col={i}
        index_row={rowIndex}
        handleClick={handleClick}
      />
    );
  }
  return <div className="row">{squares}</div>;
};
