import { Row, initialBoardState } from "./BoardComponent";
import { useState, useEffect } from "react";
import * as PossibleMoves from "./PossibleMoves";
import { Link } from "react-router-dom";
import "./App.css";

const NUM_PLAYERS = 2;
let turnIndex = 0;
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

  // Create an 8x8 2D array from the words
  let chessboard = [];
  for (let i = 0; i < 8; i++) {
    let row = [];
    for (let j = 0; j < 8; j++) {
      // If the word is "null", push null instead of the word
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
const ChessBoard = () => {
  const [board, setBoard] = useState(initialBoardState);
  const [previousMove, setPreviousMove] = useState();
  const turn = ["w", "b"];
  let whiteTurn;
  useEffect(() => {
    fetch("http://localhost:5000/play", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ chessboard: chessboardToNullString(board) }),
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
        whiteTurn = stringToChessboard(newBoard);
        if (turn[turnIndex % NUM_PLAYERS] === "w") {
          turnIndex++;
          setTimeout(() => {
            setBoard(whiteTurn);
          }, 1500);
        }
      })
      .catch((error) => console.error("Error:", error));
  }, [board]);
  const handleClick = (row, col, piece) => {
    if (previousMove && previousMove.piece) {
      const pieceType = previousMove.piece.split("-")[1];
      const color = previousMove.piece.split("-")[0];
      let possibleMoves = getPossibleMoves(
        pieceType,
        color,
        previousMove,
        board
      );
      if (turn[turnIndex % NUM_PLAYERS] === "b" && color === "b") {
        handleMove(
          row,
          col,
          piece,
          color,
          possibleMoves,
          board,
          setBoard,
          previousMove
        );
      }
    }

    setPreviousMove({ row, column: col, piece });
  };

  return (
    <div className="container">
      <div>
        <h1 className="titlecb">Chess-board</h1>
      </div>

      <div className="contain-menu">
        <nav>
          <ul>
            <li>
              <Link className="navlnk" exact to="/Home">
                Home
              </Link>
            </li>
            <li>
              <Link className="navlnk" to="/ChessBoard">
                Game
              </Link>
            </li>
            <li>
              <Link className="navlnk" to="/Tutorial">
                Tutorial
              </Link>
            </li>
            <li>
              <Link className="navlnk" to="/AI">
                AI
              </Link>
            </li>
            <li className="navlnk">Contact</li>
          </ul>
        </nav>
      </div>

      <div className="gamecontainer">
        {/* Chessboard */}

        <div className="board">
          {Array.from({ length: 8 }).map((_, i) => (
            <Row
              isEvenRow={i % 2 === 0}
              rowIndex={i}
              key={i}
              board={board}
              handleClick={(row, col) => handleClick(row, col, board[row][col])}
            />
          ))}
        </div>
        {/* Numbers for rows */}
        <div className="row-numbers">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="row-number">
              {8 - i}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
function getPossibleMoves(pieceType, color, previousMove, board) {
  let possibleMoves;
  switch (pieceType) {
    case "pawn":
      possibleMoves = PossibleMoves.getPawnMoves(
        [previousMove.column, previousMove.row],
        color,
        board
      );
      break;
    case "knight":
      possibleMoves = PossibleMoves.getKnightMoves([
        previousMove.column,
        previousMove.row,
      ]);
      break;
    case "bishop":
      possibleMoves = PossibleMoves.getBishopMoves(
        [previousMove.column, previousMove.row],
        board
      );
      break;
    case "rook":
      possibleMoves = PossibleMoves.getRookMoves(
        [previousMove.column, previousMove.row],
        board
      );
      break;
    case "queen":
      possibleMoves = PossibleMoves.getQueenMoves(
        [previousMove.column, previousMove.row],
        board
      );
      break;
    case "king":
      possibleMoves = PossibleMoves.getKingMoves([
        previousMove.column,
        previousMove.row,
      ]);
      break;
    default:
      return;
  }
  return possibleMoves;
}

function handleMove(
  row,
  col,
  piece,
  color,
  possibleMoves,
  board,
  setBoard,
  previousMove
) {
  if (piece === null || piece[0] !== color) {
    possibleMoves.some((move) => {
      if (move.X === col && move.Y === row) {
        let newBoard = [...board];
        newBoard[previousMove.row][previousMove.column] = null;
        newBoard[row][col] = previousMove.piece;
        setBoard(newBoard);
        turnIndex++;
        return true;
      }
      return false;
    });
  }
}

export default ChessBoard;
