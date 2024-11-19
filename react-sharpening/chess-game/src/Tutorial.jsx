import React from 'react'
import './App.css'
import { Link } from 'react-router-dom'
import './index.css'
import Home from './Home'
import ChessBoard from './ChessBoard'

function Tutorial() {
  return (
    <div className="container2">
      <div className='tutparent'>
        <h1 className='tuth1'>Chess Tutorial</h1>

        <div className="contain-menu2">
          <nav>
            <ul>
              <li>
                <Link className='navlnk' to="/Home">Home</Link>
              </li>
              <li>
                <Link className='navlnk' to="/ChessBoard">Game</Link>
              </li>

              <li>
                <Link className='navlnk' to="/AI">AI</Link>
              </li>
              <li className='navlnk' >Contact</li>
            </ul>
          </nav>
        </div>

        <div className="chessboard-card">
          <h2 className="tuthcb">Chessboard</h2>
          <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Chess_Board.svg/1200px-Chess_Board.svg.png" alt="chessboard" className="tutimg" />
          <p className="tutpcb">The chessboard is divided up into sections called ranks and files, and the set is composed of different chessmen with different movements and powers. The chessmen are composed of eight white pieces and eight black pieces.</p>
        </div>

        <div className="piececard">
          <h2 className="tuth2">Pawn</h2>
          <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Chess_plt45.svg/800px-Chess_plt45.svg.png" alt="pawn" className="tutimg" />
          <p className="tutp">The pawn moves to one field forward or two fields at the first move of this figure; beats diagonally to one field forward. The pawn is also involved in the two special moves en passant and pawn promotion.</p>
        </div>
        
        <div className="piececard2">
          <h2 className="tuth2">Rook</h2>
          <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Chess_rlt45.svg/800px-Chess_rlt45.svg.png" alt="rook" className="tutimg" />
          <p className="tutp">The rook can move any number of squares along any rank or file, but may not leap over other pieces. Along with the king, a rook is involved during the king's castling move.</p>
        </div>
        <div className="piececard3">
          <h2 className="tuth2">Knight</h2>
          <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/Chess_nlt45.svg/800px-Chess_nlt45.svg.png" alt="knight" className="tutimg" />
          <p className="tutp">The knight moves to any of the closest squares that are not on the same rank, file, or diagonal, thus the move forms an "L"-shape: two squares vertically and one square horizontally, or two squares horizontally and one square vertically. The knight is the only piece that can leap over other pieces.</p>
        </div>
        <div className="piececard4">
          <h2 className="tuth2">Bishop</h2>
          <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Chess_blt45.svg/800px-Chess_blt45.svg.png" alt="bishop" className="tutimg" />
          <p className="tutp">The bishop has no restrictions in distance for each move, but is limited to diagonal movement. Bishops, like all other pieces except the knight, cannot jump over other pieces. A bishop captures by occupying the square on which an enemy piece sits.</p>
        </div>
        <div className="piececard5">
          <h2 className="tuth2">Queen</h2>
          <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Chess_qlt45.svg/800px-Chess_qlt45.svg.png" alt="queen" className="tutimg" />
          <p className="tutp">The queen combines the power of the rook and bishop and can move any number of squares along rank, file, or diagonal, but it may not leap over other pieces. As with all other pieces except the knight, the queen may not move to a square occupied by a friendly piece.</p>
        </div>
        <div className="piececard6">
          <h2 className="tuth2">King</h2>
          <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/Chess_klt45.svg/800px-Chess_klt45.svg.png" alt="king" className="tutimg" />
          <p className="tutp">The king moves one square in any direction. The king also has a special move called castling that involves also moving a rook.</p>
        </div>
      </div>
    </div>
  )
}

export default Tutorial;