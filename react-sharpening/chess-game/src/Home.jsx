import React from 'react'
import { Link } from "react-router-dom";
import "./App.css";
import "./index.css"
import ChessBoard from './ChessBoard';


function Home() {
  return (
<div >
{/* Main content starts from here  */}
<div className="menu3">
<nav className='nav3'>
          <ul>
            <li>
              <Link className="navlnk3" to="/Home">Home</Link>
            </li>
            <li>
              <Link className="navlnk3" to="/ChessBoard">Game</Link>
              </li>
            <li>
              <Link  className="navlnk3" to="/Tutorial">Tutorial</Link>
            </li>
            <li>
              <Link className="navlnk3" to="/AI">AI</Link>
            </li>
          </ul>
        </nav>
        </div>
<div className='rightone'>
<h1>
  One 
  <br />
  Stop
  <br />
  Platform
  <br />
  For
  <br />
  Chess.
</h1>
<Link to="/ChessBoard" >
<button className="btn btnplyn">
  Play Now
  </button>
  </Link>
<Link to="/ChessBoard">
  <button className="btn"  >
    Play Multiplayer
    </button>
    </Link>
    <Link to="/Tutorial">
    <button className="btn">
Take Tutorial
    </button>
    </Link>
    <div className="cbCard">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Chess_plt45.svg/800px-Chess_plt45.svg.png" alt="Image 2" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Chess_rlt45.svg/800px-Chess_rlt45.svg.png" alt="Image 3" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/Chess_nlt45.svg/800px-Chess_nlt45.svg.png" alt="Image 4" />

</div>
</div>
<div className=' leftone' >
<h1>
Bored?
  <br />
  Let's Chess!
</h1>
<span className="features">
  <h3>
    Features
  </h3>
  1. Play with your friends.
  <br />
  2. Play with AI.
  <br />
  3. Easy-card based Tutorial.
  <br />
  4. Coustom -trained Model
</span>
<div className="footercontainer">
<a href="https://github.com/thedemonsid"><img src="src/Images/github.png" alt="" /></a>
<a href=" https://www.gmail.com/siddheshshrirame@gmail.com"><img src="src/Images/mail-inbox-app.png" alt="" /></a>
<a href="https://twitter.com/the_demon_sid"><img src="src/Images/twitter.png" alt="" /></a>
</div>
</div>
</div>
  )
}

export default Home