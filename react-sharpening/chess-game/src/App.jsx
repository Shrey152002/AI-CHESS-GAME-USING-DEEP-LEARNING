import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import ChessBoard from './ChessBoard';
import Tutorial from './Tutorial';
import AI from './AI';
import Home from './Home';
import Contatct from './Contatct';

function App() {
  return (
    
    <Router>
      <Routes>
        <Route path="/" exact element={<Home />} />
        <Route path="/tutorial" element={<Tutorial />} />
        <Route path="/ChessBoard" element={<ChessBoard />} />
        <Route path="/AI" element={<AI/>}/>
        <Route path="/Home" exact element={<Home />} />
        <Route path="/Contatct" element={<Contatct />} />
      </Routes>
    </Router>
  );
}

export default App;