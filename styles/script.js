const board = Chessboard('board', {
    draggable: true,
    position: 'start',
    onDrop: handleMove,
});

const game = new Chess();

function handleMove(source, target) {
    const move = game.move({
        from: source,
        to: target,
        promotion: 'q' // Automatically promote to queen
    });

    if (move === null) {
        return 'snapback'; // Illegal move
    }

    updateBoard();

    // Send the move to the server to get the AI's move
    fetch('/play', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ chessboard: game.fen() })
    })
    .then(response => response.json())
    .then(data => {
        game.load(data.new_fen);
        updateBoard();
    });
}

function updateBoard() {
    board.position(game.fen());
}
