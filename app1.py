from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import chess
import torch.nn as nn
import torch.nn.functional as F
# Import your model and necessary functions here
# Assuming the code you provided is in a separate file or within this file

app = Flask(__name__)
class module(nn.Module):

    def __init__(self, hidden_size):
        super(module, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.activation1 = nn.SELU()
        self.activation2 = nn.SELU()

    def forward(self, x):
        x_input = torch.clone(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += x_input
        x = self.activation2(x)
        return x
class ChessNet(nn.Module):

    def __init__(self, hidden_layers=4, hidden_size=200):
        super(ChessNet, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
        self.module_list = nn.ModuleList([module(hidden_size) for i in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

    def forward(self, x):

        x = self.input_layer(x)
        x = F.relu(x)

        for i in range(self.hidden_layers):
            x = self.module_list[i](x)

        x = self.output_layer(x)

        return x

# Initialize your model here
model_path = 'C://Users//Shreyash Verma//Desktop//CHESS_GAME//model1.pth'
model = ChessNet(hidden_layers=4, hidden_size=200)
model.load_state_dict(torch.load(model_path))
model = model.cuda() if torch.cuda.is_available() else model
model.eval()

def predict_move(board, color):
    # Process the board and use the model to predict the next move
    b_mat = choose_move(board, color)
    return b_mat

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/make_move', methods=['POST'])
def make_move():
    data = request.json
    board = data['board']
    color = data['color']
    
    # Predict the move using the model
    b_mat = predict_move(board, color)
    
    return jsonify({'new_board': b_mat})

if __name__ == "__main__":
    app.run(debug=True)
