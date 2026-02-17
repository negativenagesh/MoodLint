class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]  # A list to hold the board state
        self.current_winner = None  # Keep track of the winner!

    def print_board(self):
        # We will print the board after each move
        for i in range(3):
            print('|'.join(self.board[i*3:(i+1)*3]))
            if i < 2:
                print('-' * 5)

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def empty_squares(self):
        return ' ' in self.board

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter  # Set the winner
            return True
        return False

    def winner(self, square, letter):
        # Check the row
        row_ind = square // 3
        if all([spot == letter for spot in self.board[row_ind * 3:(row_ind + 1) * 3]]):
            return True
        # Check the column
        col_ind = square % 3
        if all([self.board[col_ind + i * 3] == letter for i in range(3)]):
            return True
        # Check diagonals
        if square % 2 == 0:
            if all([self.board[i] == letter for i in [0, 4, 8]]):
                return True
        if square % 2 == 0 and square != 4:
            if all([self.board[i] == letter for i in [2, 4, 6]]):
                return True
        return False

    def play(self):
        letter = 'X'  # Starting letter
        while self.empty_squares():
            square = int(input(f'{letter} turn. Input move (0-8): '))
            if self.make_move(square, letter):
                self.print_board()
                if self.current_winner:
                    print(f'{letter} wins!')
                    return
                letter = 'O' if letter == 'X' else 'X'  # Switches player
        print('It\'s a tie!')

if __name__ == '__main__':
    game = TicTacToe()
    game.play()