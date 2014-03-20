import random
import copy
import math

class Board:

	def __init__(self, rows, cols, player):
		self.rows = rows
		self.cols = cols
		self.board = {}
		self.reset()
		self.player = player

	def reset(self):
		for i in range(self.rows):
			for j in range(self.cols):
				self.board[(i,j)] = 0

	def hasMoveToCombine(self):
		for i in range(self.rows):
			for j in range(self.cols):
				if i > 0:
					if self.board[(i,j)] == self.board[(i-1, j)] and self.board[(i,j)] != 1 and self.board[(i,j)] != 2 and self.board[(i,j)] != 0:
						return True
					elif self.board[(i,j)]*self.board[(i-1, j)] == 2:
						return True
				if j > 0:
					if self.board[(i, j)] == self.board[(i, j-1)] and self.board[(i,j)]!= 1 and self.board[(i,j)] != 2 and self.board[(i,j)] != 0:
						return True
					if self.board[(i,j)]*self.board[(i,j-1)] == 2:
						return True
		return False

	def hasEmptyTile(self):
		for i in range(self.rows):
			for j in range(self.cols):
				if self.board[(i, j)] == 0:
					return True
		return False

	def hasAvailableMove(self):
		return (self.hasEmptyTile() or self.hasMoveToCombine())

	def moveLeft(self):
		for i in range(self.rows):
			canCombineLeft = True
			for j in range(self.cols - 1):
				if self.board[(i,j)] == 0:
					self.board[(i,j)] = self.board[(i, j+1)]
					self.board[(i,j+1)] = 0
					canCombineLeft = False
				elif self.board[(i,j)] == self.board[(i,j+1)]: 
					if self.board[(i,j)] != 1 and self.board[(i,j)] != 2 and canCombineLeft:
						self.board[(i,j)] += self.board[(i,j)]
						self.board[(i,j+1)] = 0
						canCombineLeft = False
				elif self.board[(i,j+1)] != 0:
					if self.board[(i,j)] + self.board[(i,j+1)] == 3 and canCombineLeft:
						self.board[(i,j)] += self.board[(i,j+1)]
						self.board[(i,j+1)] = 0
						canCombineLeft = False

	def moveRight(self):
		for i in range(self.rows):
			canCombineRight = True
			for j in reversed(range(1, self.cols)):
				if self.board[(i,j)] == 0:
					self.board[(i,j)] = self.board[(i, j-1)]
					self.board[(i,j-1)] = 0
					canCombineRight = False
				elif self.board[(i,j)] == self.board[(i,j-1)]: 
					if self.board[(i,j)] != 1 and self.board[(i,j)] != 2 and canCombineRight:
						self.board[(i,j)] += self.board[(i,j)]
						self.board[(i,j-1)] = 0
						canCombineRight = False
				elif self.board[(i,j-1)] != 0: 
					if self.board[(i,j)] + self.board[(i,j-1)] == 3 and canCombineRight:
						self.board[(i,j)] += self.board[(i,j-1)]
						self.board[(i,j-1)] = 0
						canCombineRight = False

	def moveUp(self):
		for j in range(self.cols):
			canCombineUp = True
			for i in range(self.rows - 1):
				if self.board[(i,j)] == 0:
					self.board[(i,j)] = self.board[(i+1, j)]
					self.board[(i+1,j)] = 0
					canCombineUp = False
				elif self.board[(i,j)] == self.board[(i+1,j)]: 
					if self.board[(i,j)] != 1 and self.board[(i,j)] != 2 and canCombineUp:
						self.board[(i,j)] += self.board[(i,j)]
						self.board[(i+1,j)] = 0
						canCombineUp = False
				elif self.board[(i+1,j)] != 0:
					if self.board[(i,j)] + self.board[(i+1,j)] == 3 and canCombineUp:
						self.board[(i,j)] += self.board[(i+1,j)]
						self.board[(i+1,j)] = 0
						canCombineUp = False

	def moveDown(self):
		for j in range(self.cols):
			canCombineDown = True
			for i in reversed(range(1, self.rows)):
				if self.board[(i,j)] == 0:
					self.board[(i,j)] = self.board[(i-1, j)]
					self.board[(i-1,j)] = 0
					canCombineDown = False
				elif self.board[(i,j)] == self.board[(i-1,j)]: 
					if self.board[(i,j)] != 1 and self.board[(i,j)] != 2 and canCombineDown:
						self.board[(i,j)] += self.board[(i,j)]
						self.board[(i-1,j)] = 0
						canCombineDown = False
				elif self.board[(i-1,j)] != 0: 
					if self.board[(i,j)] + self.board[(i-1,j)] == 3 and canCombineDown:
						self.board[(i,j)] += self.board[(i-1,j)]
						self.board[(i-1,j)] = 0
						canCombineDown = False

	def printLine(self):
		string = ''
		for i in range(self.cols*2+1):
			string += '='
		print string

	def drawBoard(self):
		self.printLine()
		for i in range(self.rows):
			string = '|'
			for j in range(self.cols):
				if self.board[(i,j)] != 0:
					string += str(self.board[(i,j)]) + '|'
				else:
					string += '*|'
			print string
		self.printLine()

	def executeMove(self, nextMove):
		if nextMove == 'Left':
			self.moveLeft()
		elif nextMove == 'Right':
			self.moveRight()
		elif nextMove == 'Up':
			self.moveUp()
		elif nextMove == 'Down':
			self.moveDown()

	def askPlayerForMove(self, nextColor):
		nextMove = self.player.requestMove(self, nextColor)
		self.executeMove(nextMove)
		self.insertNewElement(nextMove, nextColor)

	def getAvailableSpacesForNextTileOnSide(self, MoveMade):
		if MoveMade == 'Up' or MoveMade == 'Down':
			raise Exception()
		if MoveMade == 'Left':
			col = self.cols - 1
		elif MoveMade == 'Right':
			col = 0	
		openSpaces = [(row,col) for row in range(self.rows) if self.board[(row,col)] == 0]
		return openSpaces

	def getAvailableSpacesForNextTimeOnUpDown(self, MoveMade):
		if MoveMade == 'Left' or MoveMade == 'Right':
			raise Exception()
		if MoveMade == 'Up':
			row = self.rows - 1
		elif MoveMade == 'Down':
			row = 0
		openSpaces = [(row, col) for col in range(self.cols) if self.board[(row,col)] == 0]
		return openSpaces

	def insertTile(self, square, nextColor):
		self.board[square] = nextColor

	def insertNewElement(self, MoveMade, nextColor):
		openSpaces = []
		if MoveMade == 'Up' or MoveMade == 'Down':
			openSpaces = self.getAvailableSpacesForNextTimeOnUpDown(MoveMade)
		elif MoveMade == 'Left' or MoveMade == 'Right':
			openSpaces = self.getAvailableSpacesForNextTileOnSide(MoveMade)
		if len(openSpaces) >= 1:
			choice = random.randint(0, len(openSpaces)-1)
			square = openSpaces[choice]
			self.insertTile(square, nextColor)
		
	def move(self):
		nextColor = random.randint(1, 3)
		self.askPlayerForMove(nextColor)

	def play(self):
		counter = 0
		while self.hasAvailableMove():
			counter+=1
			self.move()
		return (counter, self.score())

	def score(self):
		score = 0
		for value in self.board.values():
			if value == 1 or value == 2 or value == 0:
				score += value
			else:
				score += math.pow(3, math.log(value/3, 2)+1)
		return score


class BoardTests:

	def __init__(self):

		self.hasMoveToCombine = []
		self.hasEmptyTile = []

		self.boards = []
		board = {}
		board[(0,0)] =0
		board[(0,1)] =2
		board[(0,2)] =1
		board[(1,0)] =0
		board[(1,1)] =0
		board[(1,2)] =0
		board[(2,0)] =0
		board[(2,1)] =0
		board[(2,2)] =0

		self.hasEmptyTile.append(True)
		self.hasMoveToCombine.append(True)

		self.boards.append(board)
		board = {}
		board[(0,0)] =0
		board[(0,1)] =0
		board[(0,2)] =0
		board[(1,0)] =0
		board[(1,1)] =6
		board[(1,2)] =6
		board[(2,0)] =0
		board[(2,1)] =0
		board[(2,2)] =0

		self.hasEmptyTile.append(True)
		self.hasMoveToCombine.append(True)

		self.boards.append(board)
		board = {}
		board[(0,0)] = 1
		board[(0,1)] = 1
		board[(0,2)] = 1
		board[(1,0)] = 1
		board[(1,1)] = 1
		board[(1,2)] = 1
		board[(2,0)] = 1
		board[(2,1)] = 3
		board[(2,2)] = 3

		self.hasEmptyTile.append(False)
		self.hasMoveToCombine.append(True)

		self.boards.append(board)
		board = {}
		board[(0,0)] =2
		board[(0,1)] =2
		board[(0,2)] =2
		board[(1,0)] =2
		board[(1,1)] =2
		board[(1,2)] =2
		board[(2,0)] =2
		board[(2,1)] =2
		board[(2,2)] =2

		self.hasEmptyTile.append(False)
		self.hasMoveToCombine.append(False)

		self.boards.append(board)
		board = {}
		board[(0,0)] =1
		board[(0,1)] =0
		board[(0,2)] =0
		board[(1,0)] =2
		board[(1,1)] =3
		board[(1,2)] =6
		board[(2,0)] =12
		board[(2,1)] =24
		board[(2,2)] =48

		self.hasEmptyTile.append(True)
		self.hasMoveToCombine.append(True)

		self.boards.append(board)
		board = {}
		board[(0,0)] =1
		board[(0,1)] =1
		board[(0,2)] =1
		board[(1,0)] =1
		board[(1,1)] =1
		board[(1,2)] =1
		board[(2,0)] =1
		board[(2,1)] =1
		board[(2,2)] =0

		self.hasEmptyTile.append(True)
		self.hasMoveToCombine.append(False)

		self.boards.append(board)
		board = {}
		board[(0,0)] = 1
		board[(0,1)] = 0
		board[(0,2)] = 0
		board[(1,0)] =0
		board[(1,1)] =0
		board[(1,2)] =0
		board[(2,0)] =0
		board[(2,1)] =0
		board[(2,2)] =0

		self.hasEmptyTile.append(True)
		self.hasMoveToCombine.append(False)

		self.boards.append(board)
		board = {}
		board[(0,0)] =3
		board[(0,1)] =0
		board[(0,2)] =0
		board[(1,0)] =0
		board[(1,1)] =3
		board[(1,2)] =0
		board[(2,0)] =0
		board[(2,1)] =0
		board[(2,2)] =3

		self.hasEmptyTile.append(True)
		self.hasMoveToCombine.append(False)

		self.boards.append(board)
		board = {}
		board[(0,0)] =0
		board[(0,1)] =0
		board[(0,2)] =0
		board[(1,0)] =0
		board[(1,1)] =0
		board[(1,2)] =0
		board[(2,0)] =0
		board[(2,1)] =0
		board[(2,2)] =0

		self.hasEmptyTile.append(True)
		self.hasMoveToCombine.append(False)

		self.boards.append(board)
		board = {}
		board[(0,0)] =1
		board[(0,1)] =1
		board[(0,2)] =1
		board[(1,0)] =1
		board[(1,1)] =1
		board[(1,2)] =1
		board[(2,0)] =1
		board[(2,1)] =1
		board[(2,2)] =1
		self.boards.append(board)

		self.hasEmptyTile.append(False)
		self.hasMoveToCombine.append(False)

		board = {}
		board[(0,0)] =0
		board[(0,1)] =0
		board[(0,2)] =0
		board[(1,0)] =0
		board[(1,1)] =0
		board[(1,2)] =0
		board[(2,0)] =0
		board[(2,1)] =2
		board[(2,2)] =1
		self.boards.append(board)

		self.hasEmptyTile.append(True)
		self.hasMoveToCombine.append(True)

	def testHasMoveToCombine(self):
		player = RandomPlayer()
		for i in range(len(self.boards)):
			board = self.boards[i]
			hasMoveToCombine = self.hasMoveToCombine[i]
			rows = int(math.sqrt(len(board.keys())))
			cols = rows
			game = Board(rows, cols, player)
			game.board = board
			if hasMoveToCombine != game.hasMoveToCombine():
				print "ERROR: BOARD NUMBER %d" % i
				game.drawBoard()
				print "Has Move To Combine: " + str(hasMoveToCombine)
				print "Output from program: " + str(game.hasMoveToCombine())
				raise Exception('Values do not match!')

	def testHasEmptyTile(self):
		player = RandomPlayer()
		for i in range(len(self.boards)):
			board = self.boards[i]
			hasEmptyTile = self.hasEmptyTile[i]
			rows = int(math.sqrt(len(board.keys())))
			cols = rows
			game = Board(rows, cols, player)
			game.board = board
			if hasEmptyTile != game.hasEmptyTile():
				print "ERROR: BOARD NUMBER %d" % i
				game.drawBoard()
				print "Has Empty Tile: " + str(hasEmptyTile)
				print "Output from program: " + str(game.hasEmptyTile())
				raise Exception('Values do not match!')

	def runAllTests(self):
		print "Running testHasMoveToCombine..."
		self.testHasMoveToCombine()
		print "Running testHasEmptyTile..."
		self.testHasEmptyTile()

class RandomPlayer:

	def requestMove(self, Board, nextColor):
		moves = ['Left', 'Right', 'Up', 'Down']
		while True:
			choice = random.randint(0, len(moves) - 1)
			move = moves[choice]
			if self.canMove(Board, move):
				return move

	def canMove(self, game, move):
		simgame = Board(game.rows, game.cols, self)
		simgame.board = copy.deepcopy(game.board)
		simgame.executeMove(move)
		simgame.insertNewElement(move, 1)
		if sum(simgame.board.values()) == sum(game.board.values()):
			return False
		else:
			return True

class Player:

	def __init__(self, numLookAheads, evaluator):
		self.numLookAheads = numLookAheads
		self.evaluator = evaluator
		self.sequences = self.allSequencesOfLength(numLookAheads)

	def allSequencesOfLength(self, n):
		if n == 1:
			return [['Left'], ['Right'], ['Up'], ['Down']]
		else:
			toAdd = ['Left', 'Right', 'Up', 'Down']
			sequences = []
			previous = self.allSequencesOfLength(n-1)
			for next in toAdd:
				for after in previous:
					current = copy.deepcopy(after)
					current.append(next)
					sequences.append(current)
			return sequences

	def requestMove(self, Board, nextColor):
		possibleMoves = ['Left', 'Right', 'Up', 'Down']
		maxscore = 0
		maxMove = 'Left'
		for moves in self.sequences:
			tempBoard = copy.deepcopy(Board)
			for move in moves:
				if self.canMove(tempBoard, move):
					tempBoard.executeMove(move)
					tempBoard.insertNewElement(move, nextColor)
					nextColor = random.randint(1,3)
				else:
					break
			boardScore = self.evaluator.evalBoard(tempBoard)
			if boardScore > maxscore:
				maxscore = boardScore
				maxMove = moves[0]
		if self.canMove(Board, maxMove):
			return maxMove
		else:
			for i in possibleMoves:
				if self.canMove(Board,i):
					return i

	def canMove(self, game, move):
		simgame = Board(game.rows, game.cols, self)
		simgame.board = copy.deepcopy(game.board)
		simgame.executeMove(move)
		simgame.insertNewElement(move, 1)
		if sum(simgame.board.values()) == sum(game.board.values()):
			return False
		else:
			return True

class SumOfSquares:

	def evalBoard(self, Board):
		score = 0
		for val in Board.board.values():
			score += val*val
		return score

class SumOfCubes:
	def evalBoard(self, Board):
		score = 0
		for val in Board.board.values():
			score += val**3
		return score

class EmptySquares:

	def evalBoard(self, Board):
		score = 0
		for val in Board.board.values():
			if val == 0:
				score += 1
		return score

class MinOneTwo:

	def evalBoard(self, Board):
		score = 0
		for val in Board.board.values():
			if val != 1 and val != 2:
				score += 1
		return score

class PositionOfHighest:

	def evalBoard(self, Board):
		# only valid in 4 by 4
		valuesToSum = { (1, 1): 0,
						(1, 2): 0,
						(2, 1): 0,
						(2, 2): 0,
						(0, 0): 2,
						(3, 3): 2,
						(3, 0): 2,
						(0, 3): 2,
						(3, 1): 1,
						(1, 3): 1,
						(3, 2): 1,
						(2, 3): 1,
						(0, 1): 1,
						(0, 2): 1,
						(1, 0): 1,
						(2, 0): 1}
		maxLoc = None
		maxValue = max(Board.board.values())
		for key in Board.board.keys():
			if Board.board[key] == maxValue:
				maxLoc = key
		return valuesToSum[maxLoc]

class ClosenessOfValues:

	def evalBoard(self, Board):
		score = 0
		directions = [(1, 0), (0, 1)]
		for pos in Board.board.keys():
			(x, y) = pos
			closestValue = float('inf')
			for d in directions:
				(a, b) = (x + d[0], y + d[1])
				if (a,b) in Board.board.keys() and Board.board[(a,b)]*Board.board[(x,y)] != 0:
					difference = abs(Board.board[(a,b)] - Board.board[(x,y)])
					if difference < closestValue:
						closestValue = difference
			score -= closestValue
		return score

class StrategyTester:

	@staticmethod
	def testEvaluators(listOfEvaluators, numLookAheads, numGames):
		for evaluator in listOfEvaluators:
			print "Testing Evaluator: " + str(evaluator)
			player = Player(numLookAheads, evaluator)
			game = Board(4,4,player)
			totalScore = 0
			maxScore = 0
			minScore = float('inf')
			for i in range(numGames):
				score = game.play()[1]
				game.reset()
				totalScore += score
				if score > maxScore:
					maxScore = score
				if score < minScore:
					minScore = score
			print "Average Score: %d" % (totalScore / numGames)
			print "Max Score: %d" % maxScore
			print "Min Score: %d" % minScore

test = BoardTests()
test.runAllTests()

StrategyTester.testEvaluators([SumOfSquares(), SumOfCubes(), EmptySquares(), MinOneTwo(), PositionOfHighest(), ClosenessOfValues()], 2, 10)
