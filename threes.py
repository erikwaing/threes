import random
import copy
import math
import numpy
from matplotlib import pyplot
import time

from evaluators import *

class Board:

	def __init__(self, rows, cols, player):
		self.rows = rows
		self.cols = cols
		self.board = {}
		self.reset()
		self.player = player
		self.moveNumber = 0

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
			choice = 0 #random.randint(0, len(openSpaces)-1)
			square = openSpaces[choice]
			self.insertTile(square, nextColor)
		
	def move(self):
		nextColor = self.moveNumber % 3 + 1 #random.randint(1, 3)
		self.moveNumber += 1
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

	def requestMove(self, game, nextColor):
		possibleMoves = ['Left', 'Right', 'Up', 'Down']
		maxscore = -float('inf')
		maxMove = 'Left'
		tempBoard = Board(game.rows, game.cols, None)
		for moves in self.sequences:
			tempBoard.board = copy.deepcopy(game.board)
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
		if self.canMove(game, maxMove):
			return maxMove
		else:
			for i in possibleMoves:
				if self.canMove(game,i):
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

class PrunningPlayer:

	def __init__(self, numLookAheads, keeps, evaluator):
		self.numLookAheads = numLookAheads
		self.evaluator = evaluator
		self.keeps = keeps

	def requestMove(self, game, nextColor):
		#game.drawBoard()
		games = self.generateGames(game, nextColor)
		for t in range(self.numLookAheads):
			games = self.branchGames(games)
			games = self.pruneGames(games)
			if self.allSameMove(games):
				break
		game = self.findMaximumGame(games)
		#print "Next Move: ", game[0][0]
		#for i in games:
		#	print i
		#raw_input('Press enter to continue')
		return game[0][0]

	def findMaximumGame(self, games):
		maxGame = None
		maxScore = -float('inf')
		assert len(games) > 0
		for game in games:
			if game[2] > maxScore:
				maxGame = game
				maxScore = game[2]
		return maxGame

	def generateGames(self, game, nextColor):
		moves = ["Left", "Right", "Up", "Down"]
		games = []
		for move in moves:
			if self.canMove(game, move):
				simgame = Board(game.rows, game.cols, self)
				simgame.moveNumber = game.moveNumber
				simgame.board = copy.deepcopy(game.board)
				simgame.executeMove(move)
				simgame.moveNumber += 1
				simgame.insertNewElement(move, nextColor)
				score = self.evaluator.evalBoard(simgame)
				games.append([[move], simgame, score])
		return games

	def allSameMove(self, games):
		if len(games) == 0:
			return True
		firstGameMove = games[0][0][0]
		for game in games:
			firstMove = game[0][0]
			if firstMove != firstGameMove:
				return False
		return True

	def pruneGames(self, games):
		newgames = []
		assert len(games) > 0
		for i in range(self.keeps):
			if len(games) == 0:
				break
			maxScore = -float('inf')
			maxGame = None
			for game in games:
				if game[2] > maxScore:
					maxGame = game
			newgames.append(maxGame)
			if maxGame in games:
				games.remove(maxGame)
		return newgames

	def branchGames(self, games):
		newgames = []
		moves = ["Left", "Right", "Up", "Down"]
		for move in moves:
			for game in games:
				if self.canMove(game[1], move):
					simgame = Board(game[1].rows, game[1].cols, self)
					simgame.moveNumber = game[1].moveNumber
					simgame.board = copy.deepcopy(game[1].board)
					simgame.executeMove(move)
					simgame.moveNumber += 1
					nextColor = simgame.moveNumber % 3 + 1 #random.randint(1,3)
					simgame.insertNewElement(move, nextColor)
					score = self.evaluator.evalBoard(simgame)
					game[0].append(move)
					newgames.append([game[0], simgame, score])
				else:
					newgames.append(game)
		return newgames

	def canMove(self, game, move):
		simgame = Board(game.rows, game.cols, self)
		simgame.board = copy.deepcopy(game.board)
		simgame.executeMove(move)
		simgame.insertNewElement(move, 1)
		if sum(simgame.board.values()) == sum(game.board.values()):
			return False
		else:
			return True

class StrategyTester:

	@staticmethod
	def testEvaluators(listOfEvaluators, numLookAheads, numGames, numberOfKeepsForPrunning, playerName='PrunningPlayer'):
		print "Testing Evaluators..."
		print "Number of Look Aheads: %d" % numLookAheads
		print "Number of Games: %d" % numGames
		for k, evaluator in enumerate(listOfEvaluators):
			print "Testing Evaluator %d: %s" % (k, str(evaluator))
			if playerName == 'PrunningPlayer':
				player = PrunningPlayer(numLookAheads, numberOfKeepsForPrunning, evaluator)
			else:
				player = Player(numLookAheads, evaluator)
			game = Board(4,4,player)
			scores = []
			for i in range(numGames):
				score = game.play()[1]
				scores.append(score)
				game.reset()
			print "Average Score: %d" % (sum(scores) / numGames)
			print "Median Score: %d" % numpy.median(numpy.array(scores))
			print "Max Score: %d" % max(scores)
			print "Min Score: %d" % min(scores)
			print "Standard Deviation: %d" % numpy.std(scores)
			#StrategyTester.plotScores(scores)
		#pyplot.show()

	@staticmethod
	def plotScores(scores):
		pyplot.figure()
		hist, bins = numpy.histogram(scores, bins=50)
		width = 0.7 * (bins[1] - bins[0])
		center = (bins[:-1] + bins[1:]) / 2
		pyplot.bar(center, hist, align='center', width=width)

class GeneticAlgorithm:

	CURRENT_BEST_GA_COEFFICIENTS = [0.12027059607492636, 0.1471465166193795, 0.07977407750482718, 0.11899206288905258, 0.12155425087545638, 0.10730058812753732, 0.09320185892593419, 0.07302076375403793]
	CURRENT_BEST_GA_EVALUATORS = [MaximizeScore(), SumOfSquares(), SumOfCubes(), Gravity(), SumOfBottom(), EmptySquares(), MinOneTwo(), PositionOfHighest()]

	def __init__(self, numOfGames, numLookAheads, branchingFactor, numOfKeeps):
		self.numOfGames = numOfGames
		self.numLookAheads = numLookAheads
		listOfCoefficients = numpy.eye(len(GeneticAlgorithm.CURRENT_BEST_GA_EVALUATORS)).tolist()
		self.currentBest = [CombinedEvaluators(GeneticAlgorithm.CURRENT_BEST_GA_EVALUATORS, coef) for coef in listOfCoefficients]
		self.currentCost = []
		self.branchingFactor = branchingFactor
		self.numOfKeeps = numOfKeeps

	def cost(self, evaluator):
		player = Player(self.numLookAheads, evaluator)
		game = Board(4,4,player)
		scores = []
		for i in range(self.numOfGames):
			score = game.play()[1]
			scores.append(score)
			game.reset()
		return - numpy.median(numpy.array(scores))

	def mutate(self, evaluator):
		newevaluators = [evaluator]
		coef = evaluator.listOfCoefficients
		for i in range(self.branchingFactor):
			newcoef = []
			for value in coef:
				newcoef.append(value + random.random()/10)
			total = sum(newcoef)
			newcoef = [value/total for value in newcoef]
			newevaluators.append(CombinedEvaluators(evaluator.listOfEvaluators, newcoef))
		return newevaluators

	def maintainBest(self):
		best = []
		cost = []
		for i in range(self.numOfKeeps):
			minIndex = self.currentCost.index(min(self.currentCost))
			cost.append(self.currentCost.pop(minIndex))
			best.append(self.currentBest.pop(minIndex))
		self.currentBest = best
		self.currentCost = cost

	def computeCosts(self):
		costs = []
		for evaluator in self.currentBest:
			costs.append(self.cost(evaluator))
		self.currentCost = costs

	def branchEvaluators(self):
		newlist = []
		for evaluator in self.currentBest:
			newlist = newlist + self.mutate(evaluator)
		self.currentBest = newlist

	def runIter(self):
		self.branchEvaluators()
		self.computeCosts()
		self.maintainBest()
		currentBestStrategy = self.currentBest[self.currentCost.index(min(self.currentCost))]
		return currentBestStrategy

	def runForeverWithUpdates(self):
		print "Running Genetic Algorithm (numOfGames=%d, numLookAheads=%d, branchingFactor=%d, numOfKeeps=%d)" % (self.numOfGames, self.numLookAheads, self.branchingFactor, self.numOfKeeps) 
		counter = 1
		while True:
			best = self.runIter()
			print "Current Best After %d Iterations: " % counter
			print "Evaluators: " + str(best.listOfEvaluators)
			print "Coefficients: " + str(best.listOfCoefficients)
			print "Score: %d" % (-self.currentCost[self.currentBest.index(best)])
			counter += 1

test = BoardTests()
test.runAllTests()

evalsToTest = [MaximizeScore(), SumOfSquares(), SumOfCubes(), Gravity(), SumOfBottom(), EmptySquares(), MinOneTwo(), PositionOfHighest()]
start = time.time()
StrategyTester.testEvaluators(evalsToTest, 2, 1, 100)
end = time.time()
print "Time Elapsed: " 
print end - start

print "Random Player"
rand = RandomPlayer()
game = Board(4,4, rand)
scores = []
for i in range(100):
	scores.append(game.play()[1])
	game.reset()
print numpy.mean(scores)


#ga = GeneticAlgorithm(10, 2, 4, 5)
#ga.runForeverWithUpdates()