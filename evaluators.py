
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

class MaximizeScore:
	
	def evalBoard(self, Board):
		return Board.score()

class SumOfBottom:
	def evalBoard(self,Board):
		score = 0
		for key in Board.board.keys():
			if key[1] == Board.rows - 1:
				score += Board.board[key]
		return score

class Gravity:
	def evalBoard(self,Board):
		score = 0
		for key in Board.board.keys():
			score += -1*key[1]*Board.board[key]
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

class CombinedEvaluators:
	
	def __init__(self, listOfEvaluators, listOfCoefficients):
		self.listOfEvaluators = listOfEvaluators
		self.listOfCoefficients = listOfCoefficients

	def evalBoard(self, Board):
		scores = [evaluator.evalBoard(Board)*self.listOfCoefficients[i] for i,evaluator in enumerate(self.listOfEvaluators)]
		return sum(scores)

	@staticmethod
	def generateRandom():
		evaluators = [MaximizeScore(), SumOfSquares(), SumOfCubes(), Gravity(), SumOfBottom(), EmptySquares(), MinOneTwo(), PositionOfHighest(), ClosenessOfValues()]
		coef = [random.random() for i in evaluators]
		total = sum(coef)
		coef = [value / total for value in coef]
		return CombinedEvaluators(evaluators, coef)
