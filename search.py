import util
from game import Directions

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class Node(object):
        def __init__(self, value, direction, cost, parent = None, parentPathCost = 0, heuristic = 0):
                self.value = value
                self.direction = direction
                self.cost = cost
                self.parent = parent
                self.pathCost = parentPathCost + cost
                self.heurPathSum = heuristic + self.pathCost

        #when object instance is printed, prints value list
        def __repr__(self):
                return self.getValue().__str__() + ", " + self.direction + ", " + str(self.cost)

        #shows this branch of the tree all the way up to the original parent
        def getPath(self, branch = []):
                branch.insert(0,self)
                if self.parent == None:
                	return branch

                return self.getParent().getPath(branch)
        
        def getValue(self):
                return self.value

        def getParent(self):
                return self.parent

        def getCost(self):
            return self.cost

        def getDirection(self):
            return self.direction

        def getPathCost(self):
        	return self.pathCost

        def getHeurPathSum(self):
        	return self.heurPathSum

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

#--frameTime 0 to speed up pacman
#python2 pacman.py -l tinyMaze -p SearchAgent -a fn=depthFirstSearch
#python2 pacman.py -l mediumMaze -p SearchAgent -a fn=depthFirstSearch
#python2 pacman.py -l bigMaze -p SearchAgent -a fn=depthFirstSearch --frameTime 0 
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())

    closedSet =  set()
    fringe = [Node(problem.getStartState(), "None", 0)]

    while(True):
        #check if we have failed
        if not fringe:
            print "Failed to Sort"
            return
        target = len(fringe)-1 #Depth Search
        currentNode = fringe[target] #what we are currently looking at
        position = currentNode.getValue()
        print position
        #make sure node wasn't already checked
        if position not in closedSet:
            closedSet.add(position)

            #if we are at the goal state then return a path of directions
            if problem.isGoalState(position):
                finalPath = []
                branchPath = currentNode.getPath()
                for i in branchPath:
                    if i.getDirection() == 'North':
                        finalPath.append(Directions.NORTH)
                    elif i.getDirection() == 'South':
                        finalPath.append(Directions.SOUTH)
                    elif i.getDirection() == 'West':
                        finalPath.append(Directions.WEST)
                    elif i.getDirection() == 'East':
                        finalPath.append(Directions.EAST)
                #print "The path we have found is: " + str(branchPath)
                return finalPath

            #expand the current Node
            expansion = problem.getSuccessors(position)

            #add expansion to fringe
            for i in expansion:
                fringe.append(Node(i[0], i[1], i[2], currentNode))

        #delete the fringe that was expanded
        del fringe[target]


#--frameTime 0 to speed up pacman
#python2 pacman.py -l tinyMaze -p SearchAgent -a fn=breadthFirstSearch
#python2 pacman.py -l mediumMaze -p SearchAgent -a fn=breadthFirstSearch
#python2 pacman.py -l bigMaze -p SearchAgent -a fn=breadthFirstSearch --frameTime 0
def breadthFirstSearch(problem):
    closedSet =  set()
    fringe = [Node(problem.getStartState(), "None", 0)]

    while(True):
        #check if we have failed
        if not fringe:
            print "Failed to Sort"
            return
        target = 0 #Breadth first Search
        currentNode = fringe[target] #what we are currently looking at
        position =  currentNode.getValue()

        #make sure node wasn't already checked
        if position not in closedSet:
            closedSet.add(position)

            #if we are at the goal state then return a path of directions
            if problem.isGoalState(position):
                finalPath = []
                branchPath = currentNode.getPath()
                for i in branchPath:
                    if i.getDirection() == 'North':
                        finalPath.append(Directions.NORTH)
                    elif i.getDirection() == 'South':
                        finalPath.append(Directions.SOUTH)
                    elif i.getDirection() == 'West':
                        finalPath.append(Directions.WEST)
                    elif i.getDirection() == 'East':
                        finalPath.append(Directions.EAST)
                #print "The path we have found is: " + str(branchPath)
                return finalPath

            #expand the current Node
            expansion = problem.getSuccessors(position)

            #add expansion to fringe
            for i in expansion:
                fringe.append(Node(i[0], i[1], i[2], currentNode))

        #delete the fringe that was expanded
        del fringe[target]

#--frameTime 0 to speed up pacman
#python2 pacman.py -l mediumDottedMaze -p StayEastSearchAgent
#python2 pacman.py -l mediumScaryMaze -p StayWestSearchAgent
#python2 pacman.py -l bigMaze -p SearchAgent -a fn=uniformCostSearch --frameTime 0
def uniformCostSearch(problem):
    closedSet =  set()
    fringe = [Node(problem.getStartState(), "None", 0)]

    while(True):
        #check if we have failed
        if not fringe:
            print "Failed to Sort"
            return

        #set target to lowest cost in fringe
        target = 0
        currentCost = fringe[0].getCost()
        for i in range(0, len(fringe)):
        	if fringe[i].getCost() < currentCost :
        		#print "currentCost: " +  str(currentCost) + " fringeCost: " + str(fringe[i].getCost())
        		target = i
        		currentCost = fringe[i].getCost()

        currentNode = fringe[target] #what we are currently looking at
        position =  currentNode.getValue()

        #make sure node wasn't already checked
        if position not in closedSet:
            closedSet.add(position)

            #if we are at the goal state then return a path of directions
            if problem.isGoalState(position):
                finalPath = []
                branchPath = currentNode.getPath()
                for i in branchPath:
                    if i.getDirection() == 'North':
                        finalPath.append(Directions.NORTH)
                    elif i.getDirection() == 'South':
                        finalPath.append(Directions.SOUTH)
                    elif i.getDirection() == 'West':
                        finalPath.append(Directions.WEST)
                    elif i.getDirection() == 'East':
                        finalPath.append(Directions.EAST)
                #print "The path we have found is: " + str(branchPath)
                return finalPath

            #expand the current Node
            expansion = problem.getSuccessors(position)

            #add expansion to fringe
            for i in expansion:
                fringe.append(Node(i[0], i[1], i[2], currentNode, currentNode.getPathCost()))

        #delete the fringe that was expanded
        del fringe[target]


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

#for astar big maze:
#python2 pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic

#for astar corners:
#python2 pacman.py -l tinyCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
#python2 pacman.py -l mediumCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
#using corners heuristic
#python2 pacman.py -l mediumCorners -p AStarCornersAgent -z 0.5

#for food search
#python2 pacman.py -l testSearch -p AStarFoodSearchAgent
#python2 pacman.py -l tinySearch -p AStarFoodSearchAgent
#python2 pacman.py -l trickySearch -p AStarFoodSearchAgent
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    closedDict = dict()
    fringe = [Node(problem.getStartState(), "None", 0, None, 0, heuristic(problem.getStartState(), problem))]

    while(True):
        if not fringe:
            print "Failed to Sort"
            return

        #Choose least cost leading up to path + heuristic
       	target = 0
       	currentNode = fringe[target]
       	currentTotalCost = currentNode.getHeurPathSum()
       	for i in range(0, len(fringe)):
        	if fringe[i].getHeurPathSum() < currentTotalCost :
        		target = i
        		currentNode = fringe[i]
        		currentTotalCost = fringe[i].getHeurPathSum()

        position =  currentNode.getValue()

        #only look at node if its not been looked at before or better than the one we have stored
        if (position not in closedDict or 
           currentTotalCost < closedDict[position].getHeurPathSum()):

        	#store the node
            closedDict[position] = currentNode

			#if we are at the goal state then return a path of directions
            if problem.isGoalState(position):
                finalPath = []
                branchPath = currentNode.getPath()
                for i in branchPath:
                    if i.getDirection() == 'North':
                        finalPath.append(Directions.NORTH)
                    elif i.getDirection() == 'South':
                        finalPath.append(Directions.SOUTH)
                    elif i.getDirection() == 'West':
                        finalPath.append(Directions.WEST)
                    elif i.getDirection() == 'East':
                        finalPath.append(Directions.EAST)
                #print "The path we have found is: " + str(branchPath)
                return finalPath

            #expand the current Node
            expansion = problem.getSuccessors(position)

            #add expansion to fringe
            for i in expansion:
            	#(value, direction, cost, parent, parentPathCost, heuristic)
                fringe.append(Node(i[0], i[1], i[2], currentNode, currentNode.getPathCost(), heuristic(i[0], problem)))

        #delete the fringe that was expanded
        del fringe[target]





# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
