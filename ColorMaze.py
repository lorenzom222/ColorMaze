import random

import cv2
import numpy as np



# Make a vertex class






class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.color = "white"
        self.parent = None
        self.visited = False
        # Distance is infinity
        self.distance = float("inf")
        self.in_edges = []
        self.out_edges = []


    def setColor(self, color):
        self.color = color

    def getColor(self):
        return self.color

    def setParent(self, parent):
        self.parent = parent

    def getParent(self):
        return self.parent

    def setVisited(self, visited):
        self.visited = visited

    def getVisited(self):
        return self.visited

    def getX(self):
        return self.x

    def getY(self):
        return self.y
    
    def setDistance(self, distance):
        self.distance = distance

    def getDistance(self):
        return self.distance
    
    def addInEdge(self, edge):
        self.in_edges.append(edge)

    def addOutEdge(self, edge):
        self.out_edges.append(edge)
    


class Edge:
    # Make an edge class for undirected graph
    def __init__(self, v1, v2, color=None):
        self.v1 = v1
        self.v2 = v2
        self.color = None

    def getColor(self):
        return self.color

    def getV1(self):
        return self.v1

    def getV2(self):
        return self.v2
    
    # get either vertex



class Graph:

    def __init__(self):
        self.vertices = []
        self.edges = []

    def addVertex(self, vertex):
        self.vertices.append(vertex)

    def addEdge(self, edge):
        self.edges.append(edge)

    def getVertices(self):
        return self.vertices

    def getEdges(self):
        return self.edges

    def getNeighbors(self, vertex):
        neighbors = []
        for edge in self.edges:
            if edge.getV1() == vertex:
                neighbors.append(edge.getV2())
            elif edge.getV2() == vertex:
                neighbors.append(edge.getV1())
        return neighbors

    def getEdge(self, v1, v2):
        for edge in self.edges:
            if edge.getV1() == v1 and edge.getV2() == v2:
                return edge
            elif edge.getV1() == v2 and edge.getV2() == v1:
                return edge
        return None

    def getVertex(self, x, y):
        for vertex in self.vertices:
            if vertex.getX() == x and vertex.getY() == y:
                return vertex
        return None

    def getEdgeColor(self, v1, v2):
        for edge in self.edges:
            if edge.getV1() == v1 and edge.getV2() == v2:
                return edge.getColor()
            elif edge.getV1() == v2 and edge.getV2() == v1:
                return edge.getColor()
        return None

    def setEdgeColor(self, v1, v2, color):
        for edge in self.edges:
            if edge.getV1() == v1 and edge.getV2() == v2:
                edge.color = color
            elif edge.getV1() == v2 and edge.getV2() == v1:
                edge.color = color

    def getVertexColor(self, vertex):
        return vertex.getColor()

    def setVertexColor(self, vertex, color):
        vertex.setColor(color)

    def getVertexParent(self, vertex):
        return vertex.getParent()

    def setVertexParent(self, vertex, parent):
        vertex.setParent(parent)

    def getVertexVisited(self, vertex):
        return vertex.getVisited()

    def setVertexVisited(self, vertex, visited):
        vertex.setVisited(visited)

    def getVertexX(self, vertex):
        return vertex.getX()
    
    def getVertexY(self, vertex):
        return vertex.getY()
    
    def getVertexXY(self, vertex):

        return vertex.getX(), vertex.getY()
    
    # get neighbors of a vertex
    def getNeighbors(self, vertex):
        neighbors = []
        for edge in self.edges:
            if edge.getV1() == vertex:
                neighbors.append(edge.getV2())
            elif edge.getV2() == vertex:
                neighbors.append(edge.getV1())
        return neighbors
    
    def getEdgesToNeighbors(self, vertex):
        edges = []
        for edge in self.edges:
            if edge.getV1() == vertex:
                edges.append(edge)
            elif edge.getV2() == vertex:
                edges.append(edge)
        return edges
    
# def createVertices(mazeSize):
#     # Create the vertices
#     vertices = []
#     for i in range(mazeSize):
#         for j in range(mazeSize):
#             vertices.append(Vertex(i, j))
#     return vertices

# def createEdges(mazeSize, vertices):
#     # Create the edges
#     # if edge touch start, color is red
#     # if edge touch end, color is blue
#     edges = []
#     for i in range(mazeSize):
#         for j in range(mazeSize):
#             if i < mazeSize - 1: # this goes down
#                 edges.append(Edge(vertices[i * mazeSize + j], vertices[(i + 1) * mazeSize + j]))
#             if j < mazeSize - 1: # this goes right
#                 edges.append(Edge(vertices[i * mazeSize + j], vertices[i * mazeSize + j + 1]))
#             if i < mazeSize - 1 and j < mazeSize - 1: # diagonal to the right
#                 edges.append(Edge(vertices[i * mazeSize + j], vertices[(i + 1) * mazeSize + j + 1]))
            
#     return edges

def getNextColor(color):
    if color == "red":
        return "yellow"
    elif color == "yellow":
        return "blue"
    elif color == "blue":
        return "red"



def BFS(graph, start, end): ## Shortest path must follow this order for each direction to be traversed
    # Create a queue for BFS
    queue = []
 
    # Mark the source node as visited and enqueue it
    start.setVisited(True)
    start.setColor("blue")
    queue.append(start)
 
    # Loop to traverse the graph using BFS
    while queue:
        # Dequeue a vertex from queue
        vertex = queue.pop(0)

        
        # becasue the start is red, the next color is yellow
        neigh = graph.getNeighbors(vertex)
        nextColor = getNextColor(vertex.getColor())
        # Get the neighbors of the vertex that have the next color in the sequence from their edge
        neighbors = [neighbor for neighbor in neigh if graph.getEdgeColor(vertex, neighbor) == nextColor]
 
        # Loop through all the neighbors
        for neighbor in neighbors:
            # If the neighbor is the destination vertex, then we have found the path
            if neighbor == end:
                # Update the color of the edge from vertex to neighbor
                graph.setEdgeColor(vertex, neighbor, getNextColor(vertex.getColor()))
                # Mark the neighbor as visited and return the path
                neighbor.setVisited(True)

                # set the parent of the end vertex
                neighbor.setParent(vertex)
                # set the color of the end vertex
                neighbor.setColor(getNextColor(vertex.getColor()))

                return True
            if(not neighbor.getVisited()):
                # Mark the neighbor as visited
                neighbor.setVisited(True)

                # Set the color of the neighbor to color of the edge from vertex to neighbor
                neighbor.setColor(getNextColor(vertex.getColor()))
                # Set the parent of the neighbor
                neighbor.setParent(vertex)
                # increment the distance of the neighbor
                neighbor.setDistance(vertex.getDistance() + 1)
                
                # Enqueue the neighbor
                queue.append(neighbor)

            elif neighbor.getVisited() and neighbor.getDistance() > vertex.getDistance() + 1:
                # Update the color of the edge from vertex to neighbor
                graph.setEdgeColor(vertex, neighbor, getNextColor(vertex.getColor()))
                # Set the color of the neighbor
                neighbor.setColor(getNextColor(vertex.getColor()))
                # Set the parent of the neighbor
                neighbor.setParent(vertex)
                # increment the distance of the neighbor
                neighbor.setDistance(vertex.getDistance() + 1)
                
                # Enqueue the neighbor
                queue.append(neighbor)

            else:
                # skip the neighbor
                continue

    # If we reach here, then the path does not exist
    # Raise an exception
    return False


def draw_maze(maze, cell_size=50, short_path_flag=False):
    # Create an image of the maze
    image = np.zeros((mazeSize * cell_size, mazeSize * cell_size, 3), dtype=np.uint8)
    edges = maze.getEdges()

    # Draw the edges
    drawEdges(edges,image, cell_size=50)

    # Draw the start and end points
    start = maze.getVertex(0, 0)
    end = maze.getVertex(mazeSize - 1, mazeSize - 1)
    x1, y1 = start.getX() * cell_size, start.getY() * cell_size
    x2, y2 = end.getX() * cell_size, end.getY() * cell_size
    cv2.circle(image, (x1 + cell_size // 2, y1 + cell_size // 2), cell_size // 4, (0, 255, 0), -1)
    cv2.circle(image, (x2 + cell_size // 2, y2 + cell_size // 2), cell_size // 4, (255, 255, 0), -1)

    drawVertices(maze.getVertices(),image, cell_size=50)

    if short_path_flag:
        drawShortestPath(maze, image, cell_size=50)

    return image

def drawEdges(edges,image, cell_size=50):
        # Draw the edges
    for edge in edges:
        v1 = edge.getV1()
        v2 = edge.getV2()
        x1, y1 = v1.getX() * cell_size, v1.getY() * cell_size
        x2, y2 = v2.getX() * cell_size, v2.getY() * cell_size
        color = edge.getColor()
        if color == 'red':
            cv2.line(image, (x1+ cell_size // 2, y1+ cell_size // 2), (x2+ cell_size // 2, y2+ cell_size // 2), (0, 0, 255), 5)
        elif color == 'yellow':
            cv2.line(image, (x1+ cell_size // 2, y1+ cell_size // 2), (x2+ cell_size // 2, y2+ cell_size // 2), (0, 255, 255), 5)
        elif color == 'blue':
            cv2.line(image, (x1+ cell_size // 2, y1+ cell_size // 2), (x2+ cell_size // 2, y2+ cell_size // 2), (255, 0, 0), 5)



def drawVertices(vertices,image, cell_size=50):
    # Draw the vertices as white circles
    for vertex in vertices:
        x, y = vertex.getX() * cell_size, vertex.getY() * cell_size
        color = vertex.getColor()

        if color == 'white':
            cv2.circle(image, (x + cell_size // 2, y + cell_size // 2), cell_size // 4, (255, 255, 255), -1)
        elif color == 'red':
            cv2.circle(image, (x + cell_size // 2, y + cell_size // 2), cell_size // 4, (0, 0, 255), -1)
        elif color == 'yellow':
            cv2.circle(image, (x + cell_size // 2, y + cell_size // 2), cell_size // 4, (0, 255, 255), -1)
        elif color == 'blue':
            cv2.circle(image, (x + cell_size // 2, y + cell_size // 2), cell_size // 4, (255, 0, 0), -1)

# After I run the BFS, I want to draw the shortest path from the start to the end
def getShortestPath(start, end):
    path = []
    vertex = end
    while vertex != start:
        path.append(vertex)
        vertex = vertex.getParent()
    path.append(start)
    return list(reversed(path))


def drawShortestPath(maze, image, cell_size=50):
    # Draw the shortest path
    path = getShortestPath(maze.getVertex(0, 0), maze.getVertex(mazeSize - 1, mazeSize - 1))
    for i in range(len(path) - 1):
        v1 = path[i]
        v2 = path[i + 1]
        x1, y1 = v1.getX() * cell_size, v1.getY() * cell_size
        x2, y2 = v2.getX() * cell_size, v2.getY() * cell_size
        # Let line color be green
        cv2.line(image, (x1 + cell_size // 2, y1 + cell_size // 2), (x2 + cell_size // 2, y2 + cell_size // 2), (0, 255, 0), 5)



    

if __name__ == '__main__':
    # Build the maze and draw it
    mazeSize = 5
    maze = Graph()
    for i in range(mazeSize):
        for j in range(mazeSize):
            maze.addVertex(Vertex(i, j))
    # The code to add edges and set colors goes here...
        # add edges to the maze: horizontal and vertical and diagonal
    # set the colors
    # Edges touch start must be red, end must be blue
    colors = ['red', 'yellow', 'blue']
    # I want to that there is an equal chance of getting red, yellow, or blue
    for i in range(mazeSize):
        for j in range(mazeSize):
            # color = random.choice(colors)
            # add edges to the maze: horizontal and vertical and diagonal
            if i < mazeSize - 1: # horizontal
                    maze.addEdge(Edge(maze.getVertex(i, j), maze.getVertex(i + 1, j)))
                    maze.setEdgeColor(maze.getVertex(i, j), maze.getVertex(i + 1, j), random.choice(colors))
            if j < mazeSize - 1: # vertical

                    maze.addEdge(Edge(maze.getVertex(i, j), maze.getVertex(i, j + 1)))
                    maze.setEdgeColor(maze.getVertex(i, j), maze.getVertex(i, j + 1), random.choice(colors))
            if i < mazeSize - 1 and j < mazeSize - 1: # diagonal to the right

                    maze.addEdge(Edge(maze.getVertex(i, j), maze.getVertex(i + 1, j + 1)))
                    maze.setEdgeColor(maze.getVertex(i, j), maze.getVertex(i + 1, j + 1), random.choice(colors))
            
            if(maze.getVertex(i, j) == maze.getVertex(mazeSize - 1, mazeSize - 1)):
                lastVertex = maze.getVertex(i, j)
                final_neighbors = maze.getNeighbors(lastVertex)
                for neighbor in final_neighbors:
                    maze.setEdgeColor(lastVertex, neighbor, 'blue')
            elif(maze.getVertex(i, j) == maze.getVertex(0, 0)):
                firstVertex = maze.getVertex(i, j)
                first_neighbors = maze.getNeighbors(firstVertex)
                for neighbor in first_neighbors:
                    maze.setEdgeColor(firstVertex, neighbor, 'red')

    # set the start and end points
    start = maze.getVertex(0, 0)
    end = maze.getVertex(mazeSize - 1, mazeSize - 1)
    

    image = draw_maze(maze)

    # Show the image and wait for a key press
    cv2.imshow('Maze', image)
    cv2.waitKey(0)



    # Solve the maze with BFS

    BFS(maze, start, end)   

    # draw_maze(maze)
    image = draw_maze(maze, cell_size=50, short_path_flag=True)
    cv2.imshow('BFS Solution', image)
    cv2.waitKey(0)

 

    

