import matplotlib.pyplot as plt
import psutil
import pygame
import numpy as np
import random
import time
from itertools import product

def Draw_Stones_and_Board(mat):
#Given a 2D numpy array, plot the corresponding stones on a 15 by 15 board. You may find the following function handy,
#ax.set_xlim(), ax.set_ylim(), ax.set_xticks(), ax.set_yticks(), ax.set_xticklables(), ax.set_yticklables()
#Circle=plt.Circle(), ax.add_artist()
    fig,ax=plt.subplots(figsize=(16, 16))
    ax.set_xticks(np.arange(1, 16, 1))
    ax.set_yticks(np.arange(1, 16, 1))
    ax.set_xlim(0,16)
    ax.set_ylim(0,16)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    c = {1:"red", -1:"black"}
    circles = [plt.Circle((x+1,y+1), 0.4,color=c[mat[x][y]]) if mat[x][y] != 0 else None for x in range(len(mat)) for y in range(len(mat[0]))]
    list(map(ax.add_artist, list(filter(lambda x: x is not None, circles))))
    plt.grid(linestyle='-', linewidth='1', color='black')
    plt.show()

def update_by_man(event,mat):
    """
    This function detects the mouse click on the game window. Update the state matrix of the game.
    input:
        event:pygame event, which are either quit or mouse click)
        mat: 2D matrix represents the state of the game
    output:
        mat: updated matrix
    """

    done=False
    if event.type==pygame.QUIT:
        done=True
    if event.type==pygame.MOUSEBUTTONDOWN:
        (x,y)=event.pos
        row = round((y - 40) / 40)
        col = round((x - 40) / 40)
        mat[row][col]=1
        print('mouse Click')
    return mat, done

def draw_board(screen):
    """
    This function draws the board with lines.
    input: game windows
    output: none
    """
    black_color = [0, 0, 0]
    board_color = [ 241, 196, 15 ]
    screen.fill(board_color)
    for h in range(1, 16):
        pygame.draw.line(screen, black_color,[40, h * 40], [600, h * 40], 1)
        pygame.draw.line(screen, black_color, [40*h, 40], [40*h, 600], 1)

def draw_stone(screen, mat):
    """
    This functions draws the stones according to the mat. It draws a black circle for matrix element 1(human),
    it draws a white circle for matrix element -1 (computer)
    input:
        screen: game window, onto which the stones are drawn
        mat: 2D matrix representing the game state
    output:
        none
    """
    black_color = [0, 0, 0]
    white_color = [255, 255, 255]
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i][j]==1:
                pos = [40 * (j + 1), 40 * (i + 1)]
                pygame.draw.circle(screen, black_color, pos, 18,0)
            elif mat[i][j]==-1:
                pos = [40 * (j + 1), 40 * (i + 1)]
                pygame.draw.circle(screen, white_color, pos, 18,0)


def render(screen, mat):
    """
    Draw the updated game with lines and stones using function draw_board and draw_stone
    input:
        screen: game window, onto which the stones are drawn
        mat: 2D matrix representing the game state
    output:
        none
    """
    draw_board(screen)
    draw_stone(screen, mat)
    pygame.display.update()

exploration_param = np.sqrt(2)

class Node():
    def __init__(self, state, parent, player):
        self.state = state
        self.is_expended = False
        self.parent = parent
        self.children = {}
        self.total_value = 0
        self.visited_number = 0
        self.player = player
        self.unvisited_children = 5


def move(mat,player):
    mat_temp = mat.copy()
    while True:
        random.seed(time.time())
        x = random.randrange(0, 14)
        y = random.randrange(0, 14)
        if mat_temp[x][y] !=0:
            continue
        else:
            mat_temp[x][y] = player
            break
    return mat_temp

def check_for_done(mat):
    """
    please write your own code testing if the game is over. Return a boolean variable done. If one of the players wins
    or the tie happens, return True. Otherwise return False. Print a message about the result of the game.
    input:
        2D matrix representing the state of the game
    output:
        none
    """
    offset = [-2, -1, 0, 1, 2]
    m,n = mat.shape

    def is_win(a, b):
        return a == b and b!=0

    def check_one(x, y):
        return\
        all(0 <= a[1] < m and is_win(mat[a[0]][a[1]], mat[x][y]) for a in list(map(lambda o: (x, y+o), offset))) or\
        all(0 <= a[0] < n and is_win(mat[a[0]][a[1]], mat[x][y]) for a in list(map(lambda o: (x+o, y), offset))) or\
        all(0 <= a[0] < n and 0 <= a[1] < m and is_win(mat[a[0]][a[1]], mat[x][y]) for a in list(map(lambda o: (x+o, y+o), offset))) or\
        all(0 <= a[0] < n and 0 <= a[1] < m and is_win(mat[a[0]][a[1]], mat[x][y]) for a in list(map(lambda o: (x+o, y-o), offset)))
    return any(check_one(n[0],n[1]) for n in list(product(range(n), range(m))))

def update_by_pc(mat):
    global mcts_root
    """
    This is the core of the game. Write your code to give the computer the intelligence to play a Five-in-a-Row game
    with a human
    input:
        2D matrix representing the state of the game.
    output:
        2D matrix representing the updated state of the game.
    """
    return monte_carlo_tree_search(mcts_root).state

def monte_carlo_tree_search(root):
    time_start = time.time()
    counter = 0
    while True:
        if time.time() - time_start > 3:
            break
        leaf = traverse(root) # leaf = unvisited node
        if not leaf:
            break
        new_leaf, simulation_result = rollout(leaf)
        backpropagate(new_leaf, simulation_result)
        counter+=1
    print(f'This steps run for {counter} time')
    return best_child(root)

# For the traverse function, to avoid using up too much time or resources, you may start considering only
# a subset of children (e.g 5 children). Increase this number or by choosing this subset smartly later.
def traverse(node):
    # The program traverses from the root node to leaf node by choosing the one with best reward
    while node.is_expended:
        node = best_uct(node)
    # Then it visited the siblings of the leaf node
    unvisited_sibling = pick_unvisited_node(node.parent)
    if unvisited_sibling != None:
        return unvisited_sibling
    # If all siblings are visted, the program visits the children of the leaf node
    unvisited_children = pick_unvisited_node(node)
    if unvisited_children != None:
        return unvisited_children
    # If all the childrens are visited, well, nothing to do for this round
    return None


def rollout(node):
    # Expand the tree by adding new state
    node.is_expanded = True
    mat_new = move(node.state, node.player * -1)
    mat_game = np.copy(mat_new)

    # Create new node as the children
    node.children[str(mat_new)] = Node(mat_new, parent = node, player = node.player * -1)
    player_start = node.children[str(mat_new)].player

    # Start roll out from the tree, will stop when the game ends
    while True:
        # If not ended, random move will be made
        # TODO: use dictionary for memoization
        mat_game = move(mat_game, player_start*-1)
        if check_for_done(mat_game) == True:
            break
        else:
            # Switch player to another
            player_start = player_start*-1
    return node.children[str(mat_new)], player_start

def backpropagate(node, result):
    if is_root(node):
        return
    update_stats(node, result)
    backpropagate(node.parent, result)

def pick_unvisited_node(node):
    if not node:
        return None
    if node.unvisited_children < 1:
        return None
    for n in node.children.values():
        if n.visited_number == 0:
            return n
    # Create new node as the children
    mat_new = move(node.state, node.player * -1)
    while str(mat_new) in node.children:
        mat_new = move(node.state, node.player * -1)
    new_node = Node(mat_new, parent = node, player = node.player * -1)
    node.children[str(mat_new)] = new_node
    return new_node


def is_root(node):
        return True if node.parent == None else False

def update_stats(node, result):
    if result == node.player:
        node.total_value +=1
        node.visited_number +=1
    elif result == 0.5:
        node.total_value +=0.5
        node.visited_number +=1
    else:
        node.visited_number +=1
    if node.parent:
        node.parent.unvisited_children -= 1

def best_child(node):
    return max(node.children.values(), key = lambda x : x.visited_number)

def uct_score(node):
        return node.total_value/node.visited_number + (exploration_param * node.parent.visited_number/node.visited_number)

def best_uct(node):
    return max(node.children.values(), key = lambda x : x.uct_score())

def update_root(mcts_root, mat):
    if str(mat) in mcts_root.children:
        new_root = mcts_root.children[str(mat)]
        new_root.parent = None
        return new_root
    else:
        return Node(mat, parent = None, player = 1)

mcts_root = None
def main():
    pygame.init()
    screen=pygame.display.set_mode((640,640))
    pygame.display.set_caption('Five-in-a-Row')
    done=False
    mat=np.zeros((15,15))
    global mcts_root
    mcts_root = Node(mat, parent = None, player = 1)
    print(mcts_root)

    while not done:
        for event in pygame.event.get():
            mat, done=update_by_man(event, mat)
            render(screen, mat)
            mcts_root = update_root(mcts_root, mat)
            if event.type==pygame.MOUSEBUTTONDOWN:
                done=check_for_done(mat)
                mat=update_by_pc(mat)
                render(screen, mat)
                done=check_for_done(mat)
                print('CPU Usage:', psutil.cpu_percent())
    pygame.quit()

if __name__ == '__main__':
    main()
