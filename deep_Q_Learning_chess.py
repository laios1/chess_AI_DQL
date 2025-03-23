import numpy as np 
import json
import copy 



###################################################Neural Network#################################################################
def print_graphe_graphviz(SSize,w,b,X,results) : 
    to_print = "digraph G {\nnode [shape=record,width=.1,height=.1];\n"
    k = 0
    to_print = to_print + "subgraph cluster0 {\n"
    for i in range(len(X)):
        to_print = to_print + """node%d [label = "{<n>%d|%.2f|a<p>}"];\n""" % (k,i,X[i])
        k += 1
    to_print = to_print + "}\n"

    for i in range(len(SSize)-1) :
        to_print = to_print + "subgraph cluster%d {\n" % (i+1)
        for j in range(SSize[i+1]):
            to_print = to_print + """node%d [label = "{<n>%d/%d|%.2f|%.2f|a<p>}"];\n""" % (k,i,j,b[i][j],results[i][j])
            k += 1
        to_print = to_print + "}\n"
    
    totSSize = 0
    for i in range(len(SSize)-1) :
        for j in range(SSize[i]):
            for l in range(SSize[i+1]):
                to_print = to_print + """node%d -> node%d [label = %.2f];\n""" % (j+totSSize,l+totSSize+SSize[i],w[i][j][l])
        totSSize += SSize[i]
    to_print = to_print+ "}"
    return to_print

class neural_network(object):
    """
    ### crée un réseau de neurone : 

    `SSize` = [#input_layer, #hidden_layer_1,..., #hidden_layer_n, #output_layer]

    /!\ peut ne pas avoir d'hidden layer

    `LR`(Learning rate) ; LR = 1 par default 

    `act` (activation function) ; act = "sigmoid" par defaut ; (pour l'instant seulement sigmoid and ReLU)
    """
    def __init__(self,SSize,LR = 1,act = "sigmoid"):
        self.Size = SSize
        self.ActivationType = act

        self.w = [] #les poids/ taille : (nombre_de_couches-1)*(taille de la couche i)*(taille de la couche i+1)
        self.bias = [] #les correctifs d'érreurs / taille : (nb_couches-1)*(taille de la couche i)
        for i in range(len(self.Size)-1) : 
            self.w.append(np.random.randn(self.Size[i],self.Size[i+1])) #on met les poids en random 
            self.bias.append(np.random.randn(self.Size[i+1])) #on assigne les correctifs de chaques neurones 
        #self.bias.append(np.random.randn(self.Size[-1])) #on assigne les correctifs de la derniere couche 
        
        
        self.learningRate = LR 
        self.preAct = [] # valeur de la preactivation (combinaison linéaire des poids et des valeurs)
        self.results = []
        self.dError  = []


    #-------------------------------fonctions d'activations--------------------------------------

    def identite(self,s):
        return np.array(s)

    def identiteDerive(self,s):
        return np.array([1 for i in s])

    def ReLU(self,s):
        return np.array([i if i >= 0 else 0.1*i for i in s ])

    def ReLUDerive(self,s):
        return np.array([1 if i >= 0 else 0.1 for i in s ])

    def sigmoid(self,s):
        return 1/(1+np.exp(-s))

    def sigmoidDerive(self,s):
        return self.sigmoid(s)*(1-self.sigmoid(s))

    def tanh(self,s):
        return (2/(1+np.exp(-2*s)))-1

    def tanhDerive(self,s):
        return 1-(self.tanh(s)*self.tanh(s))

    def activation(self,s):
        if self.ActivationType == "sigmoid" :
            return self.sigmoid(s)
        elif self.ActivationType == "ReLU" :
            return self.ReLU(s)
        elif self.ActivationType == "id" :
            return self.identite(s)
        elif self.ActivationType == "tanh" :
            return self.tanh(s)

    def activationDerive(self,s):
        if self.ActivationType == "sigmoid" :
            return self.sigmoidDerive(s)
        elif self.ActivationType == "ReLU" :
            return self.ReLUDerive(s)
        elif self.ActivationType == "id" :
            return self.identiteDerive(s)
        elif self.ActivationType == "tanh" :
            return self.tanhDerive(s)
        
    
    #-------------------evaluation de la fonction reseau de neurones en X--------------------------------
    def forward(self,X):
        #self.preAct = 0 
        self.preAct = [] # valeur de la preactivation (combinaison linéaire des poids et des valeurs)
        self.results = []

        self.preAct.append(np.dot(X,self.w[0]) + self.bias[0]) #produit matricielle 
        self.results.append(self.activation(self.preAct[0])) #activation du neurone (lissage des données)
        
        for i in range(1,len(self.Size)-2) : #on itere pour chaque neurone 
            self.preAct.append(np.dot(self.results[i-1],self.w[i]) + self.bias[i])
            self.results.append(self.activation(self.preAct[i]))
        
        self.preAct.append(np.dot(self.results[len(self.Size)-3],self.w[len(self.Size)-2]) + self.bias[len(self.Size)-2])
        self.results.append(self.preAct[len(self.Size)-2])

        return self.results[-1]


    #---------------------------Entrainement--------------------------------------

    def backward(self,X,y,o):
        """
        X : valeur d'entree 

        y : valeur attendu par le modele 

        o : valeur retourne par le model pour une entree donne
        """
        self.dError = [] #calcul de l'erreur 


        list.insert(self.dError,0, 2*(o-y))
        for i in range(1,len(self.Size)-1) : 
            list.insert(self.dError,0, np.dot(self.dError[0],self.w[-i].T) * self.activationDerive(self.preAct[-(i+1)]))

        self.w[0] -= np.dot(np.array(X,ndmin=2).T, np.array(self.dError[0],ndmin=2))*self.learningRate
        self.bias[0] -= self.dError[0]*self.learningRate
        

        for i in range(1,len(self.Size)-1) : 
            self.w[i] -= np.dot(np.array(self.results[i-1],ndmin=2).T, np.array(self.dError[i],ndmin=2))*self.learningRate
            self.bias[i] -= self.dError[i]*self.learningRate
        

    def train(self,X,y):
        o = self.forward(X)
        self.backward(X,y,o)
        return np.abs(o-y)


    def error(self,X,y):
        o = self.forward(X)
        return np.abs(o-y)


    def predict(self,xIncunue):
        print(xIncunue)
        print(tuple(self.forward(xIncunue)))



    #--------------------------sauvegarder, charger et afficher le reseau de Neurones-----------------------------
    def print_NN(self,X):
        o = self.forward(X)
        print(print_graphe_graphviz(self.Size,self.w,self.bias,X,self.results))

    def save_NN(self, name = "Neural_Network_save"):
        with open(name+'.json', 'w', encoding='utf-8') as f:
            
            w2 = [self.w[i].tolist() for i in range(len(self.w))]
            b2 = [self.bias[i].tolist() for i in range(len(self.bias))]
            activation_f = self.ActivationType
            dic = {"weight" : w2, "bias" : b2,"act_f" : activation_f}
            json.dump(dic, f, ensure_ascii=False, indent=4)  

    def load_NN(self,name):
        with open(name+'.json') as f:
            data_loaded = json.load(f)
            self.w = [np.array(data_loaded["weight"][i]) for i in range(len(data_loaded["weight"]))]
            self.bias = [np.array(data_loaded["bias"][i]) for i in range(len(data_loaded["bias"]))]
            self.ActivationType = data_loaded["act_f"]

    def get_best_move(self,board): 
        Q = self.forward(read_board(board))
        coup = argmax_coup(board,Q)
        return coup
    

def print_shape(array):
    for i in range(len(array)):
        print(array[i].shape)
####################################################Deep-Qlearning############################################################  
import chess
import chess.pgn
import random
import matplotlib.pyplot as plt

#############################################################IA TEST###############################################################
import chess.engine
class ChessAI:
    def __init__(self, depth):
        self.depth = depth
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish\stockfish-windows-x86-64-avx2.exe") 

    def get_best_move(self, board):
        result = self.engine.play(board, chess.engine.Limit(depth=self.depth))
        return result.move

    def close_engine(self):
        self.engine.quit()


##################################################################################################################################

def read_board(board):
    """
    nomenclature : 

    blancs : 
        rois
        reinnes 
        tours 
        fous 
        cavaliers 
        pions

    Noirs : 
        rois
        reinnes 
        tours 
        fous 
        cavaliers
        pions
    """
    entree = []
    #blancs
    #roi
    w = list(board.pieces(chess.KING,chess.WHITE))
    B = list(board.pieces(chess.KING,chess.BLACK))
    k_l = [0 for i in range(64)]
    for k in w : 
        k_l[k] = 1
    for k in B : 
        k_l[k] = -1

    entree.append(k_l)


    #reine 
    w = list(board.pieces(chess.QUEEN,chess.WHITE))
    B = list(board.pieces(chess.QUEEN,chess.BLACK))
    l = [0 for i in range(64)]
    for q in B : 
        l[q] = -1
    for q in w : 
        l[q] = 1

    entree.append(l)

    #tours 
    w = list(board.pieces(chess.ROOK,chess.WHITE))
    B = list(board.pieces(chess.ROOK,chess.BLACK))
    l = [0 for i in range(64)]
    for r in B : 
        l[r] = -1
    for r in w : 
        l[r] = 1

    entree.append(l)

    #fous 
    w = list(board.pieces(chess.BISHOP,chess.WHITE))
    B = list(board.pieces(chess.BISHOP,chess.BLACK))
    l = [0 for i in range(64)]
    for b in B : 
        l[b] = -1
    for b in w : 
        l[b] = 1

    entree.append(l)

    #cavalier 
    w = list(board.pieces(chess.KNIGHT,chess.WHITE))
    B = list(board.pieces(chess.KNIGHT,chess.BLACK))
    l = [0 for i in range(64)]
    for kn in B : 
        l[kn] = -1
    for kn in w : 
        l[kn] = 1

    entree.append(l)

    #pions
    w = list(board.pieces(chess.PAWN,chess.WHITE))
    B = list(board.pieces(chess.PAWN,chess.BLACK))
    l = [0 for i in range(64)]
    for p in B : 
        l[p] = -1
    for p in w : 
        l[p] = 1

    entree.append(l)

    #on lui donne aussi c'est au tours de qui 
    #if board.turn == chess.WHITE :
    #    entree.append([1,0])
    #else :
    #    entree.append([0,1])

    return sum(entree,[])

"""
#TODO : change 
def uci_to_int(uci):
    a = ord('a')
    un = ord('1')
    int1 = (ord(uci[0])-a) + (ord(uci[1])-un)*8
    int2 = (ord(uci[3])-a) + (ord(uci[4])-un)*8
    return int1,int2

#TODO : change 
def argmax_coup_uci(board,vecteur_coup) -> str: 
    max_coup = ""
    val_max_coup = -2

    for coup in board.legal_moves :
        coup_depart, coup_arrive = uci_to_int(coup.uci())
        if val_max_coup < vecteur_coup[coup_depart] + vecteur_coup[coup_arrive+64] :
            val_max_coup = vecteur_coup[coup_depart] + vecteur_coup[coup_arrive+64]
            max_coup = coup
    
    return max_coup
"""
def uci_to_int(uci):
    a = ord('a')
    un = ord('1')
    int_depart = (ord(uci[0])-a)*8 + (ord(uci[1])-un)
    int_arrive = (ord(uci[2])-a)*8 + (ord(uci[3])-un)
    return (int_depart,int_arrive+64)


def argmax_coup(board,vecteur_coup):
    #max_coup = ""
    val_max_coup = float("-inf")
    for coup in board.legal_moves :
        coup_index = uci_to_int(coup.uci())
        if val_max_coup < (vecteur_coup[coup_index[0]]+vecteur_coup[coup_index[1]])/2 :
            val_max_coup = (vecteur_coup[coup_index[0]]+vecteur_coup[coup_index[1]])/2
            max_coup = coup
    return max_coup


def is_check_mate(board):
    # Check if the game is over due to checkmate or stalemate
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return "b"  # Black won
        else:
            return "w"  # White won
    elif board.is_stalemate():
        return "draw"  # It's a draw due to stalemate
    
    # Check if the game is over due to insufficient material
    if board.is_insufficient_material():
        return "draw"  # It's a draw due to insufficient material
    
    # Check if the game is over due to the fifty-move rule
    if board.can_claim_fifty_moves():
        return "draw"  # It's a draw due to the fifty-move rule
    
    # Check if the game is over due to threefold repetition
    if board.can_claim_threefold_repetition():
        return "draw"  # It's a draw due to threefold repetition
    
    # If none of the above conditions are met, the game is still ongoing
    return None



def random_coup(board):
    return random.choice(list(board.legal_moves))


def max_coup(vecteur_coup,possible_moves):
    val_max_coup = float("-inf")
    for coup in possible_moves :
        coup_index = uci_to_int(coup.uci())
        if val_max_coup < (vecteur_coup[coup_index[0]]+vecteur_coup[coup_index[1]])/2 :
            val_max_coup = (vecteur_coup[coup_index[0]]+vecteur_coup[coup_index[1]])/2
    return val_max_coup


def set_reward(pre,coup1,coup2):
    reward = 0 
    pos_arrive =  pre.copy()
    pos_arrive.push(coup1)
    if is_check_mate(pos_arrive) == "w" :    
        return 100
    elif is_check_mate(pos_arrive) == "b" :
        return -100
    elif is_check_mate(pos_arrive) == "draw" :
        return -10
    else : 
        if pos_arrive.is_check() : 
            reward += 3
        if pre.is_en_passant(coup1):
            reward += 1
        else:
            piece = pre.piece_type_at(coup1.to_square)

            if piece == chess.QUEEN:
                reward += 9
            elif piece == chess.ROOK:
                reward += 5
            elif piece == chess.BISHOP or piece == chess.KNIGHT:
                reward += 3
            elif piece == chess.PAWN:
                reward += 1
            else:
                reward += 0 # on a rien manger


    pos_arrive.push(coup2)
    if is_check_mate(pos_arrive) == "w" :    
        return 100
    elif is_check_mate(pos_arrive) == "b" :
        return -100
    elif is_check_mate(pos_arrive) == "draw" :
        return -10
    else : 
        if pos_arrive.is_check() : 
            reward += -3

        if pre.is_en_passant(coup2):
            reward = -1
        else:
            piece = pre.piece_type_at(coup2.to_square)
            if piece == chess.QUEEN:
                reward += -9
            elif piece == chess.ROOK:
                reward += -5
            elif piece == chess.BISHOP or piece == chess.KNIGHT:
                reward += -3
            elif piece == chess.PAWN:
                reward += -1
            else:
                reward += 0 # on a rien manger
        
        return reward
        


##########################################Q learning######################################""



def train_Q():

    wins = []
    index = []

    ai_enemie = ChessAI(10)

    
    
    gamma = 0.9
    colour = "w"
    M = 200
    N = 1500

    eps = 1

    
    
    replay_memory = []
    Q = neural_network([384,600,400,400,400,128],0.0001,"tanh") 
    # Qt = neural_network([384,600,400,200,400,4096],0.005,"tanh") 

    # Qt.w = copy.deepcopy(Q.w)
    # Qt.bias = copy.deepcopy(Q.bias)
    
    board = chess.Board()
    print("start")
    for i in range(N):  
        jj = 0
        #game = chess.pgn.Game()
        #node = game
        print("tour : n°",i)
        while is_check_mate(board) == None :
            jj += 1 
            #print(jj)

            #joue un coup et s'entraine 
            eps = (i+1)**(-1)

            # on choisit un coup et on le joue 
            pre = board.copy()

            if random.random() <= eps : #joue un coup avec une probabilite eps de faire random 
                coup1 = random_coup(board)
            else : 
                list_actions = Q.forward(read_board(pre))
                coup1 = argmax_coup(board,list_actions)

            #node = node.add_main_variation(coup)
            board.push(coup1)


            #Tour de l'autre
            coup2 = None
            if is_check_mate(board) == None : 
                coup2 =  ai_enemie.get_best_move(board) #random_coup(board) 
                #node = node.add_main_variation(best_move_opponent)
                board.push(coup2)

            r = set_reward(pre,coup1,coup2)
            is_win = (is_check_mate(board) != None)
            possible_moves = list(board.legal_moves) if not is_win else []

            replay_memory.append((read_board(pre),coup1,r,read_board(board),is_win,possible_moves)) #on met le coup dans la liste des replay


            #Entrainement 
            train_sample = random.sample(replay_memory,min(10,len(replay_memory))) 

            for pre, coup, r, post, is_win, possible_moves in train_sample :  
                #y = set_reward(Q,prej,aj,gamma)
                if is_win :
                    y = r
                else :
                    vecteur_coup = Q.forward(post)
                    y = gamma*max_coup(vecteur_coup,possible_moves)+r


                o = Q.forward(pre)
                vy = o.copy()
                index_coup = uci_to_int(coup.uci())
                vy[index_coup[0]] = y #crée le vecteur attendue par le reseau de neurone
                vy[index_coup[1]] = y

                Q.backward(pre,vy,o)

            

            #print(board,"\n") 

        # Qt.w = copy.deepcopy(Q.w)
        # Qt.bias = copy.deepcopy(Q.bias)
        #print(board)   
        index.append(i)
        if is_check_mate(board) == 'w': 
            wins.append(1)
        elif is_check_mate(board) == 'b' :
            wins.append(-1)
        else : 
            wins.append(0)
            

        print(is_check_mate(board))
        print(board.fen()) 
        board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        print("fin")

        #on vide la memoire
        if len(replay_memory) > M : 
            replay_memory.pop(0)

    ai_enemie.close_engine()
    
    plt.scatter(index,wins)
    plt.show()

    loses = [-w for w in wins]
    plt.scatter(index,loses)
    plt.show()
    # Output the PGN to a file
    # pgn_file_path = "game.pgn"
    # with open(pgn_file_path, "w") as pgn_file:
    #     pgn_file.write(str(game))
    #     print(f"PGN file '{pgn_file_path}' generated successfully!")

    Q.save_NN("ChessAI")


    


def test_Q(ai_1,ai_2,N):
    score = [0,0,0] #win , lose , draw 
    board = chess.Board()

    for i in range(N): 
        #pour les test 
        # if i %10 == 0:  
        #     print(i)
        while True :
            if  is_check_mate(board) == None :
                best_move = ai_1(board)
                board.push(best_move)
            else :
                break
            #print(board)
            #print()
            if  is_check_mate(board) == None :
                best_move_opponent = ai_2(board)
                board.push(best_move_opponent)
            else :
                break
            #print(board)
            #print("\n--\n")

        if is_check_mate(board) == "w":
            score[0] = score[0]+1 
        elif is_check_mate(board) == "b":
            score[1] = score[1]+1 
        elif is_check_mate(board) == "draw":
            score[2] = score[2]+1
        board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    return score




if __name__ == '__main__' :

    print("start")
    Q = neural_network([384,600,400,400,400,128],0.0001,"sigmoid") 
    #Q.load_NN("machin_truc")
    #score = test_Q(Q.get_best_move,random_coup,1000)
    #print(score)
    
    #board = chess.Board()
    #print(len(read_board(board)))
    train_Q()


"""
Problemes : 


"""






























################################################## main #############################################################


# if __name__ == '__main__':
#     NN = neural_network([2,5,1],0.01,"ReLU")
#     print(NN.bias[0])
#     print('\n')
#     # print_shape(NN.bias)
#     print("before :",NN.forward([1,0]))
#     print("\n")
#     for i in range(1000):
#         NN.train([0,0],[1])
#         NN.train([0,1],[0])
#         NN.train([1,0],[0])
#         NN.train([1,1],[1])
        
#         if i % 10000 == 0 :
#             print("during :",NN.forward([0,0]))
#             print("during :",NN.forward([0,1]))
#             print("during :",NN.forward([1,0]))
#             print("during :",NN.forward([1,1]))
#             print(i/1000)

#     print("after :",NN.forward([0,0]))
#     print("after :",NN.forward([0,1]))
#     print("after :",NN.forward([1,0]))
#     print("after :",NN.forward([1,1]))
#     print("\n") 
#     """
#     #NN.print_NN([0,0])
#     #NN.print_NN([1,1])

#     x_entree = np.array(([0,1.5],[2,1],[4,1.5],[3.7,1.3],[3.5,0.5],[2,0.5],[5.5,1],[1,1],[4,2.4]),dtype=float)
#     y = np.array(([1,0],[0,1],[1,0],[0,1],[1,0],[0,1], [1,0],[0,1]),dtype=float) #1 rouge et 0 bleu 

#     # print(np.amax(x_entree,axis=0))
#     # print(np.amax(x_entree,axis=1))
#     # print(x_entree)
#     # print(x_entree/np.amax(x_entree,axis=0))
#     # print(x_entree/np.amax(x_entree.T,axis=1))


#     x_entree = x_entree/np.amax(x_entree,axis=0)#normalise les entrée pour qu'elles soient entre 0 et 1 

#     X = np.split(x_entree,[8])[0] # X toutes les données
#     xOnsepa = x_entree[8] #truc a trouver 

#     NN = neural_network([2,5,2])
#     print(NN.forward(X))

#     #for i in range(10000):
#     #    NN.train(X,y)

#     # print("# "+ str(i) + "\n")
#     # print("valeur d'entrées : \n" + str(x_entree))
#     # print("eortie actuelle : \n" + str(y))
#     # print("sortie predite : \n" + str(np.matrix.round(NN.forward(x_entree),2)))

#     # print("\n")

#     # print("\n")

#     #NN.predict(xOnsepa)
#     #print("\n")
#     #print(NN.w)
#     #print(X)
#     """



