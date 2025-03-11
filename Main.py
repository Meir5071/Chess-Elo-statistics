import re
import sys
import pickle
import math
import colorsys
import numpy as np
import scipy
import cv2
from scipy.ndimage import gaussian_filter
from PIL import Image
from chess.pgn import read_game
import codecs
from time import time

#import os
sys.path.insert(1, './doc')

def init_kernel(sigma:int):
    tmp=init_i_j(sigma)
    i=tmp[0]
    j=tmp[1]
    kernel=np.exp((1/(2*math.pi*sigma**2))*((i**2+i**2)/(-2*sigma**2)))
    return kernel
def init_i_j(sigma:int):
    j_arr=np.array(range(3*sigma))
    i_arr=(j_arr*0)+1
    i=i_arr*0
    j=j_arr*1
    for k in range(3*sigma-1):
        i=np.concatenate((i,i_arr*(k+1)))
        j=np.concatenate((j,j_arr))
    return [i.reshape(3*sigma, 3*sigma), j.reshape(3*sigma, 3*sigma)]
def pars(file: str) -> list:
    tmp = open(file, "r")
    file_data = tmp.readlines()
    tmp.close()
    clean_file = clean(file_data)
    parssed_file = []
    i = 0
    while i < len(clean_file) - 2:
        w = 0
        b = 0
        r = True
        is_w_elo  = True
        is_b_elo  = True
        if clean_file[i + 1][0] == "w":
            w = 1
            is_w_elo = clean_file[i + 1] != "w:"
        if clean_file[i + 2][0] == "b":
            b = 1
            is_b_elo = clean_file[i + 2] != "b:"
        if clean_file[i] == "r:r":
            r = False
        if (((is_b_elo) and (is_w_elo)) and ((w) and (b))) and r:
            parssed_file = parssed_file + [[int(clean_file[i][2:len(clean_file[i])]), int(clean_file[i + 1][2:len(clean_file[i + 1])]), int(clean_file[i + 2][2:len(clean_file[i + 2])])]]
        i = i + 1 + w + b
    return parssed_file
def count_wins_all_table2(file_in: str, file_out: str, min_b_elo:int, max_b_elo:int, min_w_elo:int, max_w_elo:int) -> None:
    file = codecs.open(file_in+".pgn", "r",  encoding="utf8", errors="ignore")
    table=init_table([max_w_elo-min_w_elo,max_b_elo-min_b_elo])
    preveios_lines=[]
    lines=0
    games=0
    start_time=time()
    while True:
        line=file.readline()
        lines+=1
        if line == "":
            file.close()
            file_o = open(file_out + ".p2pgn", "wb")
            pickle.dump(table, file_o)
            file_o.close()
            #print(table)
            return
        if int(lines/1000000)*1000000==lines:
            file_o = open(file_out + ".p2pgn", "wb")
            pickle.dump(table, file_o)
            file_o.close()
            #print(table)
        if int(lines/100000)*100000==lines:
            file_doc=open("doc2.txt","a")
            file_doc.write("line number: " + str(lines) + "\n")
            file_doc.write("current time: " + str(time() - start_time) + "\n")
            file_doc.write("number of games: " + str(games) + "\n")
            file_doc.close()
        parssed_line = clean_line(line)
        if not(parssed_line==""):
            if parssed_line[0]=="r":
                if len(preveios_lines)==3:
                    r = True
                    is_w_elo  = True
                    is_b_elo  = True
                    if preveios_lines[1][0] == "w":
                        is_w_elo = preveios_lines[1] != "w:"
                    if preveios_lines[2][0] == "b":
                        is_b_elo = preveios_lines[2] != "b:"
                    if preveios_lines[0] == "r:r":
                        r = False
                    if ((is_b_elo) and (is_w_elo)) and r:
                        adj = [int(preveios_lines[1][2:len(preveios_lines[1])]), int(preveios_lines[2][2:len(preveios_lines[2])]),int(preveios_lines[0][2:len(preveios_lines[0])])]
                        if ((adj[0]>min_w_elo) and (adj[0]<max_w_elo)) and ((adj[1]>min_b_elo) and (adj[1]<max_b_elo)):
                            games+=1
                            adj[0]=adj[0]-min_w_elo
                            adj[1]=adj[1]-min_b_elo
                            place=table[adj[0]][adj[1]]
                            place[0] = place[0] + chainge_result_2(adj[2],0)
                            place[1] = place[1] + chainge_result_2(adj[2],1)
                            place[2] = place[2] + chainge_result_2(adj[2],2)
                preveios_lines=[]
            preveios_lines=preveios_lines+[parssed_line]
def count_wins_all_table2_no_doc(file_in: str, file_out: str, min_b_elo:int, max_b_elo:int, min_w_elo:int, max_w_elo:int) -> None:
    file = codecs.open(file_in+".pgn", "r",  encoding="utf8", errors="ignore")
    table=init_table([max_w_elo-min_w_elo,max_b_elo-min_b_elo])
    preveios_lines=[]
    lines=0
    games=0
    while True:
        line=file.readline()
        lines+=1
        if line == "":
            file.close()
            file_o = open(file_out + ".p2pgn", "wb")
            pickle.dump(table, file_o)
            file_o.close()
            #print(table)
            return
        parssed_line = clean_line(line)
        if not(parssed_line==""):
            if parssed_line[0]=="r":
                if len(preveios_lines)==3:
                    r = True
                    is_w_elo  = True
                    is_b_elo  = True
                    if preveios_lines[1][0] == "w":
                        is_w_elo = preveios_lines[1] != "w:"
                    if preveios_lines[2][0] == "b":
                        is_b_elo = preveios_lines[2] != "b:"
                    if preveios_lines[0] == "r:r":
                        r = False
                    if ((is_b_elo) and (is_w_elo)) and r:
                        adj = [int(preveios_lines[1][2:len(preveios_lines[1])]), int(preveios_lines[2][2:len(preveios_lines[2])]),int(preveios_lines[0][2:len(preveios_lines[0])])]
                        if ((adj[0]>min_w_elo) and (adj[0]<max_w_elo)) and ((adj[1]>min_b_elo) and (adj[1]<max_b_elo)):
                            adj[0]=adj[0]-min_w_elo
                            adj[1]=adj[1]-min_b_elo
                            place=table[adj[0]][adj[1]]
                            place[0] = place[0] + chainge_result_2(adj[2],0)
                            place[1] = place[1] + chainge_result_2(adj[2],1)
                            place[2] = place[2] + chainge_result_2(adj[2],2)
                preveios_lines=[]
                games+=1
            preveios_lines=preveios_lines+[parssed_line]
def count_wins_all_table3(file_in: str, file_out: str, min_b_elo:int, max_b_elo:int, min_w_elo:int, max_w_elo:int) -> None:
    i=0
    file = open(file_in+".pgn", "r")
    table=init_table([max_w_elo-min_w_elo,max_b_elo-min_b_elo])
    game=read_game(file)
    while game:
        i+=1
        if ('WhiteElo' in game.headers) and ('BlackElo' in game.headers):
            r=chainge_result(game.headers['Result'])
            w=game.headers['WhiteElo']
            b=game.headers['BlackElo']
            if (is_num(w) and is_num(b)) and (r != "r"):
                adj = [int(w), int(b),int(r)]
                if ((adj[0]>min_w_elo) and (adj[0]<max_w_elo)) and ((adj[1]>min_b_elo) and (adj[1]<max_b_elo)):
                    adj[0]=adj[0]-min_w_elo
                    adj[1]=adj[1]-min_b_elo
                    place=table[adj[0]][adj[1]]
                    place[0] = place[0] + chainge_result_2(adj[2],0)
                    place[1] = place[1] + chainge_result_2(adj[2],1)
                    place[2] = place[2] + chainge_result_2(adj[2],2)
        game=read_game(file)
    file.close()
    file_o = open(file_out + ".p2pgn", "wb")
    pickle.dump(table, file_o)
    file_o.close()
def is_num(num:str)-> bool:
    return (num!="")
def init_table(size:list):
    table = [[ [0,0,0] for _ in range(size[1]) ]for _ in range(size[0])]
    return np.array(table)
    #return table
def clean_line(line: str) -> list:
    new_line = ""
    if len(line)>9:
        if line[0:7]=="[Result":
            new_line = "r:" + chainge_result(line.split('"')[1])
        elif line[0:9]=="[WhiteElo":
            new_line = "w:" + line.split('"')[1]
        elif line[0:9]=="[BlackElo":
            new_line = "b:" + line.split('"')[1]
    return new_line
def clean_line_old(line: list) -> list:
    new_line = ""
    if re.search("\[Result .*\]",line):
        new_line = "r:" + chainge_result(line[9:len(line)-3])
    elif re.search("\[WhiteElo .*\]",line):
        new_line = "w:" + line[11:len(line)-3]
    elif re.search("\[BlackElo .*\]",line):
        new_line = "b:" + line[11:len(line)-3]
    return new_line
def clean_test_ver(file: list) -> list:
    clean_file = []
    for i in range(len(file)):
        line = file[i]
        if re.search("\[Result .*\]",line):
            clean_file = clean_file + ["r:" + chainge_result(line[9:len(line)-3])]
        elif re.search("\[WhiteElo .*\]",line):
            clean_file = clean_file + ["w:" + line[11:len(line)-3]]
        elif re.search("\[BlackElo .*\]",line):
            clean_file = clean_file + ["b:" + line[11:len(line)-3]]
    return clean_file
def clean(file: list) -> list:
    clean_file = []
    for line in file:
        if re.search("\[Result .*\]",line):
            clean_file = clean_file + ["r:" + chainge_result(line[9:len(line)-3])]
        elif re.search("\[WhiteElo .*\]",line):
            clean_file = clean_file + ["w:" + line[11:len(line)-3]]
        elif re.search("\[BlackElo .*\]",line):
            clean_file = clean_file + ["b:" + line[11:len(line)-3]]
    return clean_file
def chainge_result(result:str) -> str:
    if result[:3] == "1-0":
        return "1"
    elif result[:7] == "1/2-1/2":
        return "0"
    elif result[:3] == "0-1":
        return "-1"
    else:
        return "r"
def chainge_result_2(result:int,place:int) -> int:
    if (result == 1) and (place==0):
        return 1
    elif (result == 0) and (place==1):
        return 1
    elif (result == -1) and (place==2):
        return 1
    else:
        return 0
def create_table(file: str, min_b_elo:int, max_b_elo:int, min_w_elo:int, max_w_elo:int, n_of_b_steps:int, n_of_w_steps:int) -> list:
    #parssed_file = pars(file)
    tmp = open(file + ".ppgn", "rb")
    parssed_file = pickle.load(tmp)
    tmp.close()
    table = [[ [] for _ in range(n_of_b_steps) ]for _ in range(n_of_w_steps)]
    #table = [[[]]*n_of_b_steps]*n_of_w_steps
    delta_w = (max_w_elo - min_w_elo)/n_of_w_steps
    delta_b = (max_b_elo - min_b_elo)/n_of_b_steps
    for game in parssed_file:
        w_index = int((game[1] - min_w_elo)/delta_w)
        b_index = int((game[2] - min_b_elo)/delta_b)
        if ((min_w_elo < game[1]) and (game[1] < max_w_elo)) and ((min_b_elo < game[2]) and (game[2] < max_b_elo)):
            table[w_index][b_index] = table[w_index][b_index] + [game]
    return table
def count_wins(w_elo:int, b_elo:int, table:list, min_b_elo:int, max_b_elo:int, min_w_elo:int, max_w_elo:int, n_of_b_steps:int, n_of_w_steps:int, sigma:float) -> list:
    w_wins = 0
    draws = 0
    b_wins = 0
    relvant_games = find_relvant_games(w_elo, b_elo, table, min_b_elo, max_b_elo, min_w_elo, max_w_elo, n_of_b_steps, n_of_w_steps)
    for game in relvant_games:
        x = math.exp(-((game[1]-w_elo)**2 + (game[2]-b_elo)**2)/(sigma**2))
        if game[0] == 1:
            w_wins = w_wins + x
        if game[0] == 0:
            draws = draws + x
        if game[0] == -1:
            b_wins = b_wins + x
    return [w_wins, draws, b_wins]
def find_relvant_games(w_elo:int, b_elo:int, table:list, min_b_elo:int, max_b_elo:int, min_w_elo:int, max_w_elo:int, n_of_b_steps:int, n_of_w_steps:int) -> list:
    relvant_games = []
    delta_w = (max_w_elo - min_w_elo)/n_of_w_steps
    delta_b = (max_b_elo - min_b_elo)/n_of_b_steps
    place_w = int((w_elo - min_w_elo)/delta_w)
    place_b = int((b_elo - min_b_elo)/delta_b)
    for i in line_erea(place_w, n_of_w_steps):
        for j in line_erea(place_b, n_of_b_steps):
            relvant_games = relvant_games + table[i][j]
    return relvant_games
def line_erea(place:int, length:int) -> list:
    if place == 0:
        return [place, place + 1]
    elif place == length - 1:
        return [place - 1, place]
    else:
        return [place - 1, place, place + 1]
def count_wins_in_all_table(file: str, min_b_elo:int, max_b_elo:int, min_w_elo:int, max_w_elo:int, n_of_b_steps:int, n_of_w_steps:int, sigma:float) -> list:
    old_table = create_table(file, min_b_elo, max_b_elo, min_w_elo, max_w_elo, n_of_b_steps, n_of_w_steps)
    new_table = [[ [] for _ in range(max_b_elo - min_b_elo - 1) ]for _ in range(max_w_elo - min_w_elo - 1)]
    for i in range(len(new_table)):
        for j in range(len(new_table[i])):
            new_table[i][j] = count_wins(min_w_elo + i + 1, min_b_elo + j + 1, old_table, min_b_elo, max_b_elo, min_w_elo, max_w_elo, n_of_b_steps, n_of_w_steps, sigma)
    return new_table
def find_global_max(line: list) -> int:
    maximum = 0
    for entery in line:
        for result in entery:
            maximum = max(result, maximum)
    return maximum
def table_to_line(table: list) -> list:
    line = []
    for table_line in table:
        line = line + table_line
    return line
def divide_by_global_max(line:list, global_maximum: int) -> list:
    divided_line = [[] for _ in range(len(line))]
    for i in range(len(line)):
        entery = line[i]
        divided_line[i] = (int(entery[0]/(global_maximum/255)), int(entery[1]/(global_maximum/255)), int(entery[2]/(global_maximum/255)))
    return divided_line
def divide_by_local_max(line:list) -> list:
    divided_line = [[] for _ in range(len(line))]
    for i in range(len(line)):
        entery = line[i]
        if entery == [0, 0, 0]:
            divided_line[i] = (0,0,0)
        else:
            local_maximum = max(max(entery[0], entery[1]), entery[2])
            divided_line[i] = (int(entery[0]/(local_maximum/255)), int(entery[1]/(local_maximum/255)), int(entery[2]/(local_maximum/255)))
    return divided_line
def one_bit(line:list):
    new_line = [[] for _ in range(len(line))]
    for i in range(len(line)):
        entery = line[i]
        if ((entery[0] == 0) and (entery[1] == 0)) and (entery[2] == 0):
            new_line[i] = (0,0,255)
        elif entery[0] == entery[2]:
            new_line[i] = (0,255,0)
        elif entery[0] < entery[2]:
            new_line[i] = (0,0,0)
        else:
            new_line[i] = (255,255,255)
    return new_line
def HSV(line:list):
    const_s=1
    new_line = [[] for _ in range(len(line))]
    for i in range(len(line)):
        entery = line[i]
        if ((entery[0] == 0) and (entery[1] == 0)) and (entery[2] == 0):
            new_line[i] = (0,0,0)
        #elif (entery[0] == 0) and (entery[2] == 0):
        #    new_line[i] = (255,255,255)
        else:
            tmp = colorsys.hsv_to_rgb(((entery[0]+entery[1]/2)*(240/360))/(entery[0]+entery[1]+entery[2]),const_s-(const_s*entery[1])/((entery[0]+entery[1]+entery[2])),255)
            new_line[i] = (int(tmp[0]),int(tmp[1]),int(tmp[2]))
    return new_line
#def pgn_file_to_image(file_in: str, file_out: str, min_b_elo:int, max_b_elo:int, min_w_elo:int, max_w_elo:int, n_of_b_steps:int, n_of_w_steps:int) -> None:
#    table = count_wins_in_all_table(file_in, min_b_elo, max_b_elo, min_w_elo, max_w_elo, n_of_b_steps, n_of_w_steps)
#    line = table_to_line(table)
#    line = divide_by_max(line, find_max(line))
#    image = Image.new("RGB", (n_of_w_steps, n_of_b_steps))
#    image.putdata(data = line)
#    image.save(file_out + ".png")
def parssed_file_to_image(file_in: str, file_out: str, min_b_elo:int, max_b_elo:int, min_w_elo:int, max_w_elo:int, n_of_b_steps:int, n_of_w_steps:int, sigma:float, mode:str) -> None:
    #file = open(file_in + ".table", "rb")
    #table = pickle.load(file)
    #file.close()
    tmp = open(file_in + ".line", "rb")
    line = pickle.load(tmp)
    tmp.close()
    image = line_to_image(line, mode, (max_w_elo - min_w_elo - 1, max_b_elo - min_b_elo - 1))
    if image == "there is no such mode":
        print("there is no such mode " + str(mode))
        return
    #image = Image.new("RGB", (n_of_w_steps, n_of_b_steps))
    #image.putdata(data = line)
    image.save(file_out + ".png")
def table_to_image(file_in: str, file_out: str, min_b_elo:int, max_b_elo:int, min_w_elo:int, max_w_elo:int, n_of_b_steps:int, n_of_w_steps:int, sigma:float, mode:str, big_const:int) -> None:
    #file = open(file_in + ".table", "rb")
    #table = pickle.load(file)
    #file.close()
    tmp = open(file_in + ".p2pgn", "rb")
    table = pickle.load(tmp)
    tmp.close()
    #print(table==init_table([max_w_elo-min_w_elo,max_b_elo-min_b_elo]))
    #print(table.tolist()==init_table([max_w_elo-min_w_elo,max_b_elo-min_b_elo]).tolist())
    table = cv2.filter2D(table,-1,init_kernel(sigma))
    line = table.transpose(2,0,1).reshape(3,-1).transpose().tolist()
    #print(table==init_table([max_w_elo-min_w_elo,max_b_elo-min_b_elo]).tolist())
    #line = table_to_line(table)
    image = line_to_image(line, mode, (max_w_elo - min_w_elo, max_b_elo - min_b_elo))
    if image == "there is no such mode":
        print("there is no such mode " + str(mode))
        return
    #image = Image.new("RGB", (n_of_w_steps, n_of_b_steps))
    #image.putdata(data = line)
    image.save(file_out + ".png")
def save_parssed_file(file_in: str, file_out: str) -> None:
    parssed_file = pars(file_in)
    file = open(file_out + ".ppgn", "wb")
    pickle.dump(parssed_file, file)
    file.close()
def save_line(file_in: str, file_out: str, min_b_elo:int, max_b_elo:int, min_w_elo:int, max_w_elo:int, n_of_b_steps:int, n_of_w_steps:int, sigma:float)-> None:
    table = count_wins_in_all_table(file_in, min_b_elo, max_b_elo, min_w_elo, max_w_elo, n_of_b_steps, n_of_w_steps, sigma)
    line = table_to_line(table)
    file = open(file_out + ".line", "wb")
    pickle.dump(line, file)
    file.close()
def line_to_image(line:list, mode:str, image_size:tuple) -> Image:
    if mode == "divide_by_global_max":
        data = divide_by_global_max(line, find_global_max(line))
    elif mode == "divide_by_local_max":
        data = divide_by_local_max(line)
    elif mode == "one_bit":
        data = one_bit(line)
    elif mode == "HSV":
        data = HSV(line)
    else:
        return "there is no such mode"
    image = Image.new("RGB", image_size)
    image.putdata(data)
    return image
def create_triangle(table,table_size):
    res=table_size*2
    point_1=[table_size,0]
    point_2=[table_size/2,int(math.sqrt(3)*table_size/2)]
    for i in range(res):
        for j in range(res-i):
            point=table[int((i*point_1[0]+j*point_2[0])/res)][int((i*point_1[1]+j*point_2[1])/res)]
            point[0] = res-i-j
            point[1] = i
            point[2] = j
def create_triangle_image(file_out: str, min_b_elo:int, max_b_elo:int, min_w_elo:int, max_w_elo:int, mode:str) -> None:
    table=init_table([max_w_elo-min_w_elo,max_b_elo-min_b_elo])
    create_triangle(table,max_w_elo-min_w_elo)
    line = table.transpose(2,0,1).reshape(3,-1).transpose().tolist()
    #print(table==init_table([max_w_elo-min_w_elo,max_b_elo-min_b_elo]).tolist())
    #line = table_to_line(table)
    image = line_to_image(line, mode, (max_w_elo - min_w_elo, max_b_elo - min_b_elo))
    if image == "there is no such mode":
        print("there is no such mode " + str(mode))
        return
    #image = Image.new("RGB", (n_of_w_steps, n_of_b_steps))
    #image.putdata(data = line)
    image.save(file_out + ".png")

def test_create_table(file: str) -> None:
    print(create_table(file, 2000, 2650, 2000, 2650, 2, 2))
def test_parsser(file: str) -> None:
    print(pars(file))
def test_count_wins_in_all_table(file: str) -> None:
    print(count_wins_in_all_table(file, 2000, 2650, 2000, 2650, 2, 2))
#def test_pgn_file_to_image(file_in: str, file_out: str) -> None:
    #pgn_file_to_image(file_in, file_out, 2000, 2800, 2000, 2800, 80, 80)


#a = np.arange(50, step=2).reshape((5,5))
#print(gaussian_filter(a, sigma=3))

#test_parsser("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\test_1.pgn")
#test_parsser("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\test_2.pgn")
#test_parsser("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\test_3.pgn")
#test_create_table("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\test_2.pgn")
#test_count_wins_in_all_table("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\test_2.pgn")
#test_count_wins_in_all_table("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002.pgn")
#test_pgn_file_to_image("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002.pgn", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002")
#save_table("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002.pgn", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002", 2000, 2800, 2000, 2800, 80, 80)
#save_table("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002.pgn", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_ver_2", 2000, 2800, 2000, 2800, 40, 40)
#save_table("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002.pgn", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_ver_2", 2000, 2800, 2000, 2800, 32, 32)
#save_table("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2011.pgn", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2011_ver_2", 2000, 2800, 2000, 2800, 40, 40)
#save_table("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2011.pgn", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2011_ver_2", 2000, 2800, 2000, 2800, 32, 32)
#save_table("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2011.pgn", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2011_ver_3", 2000, 2800, 2000, 2800, 20, 20)
#table_to_image("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002", 80, 80, "divide_by_global_max")
#table_to_image("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_ver_2", 80, 80, "divide_by_local_max")
#table_to_image("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_ver_3", 80, 80, "one_bit")
#table_to_image("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_ver_2", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_ver_3_2", 40, 40, "one_bit")
#table_to_image("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_ver_2", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_ver_3_2", 32, 32, "one_bit")
#save_table("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2011.pgn", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2011", 2000, 2800, 2000, 2800, 80, 80)
#table_to_image("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2011", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2011", 80, 80, "divide_by_global_max")
#table_to_image("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2011", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2011_ver_2", 80, 80, "divide_by_local_max")
#table_to_image("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2011", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2011_ver_3", 80, 80, "one_bit")
#table_to_image("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2011_ver_2", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2011_ver_3_2", 40, 40, "one_bit")
#table_to_image("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2011_ver_3", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2011_ver_3_3", 20, 20, "one_bit")
#save_parssed_file("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002.pgn", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002")
#table_to_image("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_ver_5", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_ver_5_1", 2000, 2800, 2000, 2800, 40, 40, 14, "one_bit", 10)
#count_wins_all_table2("D:\\Meir\\chess\\DATABASE4U", "D:\\Meir\\chess\\DATABASE4U", 2, 4000, 2, 4000)
#table_to_image("D:\\Meir\\chess\\DATABASE4U", "D:\\Meir\\chess\\DATABASE4U", 2, 4000, 2, 4000, 40, 40, 14, "HSV", 10)
#table_to_image("D:\\Meir\\chess\\DATABASE4U", "D:\\Meir\\chess\\DATABASE4U_2", 2, 4000, 2, 4000, 40, 40, 14, "one_bit", 10)
#table_to_image("D:\\Meir\\chess\\DATABASE4U", "D:\\Meir\\chess\\DATABASE4U_3", 2, 4000, 2, 4000, 40, 40, 14, "divide_by_local_max", 10)
#table_to_image("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_ver_5", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_ver_5_2", 2000, 2800, 2000, 2800, 40, 40, 14, "divide_by_local_max", 10)
#table_to_image("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_ver_5", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_ver_5_3", 2000, 2800, 2000, 2800, 40, 40, 14, "HSV", 10)
create_triangle_image("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\traingle_rgb", 2000, 3000, 2000, 3000, "divide_by_local_max")
create_triangle_image("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\traingle_one_bit", 2000, 3000, 2000, 3000, "one_bit")
create_triangle_image("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\traingle_hsv", 2000, 3000, 2000, 3000, "HSV")
#save_line("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_ver_2_1", 2000, 2800, 2000, 2800, 40, 40, 13)
#count_wins_all_table2("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_ver_5", 2000, 2800, 2000, 2800)
#count_wins_all_table3("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\test_3", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\test_3", 2000, 2800, 2000, 2800)
#count_wins_all_table3("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_ver_6", 2000, 2800, 2000, 2800)
#parssed_file_to_image("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_ver_2_1", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_ver_2_1_1", 2000, 2800, 2000, 2800, 40, 40, 13, "HSV")
#save_line("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_ver_2_2", 2000, 2800, 2000, 2800, 40, 40, 7)
#parssed_file_to_image("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002", "C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002_one_bit_40", 2000, 2800, 2000, 2800, 40, 40, 7, "one_bit")
#current 2001-2002
#putdata
#print(init_i_j(5))
#init_kernel(5)
e=1
#file = open("C:\\Users\\Meir\\Dropbox\\code\\chess project\\pgn files\\current 2001-2002.table", "rb")
#print(pickle.load(file))
#file.close()
