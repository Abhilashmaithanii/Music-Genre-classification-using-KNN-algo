from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np

from tempfile import TemporaryFile

import os
import pickle
import random
import operator

import math

from collections import defaultdict

#UI Libraries
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox as mess
from tkinter import filedialog


# function to get the distance between feature vecotrs and find neighbors
def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    
    return neighbors

# identify the class of the instance
def nearestClass(neighbors):
    classVote = {}

    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1

    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)

    return sorter[0][0]

def distance(instance1 , instance2 , k ):
    distance =0 
    mm1 = instance1[0] 
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
    distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 
    distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance-= k
    return distance


# Split the dataset into training and testing sets respectively
dataset = []

def loadDataset():
    with open('my.dat', 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
   











########---------------------------------------------------##########
# Function for opening the
# file explorer window
def browseFiles():
	filename = filedialog.askopenfilename(initialdir = "/",
										title = "Select a File",
										filetypes = (("Text files",
														"*.wav*"),
													("all files",
														"*.*")))
    	# Change label contents
	label_file_explorer.configure(text=filename)
#------------------------------------------------------------


	dataSet = []

	loadDataset()
	
	directory ="/Users/lenovo/Desktop/Project/Data/genres_original/"
	results = defaultdict(int)	

	i = 1
	for folder in os.listdir(directory):
		results[i] = folder
		i += 1

	(rate, sig) = wav.read(filename)
	mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
	covariance = np.cov(np.matrix.transpose(mfcc_feat))
	mean_matrix = mfcc_feat.mean(0)
	feature = (mean_matrix, covariance, i)
	pred = nearestClass(getNeighbors(dataset, feature, 5))
	genre =  Tk()
	genre.geometry("150x150")
	mess.showinfo("genre",results[pred])




#-----------------------GUI----------------------------
window = tk.Tk()
window.geometry("500x500")
window.resizable(False,False)
window.title("Music Genres Classification")
window.configure(background='#191919')

message3 = tk.Label(window, text="Music Genres Classification System" ,fg="white",bg="#191919" ,width=30 ,height=1,font=('times', 20, ' bold '))
message3.place(x=13, y=5)

frame1 = tk.Frame(window, bg="#04293A")
frame1.place(relx=0, rely=0.14, relwidth=1, relheight=0.85)

frame2 = tk.Frame(frame1, bg="white")
frame2.place(x=70, y=120, width=300, relheight=0.07)

lbl = tk.Label(frame1, text="Import Song",width=35  ,height=1  ,fg="black"  ,bg="#F3950D" ,font=('times', 17, ' bold ') )
lbl.place(x=0, y=55)

label_file_explorer = tk.Label(frame2, text="aa ",width=40  ,height=1  ,fg="black"  ,bg="white" ,font=('times', 12, ' bold ') )
label_file_explorer.place(x=0, y=0)


importSong = tk.Button(frame1, text = "import",
						command = browseFiles  ,fg="black"  ,bg="#F3950D"  
                        , activebackground = "white" ,font=('times', 15, ' bold '))
importSong.place(x=376,y=120, relwidth=0.2, relheight=0.07)



window.mainloop()