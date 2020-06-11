# -*- coding: utf-8 -*-
"""
Created on Sun May 31 22:40:49 2020

@author: Medhavi
"""

from tkinter import *
import tensorflow as tf
from PIL import ImageTk,Image
import cv2
import numpy as np
import os
import pathlib
import MySQLdb
from datetime import date 
import time  
  

#------------- Define Class --------------------

class AttendanceSystem:

    def __init__(self):        
#------------- Dividing window IN Dsiplay Area and Toolbox Frames -------------
        self.root = Tk()
        self.canvas_frame = Frame(self.root)
        self.canvas_frame.pack(expand=TRUE, fill=BOTH, side=LEFT)

        self.toolbox = Frame(self.root)
        self.toolbox.pack(expand=FALSE, fill=BOTH, side=RIGHT)

        self.drawing_area = Canvas(self.canvas_frame, bd=10, bg="white", relief=RAISED)
        self.drawing_area.pack(expand=TRUE, fill=BOTH, side=LEFT)

#-------------- Text Box to get folder in which to store face image --------------------
        self.folder = Entry(self.toolbox, width=20)
        self.folder.grid()
        self.name = Entry(self.toolbox, width=20)
        self.name.grid()

#-------------- Store Image Button --------------------
        self.storeButton = Button(self.toolbox, text="Store Image", fg="black", width=20)
        self.storeButton.bind("<ButtonPress-1>",self.store)
        self.storeButton.grid()

#-------------- Train Button to train model----------------
        self.trainButton = Button(self.toolbox, text="Train Model", fg="black", width=20)
        self.trainButton.bind("<ButtonPress-1>",self.train)
        self.trainButton.grid()

#-------------- Take Attendance Button ----------------
        self.attendanceButton = Button(self.toolbox, text="Take Attendance", fg="black", width=20)
        self.attendanceButton.bind("<ButtonPress-1>",self.attendance)
        self.attendanceButton.grid()

#-------------- Stop Taking Attendance Button ----------------
        self.stopAttendanceButton = Button(self.toolbox, text="Stop Taking Attendance", fg="black", width=20)
        self.stopAttendanceButton.bind("<ButtonPress-1>",self.stopAttendance)
        self.stopAttendanceButton.grid()
    
#-------------- Define class variables -------------------
        self.camSrc = "http://192.168.43.1:8080/video"
        self.imageDisplay = None
        self.model = None
        try:
            self.model =  tf.keras.models.load_model('FacesRecognized')
        except:
            pass
        self.takeAttendance = False
        data_dir = pathlib.Path("DataSet")
        list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
        self.CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
        
#-------------- Database Connectivity --------------------
        self.db = MySQLdb.connect("localhost","root","root","db1" ) #System, DB user , db password , database
        self.cursor = self.db.cursor()
        # Add column for today's attendance
        try:
            self.today = str(date.today()) 
            self.today = self.today.replace("-","_")
            print(self.today, type(self.today))
            self.cursor.execute("ALTER TABLE register ADD COLUMN " + self.today + " VARCHAR(1) NOT NULL DEFAULT 'N';")
        except:
            pass

        
#-------------- Run window -------------
        self.root.mainloop()
    
    
            
        

#-------------- Draw Rectangle -------------------
    def rectangle_draw(self,bounds, predim):
        x,y,w,h = bounds
        self.drawing_area.delete("all")
        self.imageDisplay = ImageTk.PhotoImage(predim)
        self.drawing_area.create_image(0,0,anchor=NW, image = self.imageDisplay)
        self.drawing_area.create_rectangle(x,y,x+w,y+h, outline="green")
        self.root.update()

#-------------- Store face Images in a folder -------------------
    def store(self, event):
        folder_name = self.folder.get().strip()
        student_name = self.name.get().strip()
        
        if( folder_name == "" or student_name == ""):
            print("Enter folder name i.e. Roll and student name.")
            return
            
        sql = "INSERT INTO register (roll, name) VALUES (%s, %s)"
        val = (folder_name , student_name )
        try:
            self.cursor.execute(sql, val)
            self.db.commit()
        except:
            pass
        
        hsc = "hfd.xml"
        fcc = cv2.CascadeClassifier(hsc)
        video = cv2.VideoCapture(self.camSrc)
        if video is None or not video.isOpened():
            print("Warning: unable to open video source")
            return
        try:  
            os.mkdir("DataSet/"+folder_name)  
        except OSError as error:  
            print(error) 

        i = 0
        while i < 20:
            time.sleep(0.1)
            _ , img = video.read()
            predim = Image.fromarray(img)
            w, h = predim.size
            predim = predim.resize((int(w/2),int(h/2)))
            img = np.asanyarray(predim)
            faces = fcc.detectMultiScale(img,1.1,4,cv2.CASCADE_SCALE_IMAGE)            

            for (x,y,w,h) in faces:
                bounds = [x,y,w,h]
                self.rectangle_draw(bounds, predim)
                i += 1      #------ To take 10 faces
                temp = predim.crop((x,y,x+w,y+h))
                temp = temp.resize((60,60))
                filepath = "DataSet/"+folder_name +"/"+str(i)+".jpg"
                temp.save(filepath)
        print("Done")

#------------- Train model ---------------------------------
    def get_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return parts[-2] == self.CLASS_NAMES

    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return img
    
    def process_path(self, file_path):
        label = self.get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label
    
    
    def prepare(self, dataset):
        train, label = dataset[:,0] , dataset[:,1]
        x_train = np.zeros((len(train), 60 , 60 , 3))
        y_train = np.zeros((len(train), 5))
        for i in range(len(train)):
            x_train[i] = train[i].numpy()
            y_train[i] = label[i].numpy()
        return x_train , y_train

    def train(self, event):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        data_dir = pathlib.Path("DataSet")
        list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
        self.CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
        print(self.CLASS_NAMES)
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        labeled_ds = list_ds.map(self.process_path, num_parallel_calls=AUTOTUNE)
        dataset = np.asarray(list(labeled_ds))
        train , label = self.prepare(dataset)
        print(label.shape)
        print(train.shape)
        self.model = tf.keras.models.Sequential([
              tf.keras.layers.Flatten(input_shape=(60, 60, 3)),
              tf.keras.layers.Dense(128, activation='relu'),
              tf.keras.layers.Dense(128, activation='relu'),
              tf.keras.layers.Dense(len(self.CLASS_NAMES), activation="softmax")
            ])

        self.model.compile('sgd', loss=tf.keras.losses.CategoricalCrossentropy())
        self.model.fit(train, label, epochs=500)
        self.model.save("FacesRecognized")


#----------- Attendance record --------------------------------------
    def attendance(self, event):
        hsc = "hfd.xml"
        fcc = cv2.CascadeClassifier(hsc)
        video = cv2.VideoCapture(self.camSrc)
        if video is None or not video.isOpened():
            print("Warning: unable to open video source")
            return
        self.takeAttendance = True
        self.cursor.execute("SELECT * FROM register")
        results = self.cursor.fetchall()

        while self.takeAttendance:
            _ , img = video.read()
            predim = Image.fromarray(img)
            w, h = predim.size
            predim = predim.resize((int(w/2),int(h/2)))
            img = np.asanyarray(predim)
            faces = fcc.detectMultiScale(img,1.1,4,cv2.CASCADE_SCALE_IMAGE)            

            for (x,y,w,h) in faces:
                bounds = [x,y,w,h]
                self.rectangle_draw(bounds, predim)
                temp = predim.crop((x,y,x+w,y+h))
                temp = temp.resize((60,60))
                arr = np.asanyarray(temp) / 255
                prediction = self.model.predict(arr[None,:])[0]
                maxProb = np.argmax(prediction)
                if(self.CLASS_NAMES[maxProb] == "0" or prediction[maxProb] < 0.7):
                    self.showName(bounds, results, 0)
                    self.root.update()
                    continue
                sql = "UPDATE register SET "+self.today+" = 'Y' WHERE roll = '"+self.CLASS_NAMES[maxProb]+"' ;"
                self.cursor.execute(sql)
                self.showName(bounds, results, self.CLASS_NAMES[maxProb])
                self.db.commit()
                self.root.update()

                
#----------- Stop Attendance -------------------------------------
    def stopAttendance(self, event):
        self.takeAttendance = False

#----------- Show name --------------------------------------------
    def showName(self , bounds, results , roll):
        if(roll == 0):
            self.drawing_area.create_text(bounds[0],bounds[1]-5,fill="red",font="Times 10 italic bold",text="Unkonwn")
            return
        for res in results:
            if(res[0] == roll):
                self.drawing_area.create_text(bounds[0],bounds[1]-5,fill="green",font="Times 10 italic bold",text=res[1])
                return

attendance = AttendanceSystem()
