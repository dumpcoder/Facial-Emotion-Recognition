import tkinter as tk
from tkinter import filedialog
from PIL import Image as im
from resizeimage import resizeimage
import numpy as np
import random
from model import *

def truncate_filename(filename):
    index = 0
    for x in range(len(filename)):
        if filename[x] == '/':
            index = x
    return filename[index+1:]

def file_path(filename):
    index = 0
    for x in range(len(filename)):
        if filename[x] == '/':
            index = x
    return filename[:index+1]
       
class MyApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        #TK INIT
        tk.Tk.__init__(self, *args, **kwargs)

        # Variables
        self.filename = "" 

        # Window
        window_color ="#88788B"
        self.title("FER")
        self.configure(bg=window_color)

        # Widgets
        self.sf_lbl = tk.Label()
        self.f_lbl = tk.Label()
        self.c_lbl = tk.Label() 
        self.f_btn = tk.Button(command=self.clicked)
        self.c_btn = tk.Button(command=self.classify)

        # Configuration
        lbl_font = ("courier new", 12)
        btn_font = ("Helvetica", 9, "bold")
    

        self.sf_lbl.configure(text="file selected:",fg='white', bg=window_color, font=lbl_font)
        self.f_lbl.configure(text="example.jpg", bg='white', fg="#7b8b78", font=lbl_font)
        self.f_btn.configure(text="Select Image", font=btn_font)
        self.c_btn.configure(text="Classify",bg="red", fg="white", font=btn_font)
      

        # Postioning
        self.sf_lbl.grid(row=0, column=0, padx=(10,0), pady=10)
        self.f_lbl.grid(row=0, column=1, padx=(0,10), pady=10)
        self.f_btn.grid(row=0, column=2, padx=10, pady=10)
        self.c_btn.grid(row=2, column=1, padx=10, pady=10)
     
    def clicked(self):
        tmp_filename = tk.filedialog.askopenfilename(initialdir = "/",title = "Select Image",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        if(tmp_filename != ""):
            self.filename = tmp_filename
            self.f_lbl.configure(text=truncate_filename(self.filename))

    def classify(self):
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        classification = None

        if self.filename == "":
            classification = "No file selected"
        else:
            image_file = im.open(self.filename, 'r')
            image_file = image_file.convert('L')
            filter_image = resizeimage.resize_cover(image_file, [48, 48])
            # f_image.save('{}crop_{}'.format(file_path(self.filename), truncate_filename(self.filename)))
            data = np.asarray(filter_image, dtype=np.float32)
            
            emotion_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn, model_dir="/users/tomas/Code/Machine_Learning/Project/emotion_convnet_model") # Change to path of emotion_convent_model on your computer
            # Evaluate the model and print results
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": data},
                num_epochs=1,
                shuffle=False)

            results = emotion_classifier.predict(input_fn=eval_input_fn)
            prediction = None
            for result in results:
                    prediction = result
            classification = emotions[prediction['classes']]
            print(prediction)
        
        myFont = ("courier new", 14, "bold")
        self.c_lbl.configure(text=classification,fg="#86b6db", bg="#88788B", font=myFont)
        self.c_lbl.grid(row=1, column=1, padx=10, pady=10)
            

if __name__ == "__main__":
    app = MyApp()
    app.mainloop()