


from pathlib import Path
from useModels import predictImage

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, filedialog, Label
import numpy as np
from PIL import Image, ImageTk


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\UVT TW\IAProj\build\assets\frame0")


def browseFiles():
    filename = filedialog.askopenfilename(initialdir = "./",
                                          title = "Select an image",
                                          filetypes = [("Cute Images",".png .jpeg .jpg")]
                                          )
      
    # Change label contents
    print("File Opened: "+filename)
    if not filename:
        return
    
    load = Image.open(filename)
    image_2=ImageTk.PhotoImage(load.resize((250,250)))

    image_label.configure(image= image_2)
    image_label.image = image_2
    

    # cnn=predictImage(filename,"./models/catsvsdogsCNN.h5")
    # print(cnn[0][0])
    # if cnn[0][0]<0.5:
    #     textCNN="CNN: Cat "+ str(cnn[0][0])
    # else:
    #     textCNN="CNN: Dog "+ str(cnn[0][0])
    
    cnn=predictImage(filename,"./models/modelCNN.h5")
    print(cnn)
    if np.argmax(cnn, axis=-1)==[0]:
        textCNN="CNN: Cat "+ str(cnn[0][0])
    else:
        textCNN="CNN: Dog "+ str(cnn[0][1])

    svm=predictImage(filename,"./models/catsvsdogsSVM.h5")
    if svm[0][0]<0:
        textSVM="SVM: Cat "+ str(svm[0][0])
    else:
        textSVM="SVM: Dog "+ str(svm[0][0])
    canvas.itemconfig(text1,text = textCNN)
    canvas.itemconfig(text2,text = textSVM)
    
    print(svm[0][0])
    

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("1280x720")
window.configure(bg = "#AAC7FF")


canvas = Canvas(
    window,
    bg = "#AAC7FF",
    height = 720,
    width = 1280,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    610.0,
    420.0,
    image=image_image_1
)
image_label=Label()
image_label.place(x=500.0,
    y=276.0)


canvas.create_text(
    127.0,
    209.0,
    anchor="nw",
    text="Upload a 🐱 or 🐶 image and let the machine guess what it is 👍",
    fill="#000000",
    font=("Inter", 36 * -1)
)

canvas.create_text(
    323.0,
    107.0,
    anchor="nw",
    text="Welcome to the cat vs dog classifier",
    fill="#000000",
    font=("Inter", 36 * -1)
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=browseFiles,
    relief="flat"
)
button_1.place(
    x=486.0,
    y=546.0,
    width=293.0,
    height=104.0
)

text1=canvas.create_text(
    850.0,
    618.0,
    anchor="nw",
    text="SVM:",
    fill="#000000",
    font=("Inter", 32 * -1)
)

text2=canvas.create_text(
    100.0,
    618.0,
    anchor="nw",
    text="CNN:",
    fill="#000000",
    font=("Inter", 32 * -1)
)
window.resizable(False, False)
window.mainloop()
