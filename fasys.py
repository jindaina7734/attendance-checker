import cv2
import face_recognition
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import winsound

# create GUI window
root = tk.Tk()
root.title("Attendance Checker")
path = Path(__file__).resolve().parent / 'Pe'

def empty_csv():
    with open(Path(__file__).with_name('Attendance.csv'), mode='w') as file:
        file.write('Student Name / Class,Time,Date')
        file.close()

# menu bar that has a file menu
menubar = tk.Menu(root) 
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Empty CSV", command=lambda: empty_csv())
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)
root.config(menu=menubar)

images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList

encoded_face_train = findEncodings(images)

def markAttendance(name):
    with open(Path(__file__).with_name('Attendance.csv'),'r+', encoding='cp932', errors='ignore') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'\n{name}, {time}, {date}')
        f.close()

# create canvas widget to display video capture
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

# create text widget to display attendance name
attendance_label = tk.Label(root, text="Attendance Checking...", font=("Helvetica", 20))
attendance_label.pack(pady=10)

# csv content
df = pd.read_csv(Path(__file__).with_name('Attendance.csv'))
# split the name and class column
df[['Student Name', 'Class']] = df['Student Name / Class'].str.split(' ', expand=True)
df = df.drop(['Student Name / Class'], axis=1)
# swap the columns
df = df[['Student Name', 'Class', 'Time', 'Date']]

# adjust the text position
tree = ttk.Treeview(root)
tree["columns"] = list(df.columns)
tree.column("#0", width=50)
for column in df.columns:
    tree.column(column, width=100)
    tree.heading(column, text=column)

# Insert data into the treeview 
for index, row in df.iterrows():
    tree.insert('', 'end', text=index, values=list(row))

# Pack the widgets into the window
tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# create video capture object
cap = cv2.VideoCapture(0)
update_after = 3000 # update after every 3 seconds
last_update_time = datetime.now()

while True:
    current_time = datetime.now()
    delta = current_time - last_update_time
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    if delta.total_seconds() * 1000 >= update_after:
        for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
            matches = face_recognition.compare_faces(encoded_face_train, encode_face)
            faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
            matchIndex = np.argmin(faceDist)
            print(matchIndex)
            if matches[matchIndex]:
                name = classNames[matchIndex].lower()
                y1,x2,y2,x1 = faceloc
                # Since we scaled down by 4 times
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                # update attendance label text
                attendance_label.config(text=f"Attendance Checked: {name}") 
                # update csv content
                df_updated = pd.read_csv(Path(__file__).with_name('Attendance.csv'))
                tree.delete(*tree.get_children())
                for index, row in df_updated.iterrows():
                    tree.insert('', 'end', text=index, values=list(row))
                winsound.PlaySound('Welcome.wav', winsound.SND_ALIAS)
                markAttendance(name)
            else:
                attendance_label.config(text="Attendance Checked: ") # reset attendance label text
        last_update_time = current_time # update last update time
    else:
        attendance_label.config(text="Attendance Checking...") # reset attendance label text
    cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor=tk.NW, image=imgtk) # display video capture on canvas
    root.update() # update GUI window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close window and release video capture object
cap.release()
cv2.destroyAllWindows()
root.mainloop()
