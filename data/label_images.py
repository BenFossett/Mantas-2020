import argparse
import tkinter as tk
import numpy as np
from tkinter import *
import os
import json
from shutil import copyfile, move
from PIL import ImageTk, Image
from scipy import misc

class App:
    def __init__(self, window, window_title, image_paths, continue_true):
        self.window = window
        self.window.title(window_title)
        self.annotations = json.load(open('mantaAnnotations.json'))
        self.annotations = self.annotations["annotations"][0]

        if continue_true and os.path.exists('data.json'):
            self.data = json.load(open('data.json'))
            self.index = self.data['position']
        else:
            self.data = {}
            self.data['mantas'] = []
            self.data['position'] = 0
            self.index = 0

        frame = tk.Frame(window)
        frame.grid()

        print(self.index)

        # Start at the first file name
        self.n_paths = len(image_paths)

        self.image_raw = None
        self.image = None
        self.image_panel = tk.Label(frame)

        self.show_image(self.annotations[self.index])

        self.technical = DoubleVar()
        slider1 = Scale(frame, variable = self.technical, from_=1, to=5, resolution=1, orient=HORIZONTAL, length=None, showvalue=0)
        tech_label1 = Label(frame, text="Technical Quality", width=10)
        tech_label2 = Label(frame, textvariable=self.technical, width=10)

        self.manta = DoubleVar()
        slider2 = Scale(frame, variable = self.manta, from_=1, to=5, resolution=1, orient=HORIZONTAL, length=None, showvalue=0)
        manta_label1 = Label(frame, text="Manta Quality", width=10)
        manta_label2 = Label(frame, textvariable=self.manta, width=10)

        # Add progress label
        progress_string = "%d/%d" % (self.index, self.n_paths)
        self.progress_label = tk.Label(frame, text=progress_string, width=10)

        # Place buttons in grid
        slider1.grid(row=2, column=0, rowspan=1, columnspan=1, sticky='we')
        slider2.grid(row=2, column=4, rowspan=1, columnspan=1, sticky='we')
        tech_label1.grid(row=0, column=0)
        tech_label2.grid(row=1, column=0)
        manta_label1.grid(row=0, column=4)
        manta_label2.grid(row=1, column=4)
        tk.Button(frame, text="Confirm", width=10, height=1, command=self.confirm).grid(row=3, column=0, sticky='we')
        tk.Button(frame, text="Save and Exit", width=10, height=1, command=self.exit).grid(row=3,column=5)

        frame.bind("<Return>", self.confirm)
        frame.bind("<Escape>", self.exit)

        # Place progress label in grid
        self.progress_label.grid(row=0, column=5, sticky='we')

        # Place the image in grid
        self.image_panel.grid(row=4, column=0, columnspan=6, sticky='we')
        frame.focus_set()

    def next_image(self):
        self.index +=1
        self.data['position'] = self.index
        self.progress_label.configure(text="%d/%d" % (self.index, self.n_paths))
        self.technical.set(1)
        self.manta.set(1)

        if self.index < self.n_paths:
            self.show_image(self.annotations[self.index])
        else:
            with open('new_labels.json', 'w') as outfile:
                json.dump(self.data, outfile, indent=4)
            self.window.quit()

    def show_image(self, annotation):
        path = "mantas/" + annotation["uniqueImageFileName"]
        image = self._load_image(path, annotation)
        self.image = image
        #self.image = ImageTk.PhotoImage(image)
        self.image_panel.configure(image=self.image)

    def confirm(self, event=None):
        image_id = self.annotations[self.index]["uniqueImageFileName"]
        image_class = self.annotations[self.index]["individualId"]
        self.data['mantas'].append({
            'image_id': image_id,
            'image_class': image_class,
            'technical': self.technical.get(),
            'manta': self.manta.get(),
        })
        self.next_image()

    @staticmethod
    def _load_image(path, annotation, size=(512,512,3)):
        box = np.array(annotation["box_xmin_ymin_xmax_ymax"])
        box = box.astype(np.int32)
        box = box - 1

        image = misc.imread(path)
        image = image[box[1]:box[3], box[0]:box[2], :]
        image = misc.imresize(image, size)
        image = Image.fromarray(image)

        image.save('mantas_cropped/' + str(annotation["uniqueImageFileName"]))
        image = ImageTk.PhotoImage(image)
        return image

    def exit(self, event=None):
        with open('data.json', 'w') as outfile:
            json.dump(self.data, outfile, indent=4)
        self.window.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', help='Input folder containing the images', default='mantas/')
    parser.add_argument('--continue_true', help='Continue from where you left off', action='store_true')
    args = parser.parse_args()

    input_folder = args.folder
    continue_true = args.continue_true

    image_paths = []
    for file in os.listdir(input_folder):
        if file.endswith(".jpg") or file.endswith(".jpeg"):
            path = os.path.join(input_folder, file)
            image_paths.append(path)

    if not os.path.exists('mantas_cropped/'):
        os.makedirs('mantas_cropped/')

    window = tk.Tk()
    app = App(window, "Manta Quality Labelling", image_paths, continue_true)
    window.mainloop()
