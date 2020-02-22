import cv2
import tkinter
import PIL.Image, PIL.ImageTk


class App:
    def __init__(self, title="CNN Demo", cap_source=1):
        # CV2 capture setup
        self.frame = None
        self.capture = Capture(cap_source)

        # Imaging
        self.thresholding = False
        self.thresh_val = None

        self.target_mode = False

        # GUI setup
        self.window = self.build_gui(title, self.capture)        

        # Run GUI
        self.delay = 15
        self.update()
        self.window.mainloop()

    def build_gui(self, title, capture):
        # Root window and container frames
        self.window = tkinter.Tk()
        self.window.title(title)
        self.thresh_val = tkinter.DoubleVar(self.window,value=127)
        self.btn_frame = tkinter.Frame(self.window)
        self.btn_frame.pack(side=tkinter.LEFT,expand=True,anchor=tkinter.NW)
        self.img_frame = tkinter.Frame(self.window)
        self.img_frame.pack()

        # Control frame items
        self.btn_thresh = tkinter.Button(self.btn_frame,text="Threshold",width=25,command=self.button_threshold)
        self.btn_thresh.pack(side=tkinter.TOP,expand=True)
        self.scale_thresh = tkinter.Scale(self.btn_frame,variable=self.thresh_val,orient=tkinter.HORIZONTAL,from_=0,to=255)
        self.scale_thresh.pack()
        self.btn_target_mode = tkinter.Button(self.btn_frame,text="TEST",width=25,command=self.button_target_mode)
        self.btn_target_mode.pack()

        # Data frame items
        self.canvas = tkinter.Canvas(self.img_frame, width=self.capture.width, height=self.capture.height)
        self.canvas.pack()

        return self.window

    def button_threshold(self):
        if self.thresholding:
            self.thresholding = False
        else:
            self.thresholding = True

    def button_target_mode(self):
        self.thresholding = True
        self.target_mode = True
            

    def update(self):
        ret,self.frame = self.capture.get_frame()
        if ret:
            self.process_frame()
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.frame))
            self.canvas.create_image(0,0,image=self.photo,anchor=tkinter.NW)
        self.window.after(self.delay, self.update)

    def process_frame(self):
        if self.thresholding:
            self.do_threshold()
        if self.target_mode:
            self.detect_targets()


    def do_threshold(self):
        ret,self.frame = cv2.threshold(self.frame,self.thresh_val.get(),255,cv2.THRESH_BINARY)
        self.frame = cv2.adaptiveThreshold(self.frame,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,7,7)
    
    def detect_targets(self):
        contours_found = False
        if self.thresholding:
            # if thresholding was already on, don't do it twice
            pass
        else:
            self.do_threshold()
        try:
            contour_img, contours = cv2.findContours(self.frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            print(contours[0])
            if len(contours) != 0:
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(self.frame,contours,-1,(0,255,0),1)
        except Exception as e:
            print(e)


class Capture:
    def __init__(self, source=0):
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise ValueError("Could not open capture source")
        
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.cx = int(self.width / 2)
        self.cy = int(self.height / 2)
        
    def __del__(self):
        self.capture.release()

    def get_frame(self):
        if self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY))
            else:
                return (ret, None)
        else:
            return None
    
App()
