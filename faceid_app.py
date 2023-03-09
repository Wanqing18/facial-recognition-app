# import kivy depencies, under conda python interpreter, reinstall and restart the VS
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

#import other kivy stuff
from kivy.clock import Clock # for real time capture
from kivy.graphics.texture import Texture #convert imgae to texture
from kivy.logger import Logger

# import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# build app and layput

class CamApp(App):

    def build(self):
        #Main layout, image, button, label
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text = 'Verify', on_press= self.verify, size_hint=(1,.1))
        self.verification_label = Label(text='Verification Uninitiated', size_hint=(1,.1))

        # add items to layout, layout positiion is following oders below
        layout = BoxLayout(orientation = 'vertical') # set the layout as vertical
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        #load keras model
        model_path = 'C:/Users/Gaming\Desktop/Test Code/FaceRecognition/App/siamesemodel.h5'
        self.model = tf.keras.models.load_model(model_path,custom_objects={'L1Dist':L1Dist})

        # Set up vedio capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0) #set up real time capture, run every 1/33 sec



        return layout
    
    # run continuosly to get webcam feed
    def update(self,*args):

        # read fram from openCV
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250,:]

        # Flip horizontal and convert image to texture
        buf= cv2.flip(frame,0).tostring() # flip into horizontal, and convert it into str
        img_texture= Texture.create(size = (frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt ='ubyte')#blit it to the texture
        self.web_cam.texture = img_texture
    
    # load image from file and convert to 100x100 pixels
    def preprocess(self, file_path):
        #read in image from file path
        byte_img=tf.io.read_file(file_path)
        #load in the image: convert img into readable code type(into np arrary)
        img=tf.io.decode_jpeg(byte_img)
        
        # resize into 100x100
        img=tf.image.resize(img,(100,100))
        #scale image's pixel value to 0 to 1, normalization 
        img=img/255.0
        return img
    
    #verification function to verify image
    def verify(self,*args):
        #specify threshold, keep adjusting improve verified rate
        detection_threshold = 0.7
        verification_threshold = 0.8

        #capture input image from our webcam
        save_path=os.path.join('application_data','input_image','input_image.jpg')

        # read fram from openCV
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250,:]
        # save img into path
        cv2.imwrite(save_path, frame)

        #build results arrary
        results=[]
        for image in os.listdir(os.path.join('application_data', 'verification_image')):
            # get the input image from webcam(screen shot) put into input file, named as input_image.jpg
            input_img =self.preprocess(os.path.join('application_data','input_image','input_image.jpg'))
            validation_img =self.preprocess(os.path.join('application_data', 'verification_image', image))

            #make predictions: single sample need to rapps sample, multiple no need to rapping into a list
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1))) # extend input dimensions into 2
            results.append(result)
            
        # detection threshold: metric above which a prediction is considered positive    
        detection = np.sum(np.array(results) > detection_threshold) # sum up all passed results
        
        # verfication threshold: proportion of positive prediction/ total positive sample
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_image'))) #proportion
        verified = verification > verification_threshold #true or false

        #update verification text
        self.verification_label.text ='verified' if verification == True else 'Unverified'
        
        #log out details 
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)
        
        return results, verified




if __name__ == '__main__':
    CamApp().run()
