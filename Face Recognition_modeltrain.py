#!/usr/bin/env python
# coding: utf-8

# In[29]:


#Import dependencies
import cv2 # for image input
import os # for load data from different path
import random
import numpy as np # we need to transfer image into arrary, tensorflow requre arrary input
from matplotlib import pyplot as plt # to visualize image through plt.imshow


# In[30]:


# import tensorflow dependencies-functional API
# we need funcitonal model for image recognition using Siamese NN model
from tensorflow.keras.models import Model # is more flexible than sequential, multiple output and input
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf


# In[72]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[70]:


gpus=tf.config.list_physical_devices('GPU')


# In[71]:


gpus


# In[31]:


# through passing 2 images at the same time, model determined if the image is same/similar
# model(inputs=[inputimages,verificaitionimage],outputs=[1,0]) 1:very similar,verified; o not verified
#different that sequential api
# class L1Dist(Layer): are able to create cutomized NN layer


# ### Create Folder Structures
# 

# In[12]:


# setup path
pos_path=os.path.join('data','positive')
neg_path=os.path.join('data','negative ')
anc_path=os.path.join('data','anchor')


# In[13]:


# files get created on jupyter notebook
os.makedirs(pos_path)
os.makedirs(neg_path)
os.makedirs(anc_path)
# anch vs positive-> return 1 similar
# anch vs positive -> return 0


# ### Collect data

# In[ ]:


# using data from http://vis-www.cs.umass.edu/lfw/#download
# once download, create a checkpoint on the folder that we want to work at, then put the zipfile into the same jupyternotbook file
# then uncompress tar_gz labelled faces

get_ipython().system('tar -xf lfw.tgz')


# In[32]:


lfw_path='path_to_lfw'
neg_path='path_to_negative'


# In[15]:


# move lfw images to the following repository data/negative
for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw', directory)):
        ex_path=os.path.join(lfw_path, directory,file)# path for each image
        new_path=os.path.join(neg_path,file)# newpath is neg folder: image
        os.replace(ex_path,new_path)
# image dimensions are 250x250       


# In[33]:


pos_path='path_to_positive'
anc_path='path_to_anchor'


# ### Collect positive and Anchor Classes

# In[34]:


# import uuid to generate unique image names
import uuid


# In[18]:


# Established a connection to the webcam, pos and an picture come from open sources

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #cut Down the image frame to 250x250px
    frame=frame[120:120+250,200:200+250,:]
    
    # collect anchors, changing different keyword let the scrrenshot belongs to different folder
    if cv2.waitKey(1) & 0XFF == ord('a'):
        #create unique file path for each anchor image
        imgname=os.path.join(anc_path,'{}.jpg'.format(uuid.uuid1()))
        #write out anchor image
        cv2.imwrite(imgname,frame)
        
    #collect positive
    if cv2.waitKey(1) & 0XFF == ord('p'):
        imgname=os.path.join(pos_path,'{}.jpg'.format(uuid.uuid1()))
        #write out positive image
        cv2.imwrite(imgname,frame)
    
    
    # Display the resulting frame
    cv2.imshow('Image Collection',frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
        
# if has error, change number x in videocapture(x)


# As we press 'a' or 'p', postive and anchor folder will start to collect our selffie
# I collect 424 for anchor, 346 for postiive

# In[14]:


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))#the shape is not matched requirement yet


# ### To improve accuracy, we make our inpur into different type--increase sample size

# In[35]:


def data_aug(img):
    data=[]
    for i in range(9):
        img= tf.image.stateless_random_brightness(img,max_delta=0.02,seed=(1,2))
        img= tf.image.stateless_random_contrast(img,lower=0.6,upper=1,seed=(1,3))
        img= tf.image.stateless_random_flip_left_right(img,seed=(np.random.randint(100),np.random.randint(100)))
        img= tf.image.stateless_random_jpeg_quality(img,min_jpeg_quality=90,max_jpeg_quality=100,seed=(np.random.randint(100),np.random.randint(100)))
        img= tf.image.stateless_random_saturation(img,lower=0.9,upper=1,seed=(np.random.randint(100),np.random.randint(100)))
        
        data.append(img)
    return data


# In[39]:


for file_name in os.listdir(os.path.join(anc_path)):
    img_path = os.path.join(anc_path, file_name)
    img= cv2.imread(img_path)
    augmented_images = data_aug(img)
    
    for image in augmented_images:
        imgname=os.path.join(anc_path,'{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, image.numpy())


# In[41]:


for file_name in os.listdir(os.path.join(pos_path)):
    img_path = os.path.join(pos_path, file_name)
    img= cv2.imread(img_path)
    augmented_images = data_aug(img)
    
    for image in augmented_images:
        imgname=os.path.join(pos_path,'{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, image.numpy())
    


# ### Load data

# In[42]:


#get image directory
# take 300 units of jpg file in anc_path, we want all the sample size be the same, thats why we choose 300
anchor=tf.data.Dataset.list_files(anc_path+'\*.jpg').take(3000) # create a pipeline
positive=tf.data.Dataset.list_files(pos_path+'\*.jpg').take(3000)
negative=tf.data.Dataset.list_files(neg_path+'\*.jpg').take(3000)


# In[19]:


anchor.as_numpy_iterator().next()


# ### Preprocessing : scale and resize

# In[43]:


def preprocess(file_path):
    #read in image from file path
    byte_img=tf.io.read_file(file_path)
    #load in the image: convert img into readable code type(into np arrary)
    img=tf.io.decode_jpeg(byte_img)
    
    # resize into 100x100
    img=tf.image.resize(img,(100,100))
    #scale image's pixel value to 0 to 1, normalization 
    img=img/255.0
    return img
    


# ### Create Labelled Dataset

# In[22]:


tf.ones(len(anchor))


# In[44]:


# create label for different class
#(anchor, positive)=>1,1,1,1,1
#(anchor, negative)=>0,0,0,0
positives=tf.data.Dataset.zip((anchor,positive,tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))# create a list of label for positive
negatives=tf.data.Dataset.zip((anchor,negative,tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data=positives.concatenate(negatives) # join pos and neg together into 1 dataset (str,str,float) anchor file, pos/neg folder,1 or 0


# In[45]:


# pack anchor,pos, label together
def preprocess_twin(input_img,validation_img,label):
    return(preprocess(input_img),preprocess(validation_img),label)


# In[46]:


data


# In[46]:


#build dataloader pipeline
data=data.map(preprocess_twin)
data=data.cache()#temperary saved
data=data.shuffle(buffer_size=10000)# when sample size increase, buffer size should also increase


# In[47]:


# Training Partition
train_data=data.take(round(len(data)*.7)) # 70% train data
train_data=train_data.batch(16) # atch is a grouping of instances from your dataset.The batch method allows users to process data when computing resources are available, and with little or no user interaction.
train_data=train_data.prefetch(8)  #This allows later elements to be prepared while the current element is being processed. This often improves latency and throughput, at the cost of using additional memory to store prefetched elements.


# In[48]:


# test partition
test_data=data.skip(round(len(data)*.7))
test_data=test_data.take(round(len(data)*.3))
test_data=test_data.batch(16) 
test_data=test_data.prefetch(8) 


# ### Build up Model-- Siamese Network

# In[ ]:


### L1 Distance layer is used to compare whether anchor is similar to positive or negative
### still using CNN, but pass 2 image at once for comparing similarity


# In[ ]:


#Padding:the amount of pixels added to an image when it is being processed by the kernel of a CNN. For example, if the padding in a CNN is set to zero, then every pixel value that is added will be of value zero.
#https://towardsdatascience.com/covolutional-neural-network-cb0883dd6529
    


# #### Create Embedding layer

# In[49]:


def make_embedding():
    #create layer: input->filter->maxpooling->dense 105x105 size required by the paper but we use 100x100 since out inputsize is 100x100
    #create input layer
    inp=Input(shape=(100,100,3),name='input_image')
    
    # Convolutional layer conv2d: Feature layer
    c1=Conv2D(64,(10,10), activation='relu')(inp) # connect with input layer
    
    # Maxpooling layer
    m1=MaxPooling2D(64,(2,2),padding='same')(c1) # set padding as same:It is used to make the dimension of output same as input.
    
    # multiple layer, through adding feature layer and pooling layer
    c2=Conv2D(128,(7,7), activation='relu')(m1)
    m2=MaxPooling2D(64,(2,2),padding='same')(c2)
    
    # third block
    c3=Conv2D(128,(4,4), activation='relu')(m2)
    m3=MaxPooling2D(64,(2,2),padding='same')(c3)
    
    # Final block
    c4=Conv2D(256,(4,4), activation='relu')(m3)
    # flatten layer
    f1=Flatten()(c4)
    # Dense layer: combine all the layer into single dimension
    d1=Dense(4096,activation='sigmoid')(f1)
    
    
    #compelling model model.summary() tells information layer and input size for each layer
    return Model(inputs=[inp],outputs=[d1],name='embedding')


# In[50]:


embedding=make_embedding()
embedding.summary()


# #### Build Distance layer

# In[20]:


# we need to join layer that processing positive/negative and anchor together
# through subtracting we calculate their differences or similarity


# #### Create Siamese L1 Distance Class

# In[51]:


# customize layer--general template
# 'layer' class is from tf pacakages, contain all the basic layer
class L1Dist(Layer):
    def __init__(self, **kwargs): # keyword argument
        super().__init__()
     # calculate similaroty   
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


# ### Build Siamese Model
# ##### first use CNN to deal with 2 different images, then use distance layer to calculate similarity

# In[52]:


def make_siamese_model():
    
    #Anchor imhe input in the network
    Input_image = Input(name='input_img', shape=(100,100,3))
    
    # validation image in the network
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    # combine siamse distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance' # model summary will show its name
    distances = siamese_layer(embedding(Input_image),embedding(validation_image))
    
    #Classification layer
    Classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[Input_image, validation_image], outputs=Classifier, name='SiameseNetwork')


# In[53]:


siamese_model=make_siamese_model()
siamese_model.summary()


# ### Training

# In[54]:


# Set up loss function
binary_cross_loss=tf.losses.BinaryCrossentropy()

#set up optimizer
opt= tf.keras.optimizers.Adam(1e-4) # learning rate in () 0.0001


# ### Set up Checkpoints

# In[55]:


checkpoint_dir='./training_checkpoints' # ./ : in current folder, 
checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt') # all checkpoints will start with ckpt
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model = siamese_model) 


# Basic flow for training on one batch is:
# 1.make a prediction
# 2.calculate loss
# 3.derive gradients
# 4.calculate new weights and apply

# In[56]:


@tf.function # compiles a function into a callable tf graph
def train_step(batch): # each batch contains 16 examples
    # record operations from automatic differentiation: record every things in the NNmodel
    with tf.GradientTape() as tape:
        
        #Get anchor and positive/negative image from train data
        X = batch[:2]
        
        #get label
        Y= batch[2]
        
        # forward pass
        yhat=siamese_model(X, training=True) # for activate certain layer when training, while in actual prediction, training dont need to be set as true
        
        #calcualte loss
        loss = binary_cross_loss(Y,yhat)
    print(loss)  
    
    # calculate gradients:Gradient Descent method aims to focus on optimising mathematical model and loss function,
    #which means amending the factors of original functions and reduce the value of loss functions.
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    # calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    
    # return loss
    return loss


# #### Build up training loop----- train every batch in the data set

# In[57]:


# import metric calculations
from tensorflow.keras.metrics import Precision, Recall


# In[68]:


def train(data, EPOCHS):
    # loop through epochs
    for epoch in range(1,EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch,EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data)) # progress bar
        
        # Creating a metric object
        
        r = Recall()
        p = Precision()
    
        #loop through each batch
        for idx, batch in enumerate(data):
            # run train step here
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat)
            progbar.update(idx+1)
        print('Loss:{}'.format(loss.numpy()), 'Recall:{}'.format(r.result().numpy()), 'Precision:{}'.format(p.result().numpy()))

        # save checkpoints for every 10 epochs (optional)
        if epoch % 10 ==0:
            checkpoint.save(file_prefix=checkpoint_prefix)


# ### Train model

# In[69]:


# try 20 epochs,using train data
train(train_data, 20)


# ### Evaluate Model

# In[22]:


# import metric calculations
from tensorflow.keras.metrics import Precision, Recall


# In[23]:


# get a batch of test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()


# In[24]:


# make predictions
y_hat = siamese_model.predict([test_input,test_val])
y_hat 


# In[25]:


# set up threshold for classification, get binary outcome
pre=[1 if prediction > 0.5 else 0 for prediction in y_hat]


# In[26]:


# Create metrics object
R = Recall()
#calculating recall value
R.update_state(y_true, y_hat)
# Return result
R.result().numpy()


# In[27]:


# Create Precision metrics
P = Precision()
#calculating recall value
P.update_state(y_true, y_hat)
# Return result
P.result().numpy()


# In[ ]:


r = Recall()
p = Precision()

#loop through each batch
for test_input, test_val, y_true in test_data.as_numpy_iterator():
    # run train step here
    yhat = siamese_model.predict(test_input, test_val)
    r.update_state(y_true, yhat)
    p.update_state(y_true, yhat)
print('Recall:{}'.format(r.result().numpy()), 'Precision:{}'.format(p.result().numpy()))


# In[28]:


# Visualize a pair of  result
plt.figure(figsize=(10,8))
#set first subplot
plt.subplot(1,2,1) # row,col,index
plt.imshow(test_input[0]) # could change different pair to see result diff
# set second subplot
plt.subplot(1,2,2)
plt.imshow(test_val[0])

# show result
plt.show()


# ### Save Model

# In[29]:


# save weights
siamese_model.save('siamesemodel.h5')


# In[30]:


# reload model with customized layer, currently they are all in the same notebook
model=tf.keras.models.load_model('siamesemodel.h5',
                                custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})


# In[31]:


model.predict([test_input,test_val])


# #### Real Time test

# #### Verification Funciton

# acess webcam->get input image->50 predictions to verify positive sample each time(1 verification cycle use 50 images from positive)->pass image into model->verification threshold at 50% and classdify->detection threshold(ie. if 30 images threshold pass over 50 images,to make sure at least 50% of image pass the verification threshold

# In[32]:


app_data_path='path_to_application_data'
ver_path = 'path_to_verification_image'
input_path= 'path_to_input_image'


# In[35]:


np.array


# In[33]:


def verify(model, detection_threshold, verification_threshold):
    #build results arrary
    results=[]
    for image in os.listdir(ver_path):
        # get the input image from webcam(screen shot) put into input file, named as input_image.jpg
        input_img = preprocess(os.path.join(input_path, 'input_image.jpg'))
        validation_img =  preprocess(os.path.join(ver_path, image))

        #make predictions: single sample need to rapps sample, multiple no need to rapping into a list
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1))) # extend input dimensions into 2
        results.append(result)
        
    # detection threshold: metric above which a prediction is considered positive    
    detection = np.sum(np.array(results) > detection_threshold) # sum up all passed results
    
    # verfication threshold: proportion of positive prediction/ total positive sample
    verification = detection / len(os.listdir(ver_path)) #proportion
    verified = verification > verification_threshold #true or false
    
    return results, verified


# ### OpenCV Real Time Verification

# In[55]:


# Established a connection to the webcam, pos and an picture come from open sources

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #cut Down the image frame to 250x250px
    frame=frame[120:120+250,200:200+250,:]

    # Display the resulting frame
    cv2.imshow('Verification',frame)
    
    #verification trigger (when hit the keyboard verify me)
    if cv2.waitKey(10) & 0XFF == ord('v'):
        # save input image to input_image folder
        cv2.imwrite(os.path.join(input_path, 'input_image.jpg'), frame)
        
        #verification function
        results, verified=verify(model,0.9, 0.7) # will need to adjust ratio to imporve varificaiton rate
        print(verified)
    
    if cv2.waitKey(10) & 0XFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
        





