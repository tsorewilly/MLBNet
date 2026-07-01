from nets.jsr_unet_multi import jsr_unet, jr_unet
from nets.Segnet import Segnet
from nets.helper_functions import U_Net, UNetPlusPlus

from lossFcns import *
from flop import generate_arrays_from_file, get_d2_flops
from metricsold import Iou_score, f_score
from draw_confusion_matrix import plot_confusion_matrix_from_data

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.tensorflow.keras.utils.data_utils import get_file

import math
import random
from sklearn import metrics
import skimage.io as io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import os
import subprocess as sp 
from os.path import join
import copy
import datetime
import time
from itertools import cycle
from lossFcns import *
#from tensorflow.tensorflow.keras.utils.data_utils import get_file


#Decide to use current GPU or iterate to next available
try:
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i,x in enumerate(memory_free_info)]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    usable_gpus = [i for i, x in enumerate(memory_free_values) if x > 3000]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(usable_gpus[0])
    """
    if memory_free_values[0] < 1000:
        #tf.device('/device:GPU:1')
    """
except:
    print("nvidia-smi is probably not installed")

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
K.set_image_data_format('channels_last')

NCLASSES, HEIGHT, WIDTH, HEIGHT1, WIDTH1 = 1, 256, 256, 256, 256

def generate_arrays_from_file(batchFile, lines, batch_size, aug = None):    
    n, i = len(lines), 0    
    while 1:
        X_train, Y_train = [], []
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)                
            name = lines[i].split(';')[0]
            
            # read images in the training set from jpg folder
            img = Image.open(r"./"+batchFile+"/images" + '/' + name)
            img = img.resize((WIDTH,HEIGHT))
            if len((np.array(img)).shape) == 3:
                tmp_im_array = np.array(img)[:,:,0]
            else:
                tmp_im_array = np.array(img)
            
            tmp_im_array = tmp_im_array/255
            tmp_im_array = tmp_im_array[np.newaxis,:,:]

            #Split file entries, trim components to read masks from png folder
            name = (lines[i].split(';')[1]).replace("\n", "")             
            img = Image.open(r"./"+batchFile+"/labels" + '/' + name)
            img = img.resize((int(WIDTH1),int(HEIGHT1)))
            if len((np.array(img)).shape) == 3:
                tmp_lb_array = np.array(img)[:,:,0]
            else:
                tmp_lb_array = np.array(img)
                
            tmp_lb_array = tmp_lb_array/255
            tmp_lb_array = tmp_lb_array[np.newaxis,:,:] #Class #1

            if len(X_train) == 0:
                X_train = tmp_im_array
                Y_train = tmp_lb_array
            else:
                X_train = np.concatenate((X_train,tmp_im_array),axis=0)
                Y_train = np.concatenate((Y_train,tmp_lb_array),axis=0)

            i = (i+1) % n

        #Store images and masks. PS: one more dimension is added
        X_train = X_train[:,:,:,np.newaxis]
        Y_train = Y_train[:,:,:,np.newaxis] 

        if (aug is not None):# and (random.random() > 0.5):
            #aug.fit(X_train)
            X_train = next(aug.flow(X_train,batch_size = batch_size,shuffle=False,seed = i))
            Y_train = next(aug.flow(Y_train,batch_size = batch_size,shuffle=False,seed = i))

        # Return data in batchsize, reshuffle after each return round
        #return [X_train, Y_train]
        yield(X_train, Y_train)
        
def fast_hist(a, b, n):
    # a is the label, shape (H×W, converted into a one-dimensional array;
    # b is the label converted into a one-dimensional array, shape (H×W, )
    k = (a >= 0) & (a < n)  # Filter out pixels that are not in the category
    # returning the value shape (n, n)
    print(n, n * a[k], n * a[k].astype(int), n * a[k].astype(int) + b[k])
    count = 0
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    # The value on matrix diagonal consists of a one-dimensional array/sum of all elements of the matrix,
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))     # returning the value shape (n,)

def compute_binaryMetrics(sample, samplelabel, NCLASSES):    
    cm1 = metrics.confusion_matrix(sample, samplelabel)
    accu = (cm1[0,0]+cm1[1,1])/sum(sum(cm1))        
    sens = cm1[0,0]/(cm1[0,0]+cm1[0,1])    
    spec = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    mcc = metrics.matthews_corrcoef(sample, samplelabel)

    precision = dict()
    recall = dict()
    pr_ap = dict()
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(NCLASSES):
        precision[i], recall[i], _ = metrics.precision_recall_curve(samplelabel[:, i], sample[:, i]) #precision_recall_curve roc_curve
        pr_ap[i] = metrics.average_precision_score(samplelabel[:, i], sample[:, i],average='weighted')
        fpr[i], tpr[i], _ = metrics.roc_curve(samplelabel[:, i], sample[:, i])
        #roc_auc[i] = metrics.roc_auc_score(samplelabel[:, i], sample[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    #print(roc_auc)
    area = np.trapz(y=precision[i][::-1], x=recall[i][::-1])
    # print(fpr[i],tpr[i])
    precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(samplelabel.ravel(),sample.ravel())
    pr_ap["micro"] = metrics.average_precision_score(samplelabel.ravel(),sample.ravel())
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(samplelabel.ravel(), sample.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    
    return cm1, accu, sens, spec, mcc, area, precision, recall, fpr, tpr, pr_ap, roc_auc

def compute_mIoU(gt_dir, pred_dir, save_dir, gw_grp, png_name_list, num_classes, name_classes):
    # A function that computes mIoU
    print('Num classes', num_classes)
    ## 1
    hist = np.zeros((num_classes, num_classes))

    # Get a list of tag paths for the validation set for easy direct reading
    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]

    # Get the list of verification set image segmentation result paths for direct reading
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    # Read each (image-tag) pair
    for ind in range(len(gt_imgs)):
        # Read segmentation result and convert it to numpy array
        pred = np.array(Image.open(pred_imgs[ind]))//255
        # Read corresponding tag and convert it to a numpy array
        img = Image.open(gt_imgs[ind])
        img = img.resize((int(WIDTH), int(HEIGHT)))#, resample=Image.BICUBIC)
        img = img.resize((int(WIDTH), int(HEIGHT)), resample=Image.ANTIALIAS) #BICUBIC
                
        #08-02-2024        Added to address cases of 2D labels
        img = np.array(img)
        if len(img.shape) == 2:
            img = img.reshape(HEIGHT, WIDTH, 1)
        
        label = np.array(img)[:, :, 0]
        
        imga = Image.fromarray(label * (WIDTH-1))
        imga.save(r"./"+save_dir+"/{}.png".format(png_name_list[ind] + "origin"))
        imgb = Image.fromarray(pred * (WIDTH-1))
        imgb.save(r"./"+save_dir+"/{}.png".format(png_name_list[ind] + "predict"))
        # If segmentation result and label size are not the same, the image is not counted
        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue
        # Calculate the histogram matrix of 19×19 on an image and accumulate
        hist += fast_hist(label.flatten(), pred.flatten(),num_classes)

        # For 10 images,output average mIoU of all categories
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.mean(per_class_iu(hist))))
    return hist


def figPlot(y_true, y_pred, saveAs='foo.png', title=''):
    print(metrics.classification_report(y_true, y_pred))
    print("Accuracy: {0}".format(metrics.accuracy_score(y_true, y_pred)))
    plot_confusion_matrix_from_data(y_true, y_pred, columns=['BG', 'GW'], annot=True, cmap='Blues', 
                fmt='.2f', fz=20, lw=0.5, cbar=False, figsize=[6, 6], show_null_values=2, 
                pred_val_axis='y', SaveFig = 1, saveAs=saveAs, title=title)


aug = ImageDataGenerator( #define a data augmentation generator
     rotation_range = 45,       #define the rotation range
     zoom_range = 0.2,          #Randomly scale the image size proportionally
     width_shift_range = 0.2,   #image horizontal shift range
     height_shift_range = 0.2,  #vertical shift range of the image     
     shear_range = 45,          #horizontal or vertical projection transformation
     horizontal_flip = True,    #Flip the image horizontally
     fill_mode = 'nearest', #"reflect",     #fill pixels, appear after rotation or translation
     #brightness_range=(1.0, 1.0),#(0.1, 0.3),
     validation_split = 0.2
)

class LossHistory(tensorflow.keras.callbacks.Callback):
    def __init__(self):

        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')
        log_dir = "logs/"
        self.log_dir    = log_dir
        self.time_str   = time_str
        self.save_path  = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses     = []
        self.val_loss   = []
        self.accuracy   = []
        self.val_acc    = []
        self.binary_crossentropy = []
        self.val_binary_crossentropy = []
        self.dice_coef   = []
        self.val_dice_coef    = []

        os.makedirs(self.save_path)

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('main_loss'))
        self.val_loss.append(logs.get('val_main_loss'))
        self.accuracy.append(logs.get('main__f_score'))
        self.val_acc.append(logs.get('val_main__f_score'))
        self.binary_crossentropy.append('main_binary_crossentropy')
        self.val_binary_crossentropy.append('val_main_binary_crossentropy')
        self.dice_coef.append('main_dice_coef')
        self.val_dice_coef.append('val_main_dice_coef')        
        
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('main_loss')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('val_main_loss')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_accuracy_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('main__f_score')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_acc_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('val_main__f_score')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_binary_crossentropy_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('main_binary_crossentropy')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_binary_crossentropy_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('val_main_binary_crossentropy')))
            f.write("\n")            
        with open(os.path.join(self.save_path, "epoch_main_dice_coef_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('main_dice_coef')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_main_dice_coef_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('val_main_dice_coef')))
            f.write("\n")
    
log_dir = "logs/"    
#model = jr_unet(n_classes=NCLASSES, input_height=HEIGHT, input_width=WIDTH)    
model = jsr_unet(n_classes=NCLASSES, input_height=HEIGHT, input_width=WIDTH, n_decoders=1)
#Output the parameter status of each layer of the model
model.model_name = "JSR_Unet"
data_dir = "ct_EXP" #"gw_Pig_RX"# "gw_TMI" #"gw_Phantom", ct_EXP, gw_EXP


with open(r"./"+data_dir+"/train.txt", "r") as f:
    lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None) #Reset the randomization algorithm to always arrange data in same manner every run

#divide data into training and validation (90:10)
num_val = int(len(lines)*0.1)
num_train = len(lines) - num_val

history = LossHistory()
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)

model.compile(
    loss = bce_dice_loss,
    #new_xy_loss_v44, CE_loss, tversky_loss, dice_loss, dice_coef_loss, focal_dice_loss, 
    #    binary_focal_loss_fixed, bce_dice_loss, jaccard, huber_loss
    optimizer = Adam(lr=0.0001),
    metrics = [f_score(), "binary_crossentropy", dice_coef])
model.summary()    
    
batch_size, epochs = 2, 200
print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

model.fit(generate_arrays_from_file(data_dir, lines[:num_train], batch_size,aug),
                    steps_per_epoch=max(1, num_train//batch_size),
                    validation_data=generate_arrays_from_file(data_dir, lines[num_train:], batch_size),
                    validation_steps=max(1, num_val//batch_size),
                    epochs=epochs, initial_epoch=0,  callbacks=[checkpoint_period, reduce_lr, history])

decodNum=1
#model.save_weights(log_dir + 'last1-MLBS-Rab-Base-Ep_50-90%Cmp_Loss_44(.45,.15).h5')    
saveFile = ''+model.model_name+'-Ep:'+str(epochs)+'-'+data_dir+'-'+datetime.datetime.now().strftime("%m:%d:%H%")
model.save_weights('PlotsOutput/Models/Full-'+saveFile+'.h5')    

# In[] Analysis and plotting     
iters = range(len(history.losses))

plt.figure()
plt.plot(iters, history.losses, 'red', linewidth = 2, label='train loss')
plt.plot(iters, history.val_loss, 'coral', linewidth = 2, label='val_loss')    
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Plot')
plt.legend(loc="upper right")  
plt.show()  

plt.figure()
plt.plot(iters,history.accuracy, 'black', linewidth = 2, label='accuracy')
plt.plot(iters,history.val_acc, 'blue', linewidth = 2, label='val_acc')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Accuracy Curve')
plt.legend(loc="lower right")    
plt.show()

#In[] EVALUATION DONE HERE 
# Set the label width W, long H


model.load_weights('model/Full-JSR_Unet-Ep:200-gw_EXP-02:24:22%.h5')
pr_thresh = 0.25

  
gw_grp = data_dir+"/"
imgs = r""+gw_grp+"images/"
labelimgs = r""+gw_grp+"labels/"
pred_dir = "./miou_pr_dir/"+gw_grp   
gt_dir = "./"+gw_grp+"labels"    
save_dir = gw_grp+"save_"+((model.model_name).split("_"))[-1]
png_name_list = open(r""+gw_grp+"test.txt", 'r').read().splitlines()

if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

num_classes = 2
name_classes = ["BG", "GW"]

# Executes a function that computes mIoU
tpf, fps, i = 0, 0, 0

start_time = time.time()
x=1 #display the frame rate every 1 second
counter = 0

for jpg in png_name_list:

    img = Image.open(imgs + jpg + '.jpg')
    labelimg = Image.open(labelimgs + jpg + '.png')
    img = img.resize((WIDTH,HEIGHT))
    if len((np.array(img)).shape) == 3:
        img = (np.array(img)[:,:,0])/255
    else:
        img = (np.array(img))/255
    img = img[np.newaxis,:,:]

    labelimg = np.array(labelimg.resize((WIDTH,HEIGHT))) 
    if len((np.array(labelimg)).shape) == 2:
            labelimg = labelimg.reshape(HEIGHT, WIDTH, NCLASSES)       
    seg_labels = np.zeros((int(HEIGHT), int(WIDTH), NCLASSES))
    
    for c in range(NCLASSES):
        seg_labels[:, :, c] = (labelimg[:, :,0] == c+1).astype(int)
    
    seg_labels = np.reshape(seg_labels, (-1, NCLASSES))
    
    t1 = time.time()
    pr = np.array(model.predict(img)[0]).reshape(HEIGHT, WIDTH, NCLASSES)
    t2 = time.time()
    tpf += t2 - t1
    fps += (1. / (t2 - t1))
    
    pr[pr > pr_thresh] = 1
    pr[pr <= pr_thresh] = 0        
    pr1 = copy.deepcopy(pr)

    if i == 0:
        sample, samplelabel = pr1, seg_labels
    else:
        sample = np.concatenate([sample,pr1],axis=0)
        samplelabel = np.concatenate([samplelabel, seg_labels], axis=0)
    i=1
    io.imsave(pred_dir + jpg + ".png", pr)#.astype(np.uint8))
#fps = fps/len(png_name_list)
#tpf = tpf/len(png_name_list)
#In[] COMPUTE BINARY METRICS FOR QUANTITATIVE PERFORMANCE ANALYSIS    

# Executes a function that computes mIoU
hist = compute_mIoU(gt_dir, pred_dir, save_dir, gw_grp, png_name_list, num_classes, name_classes)
    # Calculate the category-by-category mIoU value for all validation set pictures    

pred = np.reshape(sample,(-1,NCLASSES))
cm1, accu, sens, spec, mcc, area, precision, recall, fpr, tpr, pr_ap, roc_auc = compute_binaryMetrics(pred, samplelabel, NCLASSES)

mIoUs = per_class_iu(cm1)
# Output the mIoU values category by category
for ind_class in range(num_classes):
    print('===>' + name_classes[ind_class] + '\t' + str(round(mIoUs[ind_class] * 100, 2)))
# Average mIoU values for all categories on all validation set images, ignoring NaN values when calculating
print('===>mIoU\t ' + str(round(np.nanmean(mIoUs) * 100, 2))) 

#In[] Analysis and plotting  
figPlot(samplelabel, pred, saveAs='PlotsOutput/CF-Matrix/'+saveFile+'.png', title=saveFile)

TP, FP, FN, TN = (np.reshape(cm1, (4,1)).astype(int))
DSC = (TP[0]+TP[0])/(TP[0]+TP[0]+FP[0]+FN[0])
F1s = 2 * (precision['micro'][1] * recall['micro'][1]) / (precision['micro'][1] + recall['micro'][1])

plt.title('Receiver Operating Characteristic')
plt.plot(fpr['micro'], tpr['micro'])
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.ylabel('True Positive Rate(Sensitivity)')
plt.xlabel('False Positive Rate(Specificity)')
plt.savefig(('PlotsOutput/AROC/'+saveFile+'.png'))
plt.show()


lw=2
plt.figure()
plt.plot(precision["micro"], recall["micro"], color='deeppink', linestyle=':', linewidth=4)
color = cycle(['red','black','gold','green','purple','hotpink','aqua', 'darkorange', 'cornflowerblue'])
linestyle = cycle([(0, ()), (0, (1, 1)), (0, (5,5)),'-.',(0, (5, 1)),(0, (3, 1, 1, 1)),(0, (3, 1, 1, 1, 1, 1)),(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1,1,1))])

x_major_locator = plt.MultipleLocator(0.2)
y_major_locator = plt.MultipleLocator(0.2)  
ax=plt.gca() 
ax.yaxis.set_major_locator(y_major_locator) 
ax.xaxis.set_major_locator(x_major_locator)

plt.ylim(0,1.05)
plt.yticks(fontproperties = 'Times New Roman',fontsize=20)
plt.xticks(fontproperties = 'Times New Roman',fontsize=20)

for i, cla, linestyle, color in zip(range(NCLASSES), name_classes, linestyle, color):
    plt.plot(recall[i], precision[i], linestyle=linestyle, color =color, lw=lw, 
             label='PR curve of {0} (area = {1:0.3f})'.format(cla, pr_ap[i]))

font = font_manager.FontProperties(family='Times New Roman', size=15)
plt.xlabel('Recall',fontproperties = 'Times New Roman',fontsize=20)
plt.ylabel('Precision',fontproperties = 'Times New Roman',fontsize=20)
plt.legend(loc="lower right",prop = font)
plt.savefig(('PlotsOutput/BG-Area/'+saveFile+'.png'))
plt.show()
