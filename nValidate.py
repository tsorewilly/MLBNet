#from nets.Repunet import convnet_unet
# In[] LOAD MODULES TO USE
from nets.old_jsr_unet import jsr_unet, js_unet
#from nets.helper_functions import U_Net, wU_Net, UNetPlusPlus


import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import subprocess as sp   
import random
import copy
import os
import time
import tensorflow.keras.backend as K
from sklearn import metrics
from itertools import cycle
import tensorflow as tf
import skimage.io as io
from draw_confusion_matrix import plot_confusion_matrix_from_data
import warnings

try:
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i,x in enumerate(memory_free_info)]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    usable_gpus = [i for i, x in enumerate(memory_free_values) if x > 1000]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(usable_gpus[0])
    """
    if memory_free_values[0] < 1000:
        #tf.device('/device:GPU:1')
    """
except:
    print("nvidia-smi is probably not installed")

warnings.filterwarnings("ignore")
K.clear_session()
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# Set the label width W, long H
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)  # Filter out pixels that are not in the category
    print(n, n * a[k], n * a[k].astype(int), n * a[k].astype(int) + b[k])
    count = 0
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

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

def compute_mIoU(gt_dir, pred_dir, gw_grp, png_name_list, num_classes, name_classes):
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
        pred = np.array(Image.open(pred_imgs[ind])) // 255
        # Read corresponding tag and convert it to a numpy array
        img = Image.open(gt_imgs[ind])
        img = img.resize((int(256), int(256)), resample=Image.BICUBIC)
        label = np.array(img)[:, :, 0]
        imga = Image.fromarray(label * 255)
        imga.save(r"./"+gw_grp+"save/{}.png".format(png_name_list[ind] + "origin"))
        imgb = Image.fromarray(pred * 255)
        imgb.save(r"./"+gw_grp+"save/{}.png".format(png_name_list[ind] + "predict"))
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
    #print(metrics.classification_report(y_true, y_pred))
    #print("Accuracy: {0}".format(metrics.accuracy_score(y_true, y_pred)))

    plot_confusion_matrix_from_data(y_true, y_pred, columns=['BG', 'GW'], annot=True, cmap='Blues', 
                fmt='.2f', fz=20, lw=0.5, cbar=False, figsize=[6, 6], show_null_values=2, 
                pred_val_axis='y', SaveFig = 1, saveAs=saveAs, title=title)
 
#In[] DEPLOY MODEL FOR SEGMENTATION AND PERFORMANCE ANALYSIS 
if __name__ == "__main__":
    gw_grp = "gw_TMI/"
    imgs = r""+gw_grp+"jpg/"
    labelimgs = r""+gw_grp+"png/"
    pred_dir = "./miou_pr_dir/"+gw_grp   
    gt_dir = "./"+gw_grp+"png"    
    png_name_list = open(r""+gw_grp+"test_90%.txt", 'r').read().splitlines()
    
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.exists(gw_grp+"save"):
        os.makedirs(gw_grp+"save")
    
    num_classes = 2
    name_classes = ["BG", "GW"]
    # Executes a function that computes mIoU
    

    class_colors = [[0,0,0],[0,255,0]]
    NCLASSES = 1
    HEIGHT, WIDTH, HEIGHT1, WIDTH1 = 256, 256, 256, 256
    
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
    
    Ep = 50
    Lss = '90%FD'
    LR = '0.0001'
    modelName = 'JS-UNet-'
    #last1-MLBS-Rab-Base-Ep_50-90%myHuber_Loss 
    #last1-MLBS-Rab-Base-Ep_50-Jaccard_Loss
# In[] SEGMENTATION DONE HERE        
    model = js_unet(n_classes=NCLASSES, input_height=HEIGHT, input_width=WIDTH)            
    model.load_weights('PlotsOutput/Models/Full-JSR_Unet-Rab-Base-Ep_100-90%Cmp_Loss.h5')

# In[] EVALUATION DONE HERE 
    tpf, fps, i = 0, 0, 0
    
    start_time = time.time()
    x=1 #display the frame rate every 1 second
    counter = 0
    
    for jpg in png_name_list:
        img = Image.open(imgs + jpg + '.jpg')
        labelimg = Image.open(labelimgs + jpg + '.png')
        img = img.resize((WIDTH,HEIGHT))
        #img = (np.array(img)[:,:,0])/255 #Replaced this line with the if                
        if len((np.array(img)).shape) == 3:
            img = np.array(img)[:,:,0]/255
        else:
            img = np.array(img)/255        
        
        img = img[np.newaxis,:,:,np.newaxis]
    
        labelimg = labelimg.resize((WIDTH1,HEIGHT1))
        labelimg = np.array(labelimg)
        #if len(labelimg.shape) == 2:
        #    labelimg = labelimg[:,:,np.newaxis]
            
        seg_labels = np.zeros((int(HEIGHT1), int(WIDTH1), NCLASSES))
        for c in range(NCLASSES):
            seg_labels[:, :, c] = (labelimg[:, :,0] == c+1).astype(int)
        
        seg_labels = np.reshape(seg_labels, (-1, NCLASSES))
        
        t1 = time.time()
        #pr = (model.predict(img)[2])[0]
        pr = (model.predict(img)[0]) #Validate Unet and Others
        
        t2 = time.time()
        tpf_ds = t2 - t1
        fps_ds = (1. / (t2 - t1))
        
        #print(tpf_ds, fps_ds)
                
        tpf += tpf_ds
        fps += fps_ds
                
        
        pr[pr > 0.5] = 1
        pr[pr <= 0.5] = 0        
        pr1 = copy.deepcopy(pr)
    
        if i == 0:
            sample = pr1
            samplelabel = seg_labels
        else:
            sample = np.concatenate([sample,pr1],axis=0)
            samplelabel = np.concatenate([samplelabel,seg_labels],axis=0)
        i=1
        io.imsave(pred_dir + jpg + ".png", pr[0])#.astype(np.uint8)) 

# In[] COMPUTE BINARY METRICS FOR QUANTITATIVE PERFORMANCE ANALYSIS    
    num_classes = 2
    name_classes = ["BG", "GW"]
    
    # Executes a function that computes mIoU
    hist = compute_mIoU(gt_dir, pred_dir, gw_grp, png_name_list, num_classes, name_classes)
        # Calculate the category-by-category mIoU value for all validation set pictures    
    
    pred = np.reshape(sample,(-1,NCLASSES))
    cm1, accu, sens, spec, mcc, area, precision, recall, fpr, tpr, pr_ap, roc_auc = compute_binaryMetrics(pred, samplelabel, NCLASSES)

    mIoUs = per_class_iu(cm1)
    # Output the mIoU values category by category
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + '\t' + str(round(mIoUs[ind_class] * 100, 2)))
    #Average mIoU values for all categories on all validation set images, ignoring NaN values when calculating
    print('===>mIoU\t ' + str(round(np.nanmean(mIoUs) * 100, 2))) 

#In[] Analysis and plotting   
    saveFile ='PlotsOutputTMI/last1-'+modelName+'-Rab-Base-LR_'+str(LR)+'-Ep_'+str(Ep)+'-'+str(Lss)+'.png'
    figPlot(samplelabel, pred, saveAs=saveFile, title=modelName+'-Rab-Base-LR_'+str(LR)+'-Ep_'+str(Ep)+'-'+str(Lss))
    
    TP, FP, FN, TN = (np.reshape(cm1, (4,1)).astype(int))
    DSC = (TP[0]+TP[0])/(TP[0]+TP[0]+FP[0]+FN[0])
    F1s = 2 * (precision['micro'][1] * recall['micro'][1]) / (precision['micro'][1] + recall['micro'][1])
    print(F1s)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr['micro'], tpr['micro'])
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel('True Positive Rate(Sensitivity)')
    plt.xlabel('False Positive Rate(Specificity)')
    plt.show()
    
    lw=2
    plt.figure()
    plt.plot(precision["micro"], recall["micro"], 
             #label='Micro-avg PR curve (area = {0:0.2f})'.format(pr_ap["micro"]), 
                   color='deeppink', linestyle=':', linewidth=4)
    
    #plt.plot(fpr["macro"], tpr["macro"], label='macro-avg ROC curve (area = {0:0.2f})'
    #         ''.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)
    
    color = cycle(['red','black','gold','green','purple','hotpink','aqua', 'darkorange', 'cornflowerblue'])
    #linestyle = cycle(['-', '--', '-.',':','solid','o',' ','dashed','dashdot'])
    linestyle = cycle([(0, ()), (0, (1, 1)), (0, (5,5)),'-.',(0, (5, 1)),(0, (3, 1, 1, 1)),(0, (3, 1, 1, 1, 1, 1)),(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1,1,1))])
    #marker =  cycle(['*',',','o','v','^','<','>','s','p','h','H','+','x','D','d','|','_','.',','])
    #linestyle = cycle()
    #linestyle = cycle([':'])
    classes = ["BG", "GW"]
    x_major_locator = plt.MultipleLocator(0.2)
    y_major_locator = plt.MultipleLocator(0.2)  
    ax=plt.gca() 
    ax.yaxis.set_major_locator(y_major_locator) 
    ax.xaxis.set_major_locator(x_major_locator)
    plt.ylim(0,1.05)
    plt.yticks(fontproperties = 'Times New Roman',fontsize=20)
    plt.xticks(fontproperties = 'Times New Roman',fontsize=20)
    
    for i, cla, linestyle, color in zip(range(NCLASSES), classes, linestyle, color):
        plt.plot(recall[i], precision[i], linestyle=linestyle, color =color, 
                 lw=lw, label='PR curve of {0} (area = {1:0.3f})'
                       ''.format(cla, pr_ap[i]))
    print(pr_ap)
    #plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    font = font_manager.FontProperties(family='Times New Roman', size=15)
    #legend_font = {"family" : "Times New Roman"}
    plt.xlabel('Recall',fontproperties = 'Times New Roman',fontsize=20)
    plt.ylabel('Precision',fontproperties = 'Times New Roman',fontsize=20)
    # plt.xlabel('False positive rate',fontproperties = 'Times New Roman',fontsize=20)
    # plt.ylabel('True positive rate',fontproperties = 'Times New Roman',fontsize=20)
    #plt.title('Some extension of Receiver operating characteristic to multi-class')
    #plt.legend.get_title().set_fontsize(fontsize = 15)
    plt.legend(loc="lower right",prop = font)
    plt.show()
    
