
import numpy
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
from osgeo import gdal
import torch
import os
import cv2
import torch
import torch.nn.functional as F
import matplotlib as mpl
import scipy
from kornia.core import Module, Tensor, pad
from kornia.filters import sobel
from PIL import Image
import numpy as np




from matplotlib.pyplot import MultipleLocator
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def load_mat(path):
    data = sio.loadmat(path)
    data_dict = []
    for key, value in data.items():
        data_dict.append(key)
    return data, data_dict


def get_rgb_normalized(r, g, b):
    max_r = np.max(np.max(r, axis=1), axis=0)
    max_g = np.max(np.max(g, axis=1), axis=0)
    max_b = np.max(np.max(b, axis=1), axis=0)
    r = r/max_r*255.0
    g = g/max_g*255.0
    b = b/max_b*255.0
    img_rgb = np.stack((b, g, r), axis=2)
    img_rgb[img_rgb<0.0]=0.0
    return img_rgb

def get_rgb_normalized1(r, g, b):
    img_rgb = np.stack((b, g, r), axis=2)
    return img_rgb

def get_bands_normalized(img):
    max = np.max(np.max(img, axis=1), axis=0)
    img = img/max*255.0
    return img

def save_tif(r, g, b, h, wirte_path):
    driver = gdal.GetDriverByName("GTiff")
    New_dataset = driver.Create(wirte_path, h, h, 3, gdal.GDT_Float32)
    band1 = New_dataset.GetRasterBand(1)
    band1.WriteArray(r)
    band2 = New_dataset.GetRasterBand(2)
    band2.WriteArray(g)
    band3 = New_dataset.GetRasterBand(3)
    band3.WriteArray(b)
    New_dataset.FlushCache()



def save_pred_pic(dataset):
    for model_key, model in pred_path2.items():  #
        for ds_key, dst in dataset.items():
            ds = datasets[ds_key]
            result_path = pred_dir + model + '\\eval\\' + 'evaluation.mat'
            pred_data, pred_dict = load_mat(result_path)
            for i in range(3, len(pred_dict)):

                image = pred_data[pred_dict[i]]
                h, w, c = image.shape

                b = image[:, :, ds[0]]
                g = image[:, :, ds[1]]
                r = image[:, :, ds[2]]

                save_tif(r, g, b, h, pred_dir + model + '\\eval\\' + pred_dict[i] + '.tiff')
                img_rgb = get_rgb_normalized(b, g, r)
                img_rgb = Image.fromarray(np.uint8(img_rgb))
                img_rgb.save(pred_dir + model + '\\eval\\' + pred_dict[i] + '.jpg')
                print(pred_dir + model + '\\eval\\' + pred_dict[i] + '.jpg')


def log(base,x):
    return np.log(x)/np.log(base)

def calculate_mae(pred, ref, Max=None):
    h, w, c = pred.shape
    tmp = np.abs(ref - pred)
    if Max is None:
        #max = np.max(tmp.reshape(-1, c), axis=0)
        #mae = tmp.reshape(-1, c)/max
        mae = tmp.reshape(-1, c)
        mae = np.mean(mae, axis=1)
        #mae = np.log2(mae + 1.0)
        #mae = log(2.1, mae + 1.1)
    else:
        mae = tmp.reshape(-1, c)/Max
        mae = np.mean(mae, axis=1)
        mae = np.log2(mae+1.0)
        #mae = log(2.1, mae+1.1)
    return mae.reshape(h, w)

def plot_mae(model_result, ds, d_id):
    """
    绘制MAE
    params data: 待绘制的⼆维数组
    params cmap: 绘图的⾊带
    params blabel: ⾊带的单位
    params cscale: ⽐例尺的颜⾊
    params title: 图标题
    return: None
    """
    pred_image, mat_data = get_pair_mat(model_result, ds, d_id)
    ref = mat_data["ref"]
    mae = calculate_mae(pred_image, ref)
    plt.imshow(mae, cmap='turbo', vmin=0.0, vmax=1.0)
    plt.colorbar()
    pic_name=pred_all_path[model_result]
    plt.axis('off')

    #plt.title("Mean absolute error of predicted {}".format(d_id), fontsize=12)
    plt.savefig('mae_{}_{}.jpg'.format(pic_name, d_id))
    plt.show()

def save_mae(model_result, ds, d_id):
    pred_image, mat_data = get_pair_mat(model_result, ds, d_id)
    ref = mat_data["ref"]
    mae = calculate_mae(pred_image, ref)
    mmax=mae.max()

    if d_id=='chikusei_79' or d_id=='pavia_12' or d_id=='botswana_19':
        print("{}_{}_{}".format(ds, d_id, mmax))
    plt.imshow(mae, cmap='turbo', vmin=0.0, vmax=1.0)

    plt.axis('off')
    #plt.title("Mean absolute error of predicted {}".format(d_id), fontsize=12)
    plt.savefig('C:\\fusion_temp\\draw\\MAE5\\'+ ds +'\\mae_{}_{}.jpg'.format(model_result, d_id), transparent=True, bbox_inches='tight', pad_inches=0.0)
    plt.show()