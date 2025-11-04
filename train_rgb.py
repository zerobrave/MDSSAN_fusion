
import os
import argparse
import json
import torch
import numpy as np
from torch.nn.functional import threshold, unfold
from dataloaders.HSI_datasets import *
from utils.logger import Logger
import torch.utils.data as data
from utils.helpers import initialize_weights, initialize_weights_new, to_variable, make_patches
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from models.model import MODELS
from utils.metrics import *
import shutil
import torchvision
from torch.distributions.uniform import Uniform
import sys
import copy

from scipy.io import savemat
import torch.nn.functional as F
from utils.vgg_perceptual_loss import VGGPerceptualLoss, VGG19
from utils.spatial_loss import Spatial_Loss
from utils.tensor_rotate import rotate_tensor
import torchvision.transforms as transforms
from tqdm import tqdm
from models.VAEtest import vaeLoss
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)
        
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

def get_mean_std():
    transform = transforms.Compose(
        [
      
            transforms.ToTensor(),
        ]
    )
    data_loader = torch.utils.data.DataLoader(
            __dataset__[config["train_dataset"]](
                    config, is_train=True, want_DHP_MS_HR=config["is_DHP_MS"], ), batch_size=config["train_batch_size"],
            num_workers=config["num_workers"], shuffle=True,
            pin_memory=False,drop_last=True)

    nb_samples = 0.
    channel_mean = torch.zeros(3)
    channel_std = torch.zeros(3)
    for i, data in enumerate(data_loader, 0):
        # Reading data
        _, MS_image, images, reference = data
        N, C, H, W = images.shape[:4]
        data = images.view(N, C, -1)

        channel_mean += data.mean(2).sum(0)
        channel_std += data.std(2).sum(0)
        nb_samples += N

    channel_mean /= nb_samples
    channel_std /= nb_samples
    print(channel_mean, channel_std)

def evaluate(best_model, band_set):
    d_lambda = 0.0
    d_s = 0.0
    qnr = 0.0
    best_model.eval()
    pred_dic = {}
  
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            image_dict, MS_image, PAN_image, reference = data

            if config['train_dataset'] =='pavia_dataset':
                PAN_image = PAN_image.permute(0,3,1,2)
            # Inputs and references...
            MS_image    = MS_image.float().cuda().contiguous()
            PAN_image   = PAN_image.float().cuda().contiguous()
            reference   = reference.float().cuda().contiguous()

            # Taking model output
            out = best_model(MS_image, PAN_image)

            outputs = out["pred"]
            #coarse_hsi = out['coarse_hsi']
            # Scalling
            outputs[outputs < 0] = 0.0
            outputs[outputs > 1.0] = 1.0
            outputs = torch.round(outputs*5000)
            
            if config['train_dataset'] =='road_train_dataset' or config['train_dataset'] =="dx1_dataset" :
                pred_dic.update({image_dict["hrhsi"][0].split("/")[-1][:-4]+"_pred": torch.squeeze(outputs).permute(1,2,0).cpu().numpy()})
            elif config['train_dataset'] == "mdas_dataset":
                pred_dic.update({image_dict[0]+"_pred": torch.squeeze(outputs).permute(1,2,0).cpu().numpy()})
            else:
                pred_dic.update({image_dict["imgs"][0].split("/")[-1][:-4]+"_pred": torch.squeeze(outputs).permute(1,2,0).cpu().numpy()})
                
            reference               = torch.round(reference.detach()*5000)

            # pred_dic.update({"beijing_" + str(i) + "_pred": torch.squeeze(outputs).permute(1, 2, 0).cpu().numpy()})
            # MS_image = torch.round(MS_image.detach()*config[config["train_dataset"]]["max_value"])
            # PAN_image = torch.round(PAN_image.detach()*config[config["train_dataset"]]["max_value"])

            ### Computing performance metrics ###
            # D_lambda for an image
            D_l = 0.0
            for j in range(iters):
                D_l += D_lambda(outputs, MS_image, band_set[j])
            D_l = D_l/iters

            # D_s for an image
            ds = D_s_ms(outputs, MS_image, PAN_image, config[config["train_dataset"]]["factor"])

            # QNR
            Qnr = QNR(D_l, ds)

            print(D_l)
            print(ds)
            print(Qnr)
            qnr += Qnr
            d_lambda += D_l
            d_s += ds


    #print(d_lambda)
    #print(d_s)
    #print(qnr)
    d_lambda /= len(test_loader)
    d_s /= len(test_loader)
    qnr /= len(test_loader)


    # Return Outputs
    metrics = {
        "d_lambda": float(d_lambda),
        "d_s":      float(d_s),
        "qnr":      float(qnr),
        }
    return pred_dic, metrics

# TRAIN EPOCH
def train(epoch):
    train_loss = 0.0
    model.train()
    optimizer.zero_grad()
    for i, data in enumerate(train_loader, 0):
        # Reading data
        _, MS_image, PAN_image, reference = data
        
        test = False
        # 测试图像宽高是否
        if test:
            r_img = MS_image[0, 97, :, :].squeeze(0).numpy()
            g_img = MS_image[0, 57, :, :].squeeze(0).numpy()
            b_img = MS_image[0, 23, :, :].squeeze(0).numpy()

            rgb_hsi = np.uint8(get_rgb_normalized(r_img, g_img, b_img))

            r_img = PAN_image[0, 2, :, :].squeeze(0).numpy()
            g_img = PAN_image[0, 1, :, :].squeeze(0).numpy()
            b_img = PAN_image[0, 0, :, :].squeeze(0).numpy()

            rgb_msi = np.uint8(get_rgb_normalized(r_img, g_img, b_img))

            r_ref = reference[0, 97, :, :].squeeze(0).numpy()
            g_ref = reference[0, 57, :, :].squeeze(0).numpy()
            b_ref = reference[0, 23, :, :].squeeze(0).numpy()

            rgb_ref = np.uint8(get_rgb_normalized(r_ref, g_ref, b_ref))

            plt.subplot(3, 1, 1)
            plt.imshow(rgb_msi)
            plt.title('msi')
            plt.xticks(), plt.yticks()

            plt.subplot(3, 1, 2)
            plt.imshow(rgb_hsi)
            plt.title('hsi')
            plt.xticks(), plt.yticks()

            plt.subplot(3, 1, 3)
            plt.imshow(rgb_ref)
            plt.title('ref')
            plt.xticks(), plt.yticks()

            plt.show()
            plt.close()

        
        if config['train_dataset'] =='pavia_dataset':
                PAN_image = PAN_image.permute(0,3,1,2)
        # Taking model outputs ...
        # if config["rotate"]:
        #     #MS_image = rotate_tensor(MS_image, config["rotate_angle"])
        #     PAN_image = rotate_tensor(PAN_image, config["rotate_angle"])
        #     reference = rotate_tensor(reference, config["rotate_angle"])
        if epoch==1: 
             print(MS_image.max())
             print(MS_image.min())
        MS_image    = Variable(MS_image.float().cuda()) 
        PAN_image   = Variable(PAN_image.float().cuda()) 
        
        #PAN_image = PAN_image.permute(0,3,1,2)
        #print(MS_image.shape, PAN_image.shape)
        PAN_image = PAN_image[:, 0:3, :, :]
        out         = model(MS_image, PAN_image)

        outputs = out["pred"].contiguous()

        ######### Computing loss #########
        # Normal L1 loss
        if config[config["train_dataset"]]["Normalized_L1"]:
            max_ref     = torch.amax(reference, dim=(2,3)).unsqueeze(2).unsqueeze(3).expand_as(reference).cuda()
            loss        = criterion(outputs/max_ref, to_variable(reference)/max_ref)
        else:
            loss        = criterion(outputs, to_variable(reference))

        # VGG Perceptual Loss
        if config[config["train_dataset"]]["VGG_Loss"]:
            predicted_RGB   = torch.cat((torch.mean(outputs[:, 0:config[config["train_dataset"]]["G"], :, :], 1).unsqueeze(1), 
                                        torch.mean(outputs[:, config[config["train_dataset"]]["B"]:config[config["train_dataset"]]["R"], :, :], 1).unsqueeze(1), 
                                        torch.mean(outputs[:, config[config["train_dataset"]]["G"]:config[config["train_dataset"]]["spectral_bands"], :, :], 1).unsqueeze(1)), 1)
            target_RGB   = torch.cat((torch.mean(to_variable(reference)[:, 0:config[config["train_dataset"]]["G"], :, :], 1).unsqueeze(1), 
                                        torch.mean(to_variable(reference)[:, config[config["train_dataset"]]["B"]:config[config["train_dataset"]]["R"], :, :], 1).unsqueeze(1), 
                                        torch.mean(to_variable(reference)[:, config[config["train_dataset"]]["G"]:config[config["train_dataset"]]["spectral_bands"], :, :], 1).unsqueeze(1)), 1)
            VGG_loss        = VGGPerceptualLoss(predicted_RGB, target_RGB, vggnet)
            loss            += config[config["train_dataset"]]["VGG_Loss_F"]*VGG_loss

        # Transfer Perceptual Loss
        if config[config["train_dataset"]]["Transfer_Periferal_Loss"]:
            loss += config[config["train_dataset"]]["Transfer_Periferal_Loss_F"]*out["tp_loss"]

        # Spatial loss
        if config[config["train_dataset"]]["Spatial_Loss"]:
            loss += config[config["train_dataset"]]["Spatial_Loss_F"]*Spatial_loss(to_variable(reference), outputs)
        
        # Spatial loss
        if config[config["train_dataset"]]["multi_scale_loss"]:
            loss += config[config["train_dataset"]]["multi_scale_loss_F"]*criterion(to_variable(reference), out["x13"]) + 2*config[config["train_dataset"]]["multi_scale_loss_F"]*criterion(to_variable(reference), out["x23"])

        torch.autograd.backward(loss)

        if i % config["trainer"]["iter_size"] == 0 or i == len(train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()

    writer.add_scalar('Loss/train', loss, epoch)
 
# TEST EPPOCH
def test(epoch):
    test_loss   = 0.0
    cc          = 0.0
    sam         = 0.0
    rmse        = 0.0
    ergas       = 0.0
    psnr        = 0.0
    val_outputs = {}
    model.eval()
    pred_dic = {}
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            image_dict, MS_image, PAN_image, reference = data


            if config['train_dataset'] =='pavia_dataset':
                PAN_image = PAN_image.permute(0,3,1,2)
            # Inputs and references...
            # if config["rotate"]:
            #     #MS_image = rotate_tensor(MS_image, config["rotate_angle"])
            #     PAN_image = rotate_tensor(PAN_image, config["rotate_angle"])
            #     reference = rotate_tensor(reference, config["rotate_angle"])
            MS_image    = MS_image.float().cuda().contiguous()
            PAN_image   = PAN_image.float().cuda().contiguous()
            reference   = reference.float().cuda().contiguous()
            
            #PAN_image = PAN_image.permute(0,3,1,2)
            # Taking model output
            PAN_image = PAN_image[:, 0:3, :, :]
            out     = model(MS_image, PAN_image)
            
            outputs = out["pred"].contiguous()

            # Computing validation loss
            loss        = criterion(outputs, reference)
            test_loss   += loss.item()

            # Scalling
            outputs[outputs<0]      = 0.0
            outputs[outputs>1.0]    = 1.0
            #outputs                 = torch.round(outputs*config[config["train_dataset"]]["max_value"])
            outputs                 = torch.round(outputs*5000)
            if config['train_dataset'] =='road_train_dataset' or config['train_dataset'] =="dx1_dataset" :
                pred_dic.update({image_dict["hrhsi"][0].split("/")[-1][:-4]+"_pred": torch.squeeze(outputs).permute(1,2,0).cpu().numpy()})
            elif config['train_dataset'] == "mdas_dataset":
                pred_dic.update({image_dict[0]+"_pred": torch.squeeze(outputs).permute(1,2,0).cpu().numpy()})
            else:
                pred_dic.update({image_dict["imgs"][0].split("/")[-1][:-4]+"_pred": torch.squeeze(outputs).permute(1,2,0).cpu().numpy()})
                
            #reference = torch.round(reference.detach()*config[config["train_dataset"]]["max_value"])
            reference = torch.round(reference.detach()*5000)
            ### Computing performance metrics ###
            # Cross-correlation
            cc += cross_correlation(outputs, reference)
            # SAM
            sam += SAM(outputs, reference)
            # RMSE
            rmse += RMSE(outputs/torch.max(reference), reference/torch.max(reference))
            # ERGAS
            beta = torch.tensor(config[config["train_dataset"]]["HR_size"]/config[config["train_dataset"]]["LR_size"]).cuda()
            ergas += ERGAS(outputs, reference, beta)
            # PSNR
            psnr += PSNR(outputs, reference)

    # Taking average of performance metrics over test set
    cc /= len(val_loader)
    sam /= len(val_loader)
    rmse /= len(val_loader)
    ergas /= len(val_loader)
    psnr /= len(val_loader)

    # Writing test results to tensorboard
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Test_Metrics/CC', cc, epoch)
    writer.add_scalar('Test_Metrics/SAM', sam, epoch)
    writer.add_scalar('Test_Metrics/RMSE', rmse, epoch)
    writer.add_scalar('Test_Metrics/ERGAS', ergas, epoch)
    writer.add_scalar('Test_Metrics/PSNR', psnr, epoch)

    # Images to tensorboard
    # Regenerating the final image
    
    #Normalizing the images
    
    
    #if config["model"]=="HyperPNN" or config["is_DHP_MS"]==False:
        #MS_image =  F.interpolate(MS_image, scale_factor=(config[config["train_dataset"]]["factor"],config[config["train_dataset"]]["factor"]),mode ='bilinear')
    
    #ms      = torch.unsqueeze(MS_image.view(-1, MS_image.shape[-2], MS_image.shape[-1]), 1)
    #pred    = torch.unsqueeze(outputs.view(-1, outputs.shape[-2], outputs.shape[-1]), 1)
    #ref     = torch.unsqueeze(reference.view(-1, reference.shape[-2], reference.shape[-1]), 1)
    #imgs    = torch.zeros(2*pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3])
    #for i in range(pred.shape[0]):
        #imgs[5*i]   = ms[i]
        #imgs[2*i] = torch.abs(ref[i]-pred[i])/torch.max(torch.abs(ref[i]-pred[i]))
        #imgs[2*i+1] = pred[i]
        #imgs[5*i+3] = ref[i]
        #imgs[5*i+4] = torch.abs(ref[i]-ms[i])/torch.max(torch.abs(ref[i]-ms[i]))
    #imgs = torchvision.utils.make_grid(imgs, nrow=2)
    #writer.add_image('Images', imgs, epoch)

    #Return Outputs
    metrics = { "loss": float(test_loss), 
                "cc": float(cc), 
                "sam": float(sam), 
                "rmse": float(rmse), 
                "ergas": float(ergas), 
                "psnr": float(psnr)}
    return image_dict, pred_dic, metrics


if __name__ == "__main__":
      

    __dataset__ = {"pavia_dataset":    pavia_dataset, "botswana_dataset": botswana_dataset,
                   "chikusei_dataset": chikusei_dataset, "botswana4_dataset": botswana4_dataset, "road_train_dataset": road_train_dataset, "dx_dataset": dx_dataset,"dx1_dataset": dx1_dataset, "mdas_dataset": mdas_dataset
        }

    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument(
        '-c', '--config', default='config_eval/config_eval_msd.json', type=str, help='Path to the config file'
        )
    parser.add_argument(
        '-r', '--resume', default=None, type=str, help='Path to the '
                                                                                                                                              '.pth model checkpoint to resume training'
        )
    parser.add_argument(
        '-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)'
        )
    parser.add_argument('--local', action='store_true', default=False)
    args = parser.parse_args()

    # LOADING THE CONFIG FILE
    config = json.load(open(args.config))
    torch.backends.cudnn.benchmark = True

    # SEEDS
    torch.manual_seed(7)

    # NUMBER OF GPUs
    num_gpus = torch.cuda.device_count()

    # MODEL
    model = MODELS[config["model"]](config)
    print(f'\n{model}\n')

    # SENDING MODEL TO DEVICE
    if num_gpus > 1:
        print("Training with multiple GPUs ({})".format(num_gpus))
        model = nn.DataParallel(model).cuda()
    else:
        print(torch.cuda.is_available())
        print("Single Cuda Node is avaiable")
        model.cuda()


    # DATA LOADERS
    print("Training with dataset => {}".format(config["train_dataset"]))
    
    #get_mean_std()
    train_loader = data.DataLoader(
            __dataset__[config["train_dataset"]](
                    config, is_train=True, is_eval=False, want_DHP_MS_HR=config["is_DHP_MS"], ), batch_size=config["train_batch_size"],
            num_workers=config["num_workers"], shuffle=True,
            pin_memory=False,drop_last=True)
    
    val_loader = data.DataLoader(
            __dataset__[config["train_dataset"]](
                    config, is_train=False, is_eval=False, want_DHP_MS_HR=config["is_DHP_MS"], ), batch_size=config["val_batch_size"],
            num_workers=config["num_workers"], shuffle=True, pin_memory=False, )

    test_loader = data.DataLoader(
            __dataset__[config["train_dataset"]](
                    config, is_train=False, is_eval=True, want_DHP_MS_HR=config["is_DHP_MS"], ), batch_size=config["val_batch_size"],
            num_workers=config["num_workers"], shuffle=True, pin_memory=False, )

    # INITIALIZATION OF PARAMETERS
    start_epoch = 1
    total_epochs = config["trainer"]["total_epochs"]

    # OPTIMIZER
    if config["optimizer"]["type"] == "SGD":
        optimizer = optim.SGD(
                model.parameters(), lr=config["optimizer"]["args"]["lr"],
                momentum=config["optimizer"]["args"]["momentum"],
                weight_decay=config["optimizer"]["args"]["weight_decay"]
                )
    elif config["optimizer"]["type"] == "ADAM":
        optimizer = optim.Adam(
                model.parameters(), lr=config["optimizer"]["args"]["lr"],
                weight_decay=config["optimizer"]["args"]["weight_decay"]
                )
    else:
        exit("Undefined optimizer type")

    # LEARNING RATE SHEDULER
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["optimizer"]["step_size"], gamma=config["optimizer"]["gamma"])
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=config["optimizer"]["gamma"])
    # IF RESUME
    if args.resume is not None:
        print("Loading from existing FCN and copying weights to continue....")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint, strict=False)
    #else:
        initialize_weights(model)
        #initialize_weights_new(model)

    # LOSS
    if config[config["train_dataset"]]["loss_type"] == "L1":
        criterion = torch.nn.L1Loss()
        HF_loss = torch.nn.L1Loss()
    elif config[config["train_dataset"]]["loss_type"] == "MSE":
        criterion = torch.nn.MSELoss()
        HF_loss = torch.nn.MSELoss()
    else:
        exit("Undefined loss type")

    if config[config["train_dataset"]]["VGG_Loss"]:
        vggnet = VGG19()
        vggnet = torch.nn.DataParallel(vggnet).cuda()

    if config[config["train_dataset"]]["Spatial_Loss"]:
        Spatial_loss = Spatial_Loss(in_channels=config[config["train_dataset"]]["spectral_bands"]).cuda()

    # SETTING UP TENSORBOARD and COPY JSON FILE TO SAVE DIRECTORY
    PATH = "./" + config["experim_name"] + "/" + config["train_dataset"]
    ensure_dir(PATH + "/")
    writer = SummaryWriter(log_dir=PATH)
    shutil.copy2(args.config, PATH)

    # Print model to text file
    original_stdout = sys.stdout
    with open(PATH + "/" + "model_summary.txt", 'w+') as f:
        sys.stdout = f
        print(f'\n{model}\n')
        sys.stdout = original_stdout


    band_set=[]
    N_bands = config[config["train_dataset"]]["spectral_bands"]
    iters = 1
    selected_num = 150
    for i in range(iters):
        arr = np.arange(N_bands)
        #random.seed(0)
        rng = np.random.default_rng(i)
        rng.shuffle(arr)
        #np.random.shuffle(arr)
        band_list = arr[0: selected_num]
        band_list.sort()
        band_set.append(band_list)
    print(band_set)

    # MAIN LOOP
    best_psnr = 0.0
    best_model = copy.deepcopy(model)
    for epoch in range(start_epoch, total_epochs):
        scheduler.step(epoch)
        print("\nTraining Epoch: %d"%epoch)
        #train_vae(epoch)
        train(epoch)
        if epoch%config["trainer"]["test_freq"] == 0:
            print("\nTesting Epoch: %d"%epoch)
            #image_dict, pred_dic, metrics = test_vae(epoch)
            image_dict, pred_dic, metrics = test(epoch)
            # Saving the best model
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                best_model = copy.deepcopy(model)

                # Saving best performance metrics
                torch.save(model.state_dict(), PATH + "/" + "best_model.pth")
                with open(PATH + "/" + "best_val_metrics.json", "w+") as outfile:
                    json.dump(metrics, outfile)

                # Saving best prediction
                savemat(PATH + "/" + "final_prediction.mat", pred_dic)
            
    pred_dic, metrics = evaluate(best_model, band_set)
    with open(PATH + "/" + "test_metrics.json", "w+") as outfile:
        json.dump(metrics, outfile)

    # Saving best prediction
    savemat(PATH + "/" + "evaluation.mat", pred_dic)