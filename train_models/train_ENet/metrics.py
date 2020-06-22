import numpy as np
from tqdm import tqdm
import torch
from scipy import ndimage

def fill_contuor(mask):
    bin_image=mask.astype(np.int8)
    bin_image = ndimage.binary_fill_holes(bin_image)
    return bin_image

def TP(target, prediction, t=0.5):
    target = (target == 1)
    prediction = (prediction > t)
    
    # For each class
    TPs = {"OW": 0, "IW": 0, "tumor": 0}
    
    for i, region in enumerate(TPs.keys()):
        TPs[region] = (target[i] * prediction[i]).sum()
            
    TPs["total"] = (target * prediction).sum()
    return TPs

def FP(target, prediction, t=0.5):
    target = (target == 1)
    prediction = (prediction > t)
    
    # For each class
    FPs = {"OW": 0, "IW": 0, "tumor": 0}
    
    for i, region in enumerate(FPs.keys()):
        FPs[region] = ((~target[i]) * prediction[i]).sum()
            
    FPs["total"] = ((~target) * prediction).sum()
    return FPs
    
def TN(target, prediction, t=0.5):
    target = (target == 1)
    prediction = (prediction > t)
    
    # For each class
    TNs = {"OW": 0, "IW": 0, "tumor": 0}
    
    for i, region in enumerate(TNs.keys()):
        TNs[region] = ((~target[i]) * (~prediction[i])).sum()
            
    TNs["total"] = ((~target) * (~prediction)).sum()
    return TNs

def FN(target, prediction, t=0.5):
    target = (target == 1)
    prediction = (prediction > t)
    
    # For each class
    FNs = {"OW": 0, "IW": 0, "tumor": 0}
    
    for i, region in enumerate(FNs.keys()):
        FNs[region] = (target[i] * (~prediction[i])).sum()
            
    FNs["total"] = (target * (~prediction)).sum()
    return FNs

def precision(target, prediction, t=0.5):
    
    precisions = {}
    tps = TP(target, prediction, t)
    fps = FP(target, prediction, t)
    
    for region in tps.keys():
        if (tps[region] + fps[region]) > 0:
            precisions[region] = tps[region] / (tps[region] + fps[region])
        else:
            precisions[region] = None
    
    return precisions

def recall(target, prediction, t=0.5):
    
    recalls = {}
    tps = TP(target, prediction, t)
    fns = FN(target, prediction, t)
    
    for region in tps.keys():
        if (tps[region] + fns[region]) > 0:
            recalls[region] = tps[region] / (tps[region] + fns[region])
        else:
            recalls[region] = None
    
    return recalls

def DICE(target, prediction, t=0.5):
    
    DICEs = {}
    tps = TP(target, prediction, t)
    fps = FP(target, prediction, t)
    fns = FN(target, prediction, t)
    
    for region in tps.keys():
        if 2*tps[region] + fps[region] + fns[region] > 0:
            DICEs[region] = 2*tps[region] / ( 2*tps[region] + fps[region] + fns[region] )
        else:
            DICEs[region] = None
            
    return DICEs

def IOU(target, prediction, t=0.5):
    r""" Calculate IOU metric on given target mask and predicted probability mask
    
    target: numpy.ndarray of size (N, H, W)
    prediction: numpy.ndarray of size (N, H, W)
    t: int
        
    """
    
    target = (target == 1)
    prediction = (prediction > t)
    
    # Total
    if target.sum() == 0:
        IOU = None
    else:
        intersection = (target * prediction).sum(axis=1, keepdims=True).sum(axis=2, keepdims=True)
        union = (target + prediction).sum(axis=1, keepdims=True).sum(axis=2, keepdims=True)

        IOU = (intersection / (union + 1) ).mean()
    
    # For each class
    IOUs = {"OW": 0, "IW": 0, "tumor": 0}
    
    for i, region in enumerate(IOUs.keys()):
        if target[i].sum() == 0:
            IOUs[region] = None
        else:
            intersection = (target[i] * prediction[i]).sum()
            union = (target[i] + prediction[i]).sum()

            IOUs[region] = intersection / (union + 1)
            
    IOUs["total"] = IOU
    return IOUs

    
    
def evaluate_metric(model, dataloader, metric_name, t=0.5, device=torch.device("cpu")):

    if metric_name == "IOU":
        metric = lambda x, y, t: IOU(x, y, t)
    elif metric_name == "DICE":
        metric = lambda x, y, t: DICE(x, y, t)   
    elif metric_name == "recall":
        metric = lambda x, y, t: recall(x, y, t)  
    elif metric_name == "precision":
        metric = lambda x, y, t: precision(x, y, t)     
     
        
    metrics_data = {"OW": 0, "IW": 0, "tumor": 0, "total":0}
    counts_data = {"OW": 0, "IW": 0, "tumor": 0, "total":0}
        
    for X, y in tqdm(dataloader):
        with torch.no_grad():
            out_ = model(X.float().to(device))
            out = torch.sigmoid(out_)
#             out = torch.softmax(out_, 1)

#         metrics_batch = {"OW": 0, "IW": 0, "tumor": 0, "total":0}
#         counts_batch = {"OW": 0, "IW": 0, "tumor": 0, "total":0}
        
        # Calculate metric for each sample in the batch
        for i in range(X.shape[0]):
            metrics = metric(y[i, :, :, :].numpy(), out[i, :, :, :].cpu().numpy(), t=t)
            
            for region in metrics.keys():
                if metrics[region]:
                    metrics_data[region] += metrics[region]
                    counts_data[region] += 1

#         # Calculate metrics for batch
#         # Add to metrics for dataset
#         for region in metrics_batch.keys():
#             if counts_batch[region] > 0:
#                 metrics_batch[region] = metrics_batch[region] / counts_batch[region]
#                 counts_data[region] += 1
#                 metrics_data[region] += metrics_batch[region]
        
    # Calculate final
    for region in metrics_data.keys():
        if counts_data[region] > 0:
            metrics_data[region] = metrics_data[region] / counts_data[region]
            
    return metrics_data
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


