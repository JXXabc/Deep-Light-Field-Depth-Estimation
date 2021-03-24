import torch.nn.parallel
import torch.nn.parallel
import PIL.Image as Image
import numpy as np
import torch
import os

def compute_errors(deppath, gtpath,ignore_zero=True):
    gtdir = gtpath
    depdir = deppath
    files = os.listdir(gtdir)
    eps = np.finfo(float).eps
    # Initialize cumulative metric values
    crms = 0
    crmsl = 0
    cabs_rel = 0
    csq_rel = 0
    cacc1 = 0
    cacc2 = 0
    cacc3 =0
    for i, name in enumerate(files):
        if not os.path.exists(gtdir + name):
            print(gtdir + name, 'does not exist')
        gt = Image.open(gtdir + name)

        gt = np.array(gt, dtype=np.uint8)
        gt=gt/255
        gt = torch.from_numpy(gt).float()

        pred = Image.open(depdir + name).convert('L')
        pred = pred.resize((np.shape(gt)[1], np.shape(gt)[0]))
        pred = np.array(pred, dtype=np.float)
        pred=pred/255
        pred = torch.from_numpy(pred).float()
        # Count number of valid pixels
        # n_pxls += np.sum(mask)
        if ignore_zero:
            pred[gt == 0] = 0.0
            n_pxls = (gt > 0.0).sum()
        else:
            n_pxls = gt.size
        abs_diff = (pred - gt).abs()
        rms = np.square(gt - pred)
        crms += np.sum(rms)

        rmsl = np.square(np.log(gt) - np.log(pred))
        crmsl += np.sum(rmsl)

        abs_rel = np.abs(gt - pred) / gt
        cabs_rel += np.sum(abs_rel)

        # Compute SRD
        sq_rel = np.square(gt - pred) / gt
        csq_rel += np.sum(sq_rel)

        max_ratio = np.maximum(gt / pred, pred/ gt)

        # Compute accuracies for different deltas
        acc1 = np.asarray(np.logical_and(max_ratio < 1.25), dtype=np.float32)
        acc2 = np.asarray(np.logical_and(max_ratio < 1.25 ** 2), dtype=np.float32)
        acc3 = np.asarray(np.logical_and(max_ratio < 1.25 ** 3), dtype=np.float32)

        cacc1 += np.sum(acc1)
        cacc2 += np.sum(acc2)
        cacc3 += np.sum(acc3)

        RMSE = np.sqrt(crms / n_pxls)
        RMSE_log = np.sqrt(crmsl / n_pxls)
        ABS_REL = cabs_rel / n_pxls
        SQ_REL = csq_rel / n_pxls
        ACCURACY1 = cacc1 / n_pxls
        ACCURACY2 = cacc2 / n_pxls
        ACCURACY3 = cacc3 / n_pxls

        # Display metrics
        print(RMSE)
        print(RMSE_log)
        print(ABS_REL)
        print(SQ_REL)
        print(ACCURACY1)
        print(ACCURACY2)
        print(ACCURACY3)
        print(n_pxls)

def eva1(deppath, gtpath,ignore_zero=True):
    gtdir = gtpath
    depdir = deppath
    files = os.listdir(gtdir)
    eps = np.finfo(float).eps

    delta1_accuracy = 0
    delta2_accuracy = 0
    delta3_accuracy = 0
    rmse_linear_loss = 0
    rmse_log_loss = 0
    abs_relative_difference_loss = 0
    squared_relative_difference_loss = 0

    for i, name in enumerate(files):
        if not os.path.exists(gtdir + name):
            print(gtdir + name, 'does not exist')
        gt = Image.open(gtdir + name)
        gt = np.array(gt, dtype=np.uint8)
        gt = (gt - gt.min()) / (gt.max() - gt.min() + eps)
        gt = torch.from_numpy(gt).float()

        pred = Image.open(depdir + name)
        pred = pred.convert('L')
        pred = pred.resize((np.shape(gt)[1], np.shape(gt)[0]))
        pred = np.array(pred, dtype=np.float)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + eps)

        pred = torch.from_numpy(pred).float()
        if len(pred.shape) != 2:
           pred= pred[:, :, 0]
        if len(gt.shape) != 2:
           gt= gt[:, :, 0]
        if ignore_zero:
            pred[gt == 0] = 0.0
        delta1_accuracy += threeshold_percentage(pred, gt, 1.25)
        delta2_accuracy += threeshold_percentage(pred, gt, 1.25 * 1.25)
        delta3_accuracy += threeshold_percentage(pred, gt, 1.25 * 1.25 * 1.25)
        rmse_linear_loss += rmse_linear(pred, gt)
        rmse_log_loss += rmse_log(pred, gt)
        abs_relative_difference_loss += abs_relative_difference(pred, gt)
        squared_relative_difference_loss += squared_relative_difference(pred, gt)
    delta1_accuracy /= (i + 1)
    delta2_accuracy /= (i + 1)
    delta3_accuracy /= (i + 1)
    rmse_linear_loss /= (i + 1)
    rmse_log_loss /= (i + 1)
    abs_relative_difference_loss /= (i + 1)
    squared_relative_difference_loss /= (i + 1)
    print(
        '    {:.4f}      {:.4f}      {:.4f}      '
        '    {:.4f}      {:.4f}      {:.4f}      {:.4f}'.format(
            delta1_accuracy, delta2_accuracy, delta3_accuracy,
            rmse_linear_loss, rmse_log_loss,   abs_relative_difference_loss, squared_relative_difference_loss))

def threeshold_percentage(output, target, threeshold_val):
    w=target.shape[0]
    h=target.shape[1]
    output = output.view(1, 1, w, h)
    target = target.view(1, 1, w, h)

    d1 = torch.exp(output) / torch.exp(target)
    d2 = torch.exp(target) / torch.exp(output)

    # d1 = output/target
    # d2 = target/output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(output.shape[0], output.shape[1], output.shape[2], output.shape[3])
    one = torch.ones(output.shape[0], output.shape[1], output.shape[2], output.shape[3])
    bit_mat = torch.where(max_d1_d2.cpu() < threeshold_val, one, zero)
    count_mat = torch.sum(bit_mat, (1, 2, 3))
    threeshold_mat = count_mat / (output.shape[2] * output.shape[3])
    return threeshold_mat.mean()


def rmse_linear(output, target):
    w=target.shape[0]
    h=target.shape[1]
    output = output.view(1, 1, w, h)
    target = target.view(1, 1, w, h)
    actual_output = torch.exp(output)
    actual_target = torch.exp(target)
    diff = actual_output - actual_target
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (1, 2, 3)) / (output.shape[2] * output.shape[3])
    rmse = torch.sqrt(mse)
    return rmse.mean()


def rmse_log(output, target):
    w=target.shape[0]
    h=target.shape[1]
    output = output.view(1, 1, w, h)
    target = target.view(1, 1, w, h)
    diff = output - target
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (1, 2, 3)) / (output.shape[2] * output.shape[3])
    rmse = torch.sqrt(mse)
    return rmse.mean()


def abs_relative_difference(output, target):
    w=target.shape[0]
    h=target.shape[1]
    output = output.view(1, 1, w, h)
    target = target.view(1, 1, w, h)
    actual_output = torch.exp(output)
    actual_target = torch.exp(target)
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    abs_relative_diff = torch.sum(abs_relative_diff, (1, 2, 3)) / (output.shape[2] * output.shape[3])
    return abs_relative_diff.mean()


def squared_relative_difference(output, target):
    w=target.shape[0]
    h=target.shape[1]
    output = output.view(1, 1, w, h)
    target = target.view(1, 1, w, h)
    actual_output = torch.exp(output)
    actual_target = torch.exp(target)
    square_relative_diff = torch.pow(torch.abs(actual_output - actual_target), 2) / actual_target
    square_relative_diff = torch.sum(square_relative_diff, (1, 2, 3)) / (output.shape[2] * output.shape[3])
    return square_relative_diff.mean()


def main():
    print("\n evaluating ....")
    eva1(
        deppath='input the prediction depth map root'+'/',
        gtpath='input the gt depth map root' + '/')

if __name__ == '__main__':
    main()