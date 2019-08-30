import argparse
import time

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import test  # import test.py to get mAP after each epoch
from models import *
from datasets import *
from utils import *


    
hyp = {'giou': 1.2,  # giou loss gain
       'xy': 4.062,  # xy loss gain
       'wh': 0.185,  # wh loss gain
       'cls': 15.7,  # cls loss gain
       'cls_pw': 3.67,  # cls BCELoss positive_weight
       'obj': 20.0,  # obj loss gain
       'obj_pw': 1.36,  # obj BCELoss positive_weight
       'iou_t': 0.194,  # iou target-anchor threshold
       'lr0': 0.0001,  # initial learning rate
       'lrf': -4.,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.944,  # SGD momentum
       'weight_decay': 0.0005,
       'degrees': 1.03,  # image rotation (+/- deg)
       'translate': 0.0552,  # image translation (+/- fraction)
       'scale': 0.0555,  # image scale (+/- gain)
       'shear': 0.434}  # image shear (+/- deg)
  

def train(cfg,
          data,
          img_size=256,
          epochs=300,  # 500200 batches at bs 16, 117263 images = 273 epochs
          batch_size=16,
          accumulate=4):  # effective bs = batch_size * accumulate = 16 * 4 = 64
    # Initialize
    init_seeds()
    weights = 'weights' + os.sep
    last = weights + 'last.pt' #last weight
    best = weights + 'best.pt' #best weight
    device = torch.device('cuda:0') #initializing the device
    
    # Configure run
    data_dict = parse_data_cfg(data) #parsing data
    train_path = data_dict['train']  #access train data path
    nc = int(data_dict['classes'])  # number of classes

    # Initialize model
    model = Darknet(cfg).to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'],
                          nesterov=True)

    #cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_fitness = 0.
    
    # Initialize model with backbone (optional)
    #cutoff = load_darknet_weights(model, weights + 'darknet53.conv.74')

    # Remove old results
    for f in glob.glob('*_batch*.jpg') + glob.glob('results.txt'):
        os.remove(f)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in [0.8]], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    # Dataset
    dataset = LoadImagesAndLabels(train_path,
                                  img_size,
                                  batch_size,
                                  augment=True,
                                  hyp=hyp)  # augmentation hyperparameters
                                  

    # Dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=opt.num_workers,
                                             shuffle=True, 
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Start training
    model.hyp = hyp  # attach hyperparameters to model
    model_info(model, report='summary')  # 'full' or 'summary'
    nb = len(dataloader)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        print(('\n' + '%10s' * 9) %
              ('Epoch', 'gpu_mem', 'GIoU/xy', 'wh', 'obj', 'cls', 'total', 'targets', 'img_size'))

        # Update scheduler
        scheduler.step()

        mloss = torch.zeros(5).to(device)  # mean losses
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Plot images with bounding boxes
            if epoch == 0 and i == 2:
                plot_images(imgs=imgs, targets=targets, paths=paths, fname='train_batch%g.jpg' % i)

           
            # Run model
            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model, giou_loss=not opt.xywh)
            if torch.isnan(loss):
                print('WARNING: nan loss detected, ending training')
                return results
				
            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == nb:
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
            pbar.set_description(s)  # print(s)

        # Calculate mAP (always test final epoch, skip first 5 if opt.nosave)
        if not (opt.notest or (opt.nosave and epoch < 10)) or epoch == epochs - 1:
            with torch.no_grad():
                results, maps = test.test(cfg, data, batch_size=batch_size, img_size=opt.img_size, model=model,
                                          conf_thres=0.1)

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 5 % results + '\n')  # P, R, mAP, F1, test_loss

        # Update best map
        fitness = results[2]
        if fitness > best_fitness:
            best_fitness = fitness

        # Save training results
        save = (not opt.nosave) or (epoch == epochs - 1)
        if save:
            with open('results.txt', 'r') as file:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': file.read(),
                         'model': model.state_dict(),
                         'optimizer': optimizer.state_dict()}

            # Save last checkpoint
            torch.save(chkpt, last)
            if opt.bucket:
                os.system('gsutil cp %s gs://%s' % (last, opt.bucket))  # upload to bucket

            # Save best checkpoint
            if best_fitness == fitness:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, weights + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

    # Report time
    print('%g epochs completed in %.3f hours.' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    return results




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--accumulate', type=int, default=4, help='number of batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-1cls.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/ship-obj.data', help='ship-obj.data file path')
    parser.add_argument('--img-size', type=int, default=256, help='inference size (pixels)')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--transfer', action='store_true', help='transfer learning flag')
    parser.add_argument('--num-workers', type=int, default=4, help='number of Pytorch DataLoader workers')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--xywh', action='store_true', help='use xywh loss instead of GIoU loss')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--var', default=0, type=int, help='debug variable')
    opt = parser.parse_args()
    print(opt)

    # Train 
    results = train(opt.cfg,
                    opt.data,
                    img_size=opt.img_size,
                    epochs=opt.epochs,
                    batch_size=opt.batch_size,
                    accumulate=opt.accumulate)

   