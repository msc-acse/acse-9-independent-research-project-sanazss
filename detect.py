import argparse
import time
import utils
import datasets
import models
from models import *
from datasets import *
from utils import *



def detect(
        cfg,
        data_cfg,
        weights,
        images='data/samples',  # input folder
        output='output',  # output folder
        img_size=512,
        conf_thres=0.5,
        nms_thres=0.5,
        save_images=True,
):
    device = torch.device('cuda:0') #defining the device
    torch.backends.cudnn.benchmark = False  # set False for reproducible results
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder
	
	#initializing model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])

    # Eval mode
    model.to(device).eval()

    dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
   # colors = [255, 255,255]

    for i, (path, img, im0) in enumerate(dataloader):
        t = time.time()
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
             
        pred, _ = model(img)
        
        det = non_max_suppression(pred, conf_thres, nms_thres)[0]
        
        if det is not None and len(det) > 0:
            # Rescale boxes to true image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results to screen
            print('%gx%g ' % img.shape[2:], end='')  # print image size
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls_conf, cls in det:
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
               
        print('Done. (%.3fs)' % (time.time() - t))

        if save_images:  # Save image with detections
            cv2.imwrite(save_path, cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY))                    

    if save_images:
        print('Results saved to %s' % os.getcwd() + os.sep + output)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-1cls.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/ship-obj.data', help='ship-obj.data file path')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--output', type=str, default='output', help='specifies the output path for images')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt.cfg,
               opt.data,
               opt.weights,
               images=opt.images,
               img_size=opt.img_size,
               conf_thres=opt.conf_thres,
               nms_thres=opt.nms_thres,
               output=opt.output)
