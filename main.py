#running training, testing, and inference

!python3 train.py --data data/ship-obj.data --img-size 256 --batch-size 16 --accumulate 4  #train data

!python3 test.py --data data/ship-obj.data --img-size=256  #test data

import utils

utils.plot_results_orig() #plotting the results

!python3 detect.py #detection on inference image

