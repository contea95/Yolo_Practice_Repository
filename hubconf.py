"""YOLOv5 PyTorch Hub models https://pytorch.org/hub/ultralytics_yolov5/
Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
"""

from pathlib import Path

import torch

from models.yolo import Model, attempt_load
from utils.general import check_requirements, set_logging
from utils.google_utils import attempt_download
from utils.torch_utils import select_device

dependencies = ['torch', 'yaml']
check_requirements(Path(__file__).parent / 'requirements.txt',
                   exclude=('tensorboard', 'pycocotools', 'thop'))


def create(name, pretrained=True, channels=3, classes=3, autoshape=True, verbose=True):
    """Creates a specified YOLOv5 model
    Arguments:
        name (str): name of model, i.e. 'yolov5s'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
        autoshape (bool): apply YOLOv5 .autoshape() wrapper to model
        verbose (bool): print all information to screen
    Returns:
        YOLOv5 pytorch model
    """
    set_logging(verbose=verbose)
    fname = Path(name).with_suffix('.pt')  # checkpoint filename
    try:
        if pretrained and channels == 3 and classes == 3:
            # download/load FP32 model
            model = attempt_load(fname, map_location=torch.device('cpu'))
        else:
            # model.yaml path
            cfg = list((Path(__file__).parent /
                        'models').rglob(f'{name}.yaml'))[0]
            model = Model(cfg, channels, classes)  # create model
            if pretrained:
                attempt_download(fname)  # download if not found locally
                ckpt = torch.load(
                    fname, map_location=torch.device('cpu'))  # load
                msd = model.state_dict()  # model state_dict
                # checkpoint state_dict as FP32
                csd = ckpt['model'].float().state_dict()
                csd = {k: v for k, v in csd.items(
                ) if msd[k].shape == v.shape}  # filter
                model.load_state_dict(csd, strict=False)  # load
                if len(ckpt['model'].names) == classes:
                    # set class names attribute
                    model.names = ckpt['model'].names
        if autoshape:
            model = model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
        # default to GPU if available
        device = select_device('0' if torch.cuda.is_available() else 'cpu')
        return model.to(device)

    except Exception as e:
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = 'Cache may be out of date, try `force_reload=True`. See %s for help.' % help_url
        raise Exception(s) from e


def custom(path='model/best.pt', autoshape=True, verbose=True):
    # YOLOv5 custom or local model
    return create(path, autoshape=autoshape, verbose=verbose)


if __name__ == '__main__':
    # model = create(name='yolov5s', pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)  # pretrained
    model = custom(path='model/best.pt')  # custom

    # Verify inference
    import cv2
    import numpy as np
    from PIL import Image

    imgs = ['data/images/zidane.jpg',  # filename
            'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg',  # URI
            cv2.imread('data/images/bus.jpg')[:, :, ::-1],  # OpenCV
            Image.open('data/images/bus.jpg'),  # PIL
            np.zeros((320, 640, 3))]  # numpy

    results = model(imgs)  # batched inference
    results.print()
    results.save()
