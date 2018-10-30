"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot

Modify: Implement for getting small data set with one or several cls for model testing.
Updated by: Kaidi
"""
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# note: if you used our download scripts, this should be right


class VOCAnnoTransform_subcls(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            if name not in self.class_to_ind:
                continue
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCTest(data.Dataset):
    """VOC test Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        cls (list): the names of class for test
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
        sub_size (int, optional): create a sub train set with shuffle
        custz_set (str, optional): use a customized sub train set
    """

    def __init__(self, root, cls=['cow'], set='trainval', year=['2007', '2012'],
                 transform=None, target_transform=VOCAnnoTransform_subcls()):
        self.root = root
        self.cls = cls
        self.set = set
        self.year = year
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        if len(cls) <= 0:
            print("The num of class should be one at least.")
            exit(0)
        else:
            rootpath = []
            if self.set == 'test':
                rootpath.append(osp.join(self.root, 'VOC' + '2007'))
            else:

                for y in self.year:
                    rootpath.append(osp.join(self.root, 'VOC' + y))
            for clsname in self.cls:
                if clsname not in VOC_CLASSES:
                    print("Invalid class name!!")
                    exit(0)
            for c in self.cls:
                cls_file = "{}_{}.txt".format(c, self.set)
                for root in rootpath:
                    for line in open(osp.join(root, 'ImageSets', 'Main', cls_file)):
                        line = line.strip()
                        if line.count('-') > 0:
                            continue
                        else:
                            self.ids.append((root, line[:-3]))
            #clean
            clean_ids = list()
            for i in range(len(self.ids)):
                ann = self.pull_anno(i)
                if len(ann[1]) > 0:
                    clean_ids.append(self.ids[i])
                # else:
                #     print(ann[0])
            self.ids = clean_ids

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
            if len(target) == 0:
                return self.pull_item(index+1)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


if __name__ == "__main__":
    train_cls = ['cow']
    clsmap = dict(zip(train_cls, range(len(train_cls))))
    # root = 'D:/NBU_thesis\Reference\Pascal_Voc\VOC2007test'
    root = 'D:/NBU_thesis\Reference\Pascal_Voc\VOCdevkit'
    from utils.augmentations import SSDAugmentation
    dataset = VOCTest(root, train_cls, set='trainval',
                      transform=SSDAugmentation(300, (127, 127, 127)),
                      target_transform=VOCAnnoTransform_subcls(class_to_ind=clsmap))

    from data import detection_collate
    data_loader = data.DataLoader(dataset, 8,
                                  num_workers=2,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    # create batch iterator
    batch_iterator = iter(data_loader)
    for i in range(500):
        images, targets = next(batch_iterator)
        if len(targets[0]) == 0:
            print(targets)
        targets = [ann.cuda() for ann in targets]
        print(i)

