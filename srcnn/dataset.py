import sys
import os
import numpy as np
import global_path
import math
import skimage


class KITTICalib(object):
    ''' Calibration matrices and utils  校准矩阵和工具h
           3d XYZ in <label>.txt are in rect camera coord.
           2d box xy are in image2 coord
           Points in <lidar>.bin are in Velodyne coord.

           y_image2 = P^2_rect * x_rect
           y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
           x_ref = Tr_velo_to_cam * x_velo
           x_rect = R0_rect * x_ref

           P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                       0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                       0,      0,      1,      0]
                    = K * [1|t]

           image2 coord:
            ----> x-axis (u)
           |
           |
           v y-axis (v)

           velodyne coord:
           front x, left y, up z

           rect/ref camera coord:
           right x, down y, front z

           Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

           TODO(rqi): do matrix multiplication only once for each projection.
       '''

    def __init__(self, filename):
        '''
        @author chonepieceyb
        :param filename:  calibfile for one image  标定文件名
        '''
        with open(filename) as f:
            lines = f.readlines()[:-1]
            assert (len(lines) == 7)  # 判断标定文件的大小
            for i, line in enumerate(lines):
                line = line.split(':')[1].strip()  # remove white space 去除空格和冒号
                vals = np.array([float(str) for str in line.split(" ")], dtype=np.float64)
                if i == 0:
                    self.P0 = vals.reshape(3, 4)
                elif i == 1:
                    self.P1 = vals.reshape(3, 4)
                elif i == 2:
                    self.P2 = vals.reshape(3, 4)
                elif i == 3:
                    self.P3 = vals.reshape(3, 4)
                elif i == 4:
                    self.R0_rect = vals.reshape(3, 3)
                elif i == 5:
                    self.Tr_velo_to_cam = vals.reshape(3, 4)
                elif i == 6:
                    self.Tr_imu_to_velo = vals.reshape(3, 4)


class KITTIObject(object):
    class Box2D(object):
        def __init__(self):
            self.box = []  # [x1,y1,x2,y2] left top rigt bottom

    '''
        class for one object in on kittidata set image  一张图片里的单个object类
    '''

    def __init__(self, labelline, calib_object):
        self.read_kitti_object(labelline, calib_object)           # read label  读取参数
        self.calculate_boxes(calib_object)                        # cal box    计算box

    def read_kitti_object(self, labelline, calib_object):
        '''
        :usage read kitti labelfile
        @author chonepieceyb
        :param labelline: kitti label line
        :param calib_object: kitti calib_object
        :return:
        '''
        '''
        kitti label file format:
        #Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
        '''
        args = labelline.split(' ')
        self.classname = args[0]
        self.truncated = float(args[1])
        self.occluded = float(args[2])
        self.alpha = float(args[3])
        self.dimensions = tuple([float(args[i]) for i in range(8, 11)])  # (height,width,length)
        self.location = tuple([float(args[i]) for i in range(11, 14)])  # (x0,y0,z0)
        self.rotation_y = float(args[14])

    def calculate_boxes(self, calib_object):
        '''
        :param calib_object:  标定文件
        :return
        '''
        self.boxes = []
        for _ in range(3):
            box = self.Box2D()
            box.box = np.array([1000000, 1000000, 0, 0], dtype=np.float64)  # x0 y0 x1 y1  init the box
            self.boxes.append(box)
        '''
        to calculate the 2D box
        1 first calculate 3D box location: 
        1 -----------2
       /|(x0,y0,z0) /|
      0 -----------3 .
      | |          | |
      . 5 ----------6
      |/           |/
      4 -----------7
       the center is on the center of top of the box in camera coordinates
       the coordinates is 
        front:z left:x down:y
       2 the using : P2* Homogeneous(codinate)
        '''
        # calculate box3D location
        rotation_y = self.rotation_y + math.pi/2
        R_y = np.array([[math.cos(rotation_y), 0, math.sin(rotation_y)],
                        [0, 1, 0],
                        [-math.sin(rotation_y), 0, math.cos(rotation_y)]])
        h, w, l = self.dimensions
        x0, y0, z0 = self.location
        self.pos3D = []
        self.pos3D.append(np.array([x0,y0,z0],dtype=np.float64)+R_y.dot(np.array([ - w, 0,  - l]) / 2))  # x0 , y0, z0
        self.pos3D.append(np.array([x0,y0,z0],dtype=np.float64)+R_y.dot(np.array([ - w, 0,   l]) / 2))  # x1 , y1, z1
        self.pos3D.append(np.array([x0,y0,z0],dtype=np.float64)+R_y.dot(np.array([ + w, 0,   l]) / 2))  # x2 , y2, z2
        self.pos3D.append(np.array([x0,y0,z0],dtype=np.float64)+R_y.dot(np.array([ + w, 0,  -l]) / 2))  # x3 , y3, z3

        self.pos3D.append(np.array([x0,y0,z0],dtype=np.float64)+R_y.dot(np.array([ - w,   2 * h, - l]) / 2))  # x4 , y4, z4
        self.pos3D.append(np.array([x0,y0,z0],dtype=np.float64)+R_y.dot(np.array([ - w,  2 * h, l]) / 2))  # x5 , y5, z5
        self.pos3D.append(np.array([x0,y0,z0],dtype=np.float64)+R_y.dot(np.array([   w,  2 * h, l]) / 2))  # x6 , y6, z6
        self.pos3D.append(np.array([x0,y0,z0],dtype=np.float64)+R_y.dot(np.array([   w,  2 * h, - l]) / 2))  # x7 , y7, z7

        # project the 3D point to 2D  left and right image , and calculate 2D box
        def space2Image(P, pos3D):
            '''
            project 3D loaction to 2D location
            :param P:  project Matrix  [3,4] 投影矩阵
            :param pos3D:  3D location  shape:[3] 3-d 坐标
            :return:  [2] 2-d [x,y]
            '''
            pos3D = np.append(pos3D, [1])  # normoalize
            pos2D = P.dot(pos3D)
            pos2D = (pos2D / pos2D[2])[0:2]
            return pos2D

        for i in range(2):
            pos_2ds = []
            for j in range(8):
                if self.pos3D[j][2] < 0:  # exclude the pos z<0  排除在z坐标小于0的点
                    continue
                if i == 0:  # left box
                    pos_2ds.append(space2Image(calib_object.P2, self.pos3D[j]))
                elif i == 1: #right box
                    pos_2ds.append(space2Image(calib_object.P3, self.pos3D[j]))
            # calculate the box
            for pos2d in pos_2ds:
                self.boxes[i].box[0] = max(0,min(self.boxes[i].box[0],pos2d[0]))         # 所有点坐标最小的为x0
                self.boxes[i].box[1] = max(0,min(self.boxes[i].box[1],pos2d[1]) )       # 所有点坐标最小的y0
                self.boxes[i].box[2] = max(self.boxes[i].box[2], pos2d[0])          # 所有点坐标最小的为x1
                self.boxes[i].box[3] = max(self.boxes[i].box[3], pos2d[1])            # 所有点坐标最小的y1
        #calculate union box
        for i in range(2):
            self.boxes[2].box[0]=max(0,min(self.boxes[2].box[0],self.boxes[i].box[0]))   # x0
            self.boxes[2].box[1]=max(0,min(self.boxes[2].box[1],self.boxes[i].box[1]))   # y0
            self.boxes[2].box[2] = max(self.boxes[2].box[2], self.boxes[i].box[2])       # x1
            self.boxes[2].box[3] = max(self.boxes[2].box[3], self.boxes[i].box[3])        # y1
    def print_object_info(self):

        info = 'type:{},truncated:{},occluded:{},alpha:{},dimensions:{},location:{},rotation_y:{}\npos_3d:{}\n box_left:{},box_right:{},box_union:{}'.format(
            self.classname, self.truncated, self.occluded, self.alpha, self.dimensions, self.location, self.rotation_y,
            self.pos3D,
            self.boxes[0].box, self.boxes[1].box, self.boxes[2].box
        )
        print(info)

class KITTIDataset(object):
    def __init__(self):
        self.class_dict={'BG':0,'Pedestrian':1,'Car':2,'Cyclist':3}
        self.class_infos=[{'id':0, 'name': 'BG'}]
        self.image_infos=[]
        self._image_ids=[]
    @property
    def image_ids(self):
        return self._image_ids

    def add_class(self,id,classname):                                          # add class to dataset
        self.class_infos.append({'id':id, 'name':classname})

    def add_iamge(self,id,path_left,path_right,kitti_objects,**kwargs):
        info={'id':id,'path_left':path_left,'path_right':path_right,'kitti_objects':kitti_objects}
        info.update(kwargs)
        self.image_infos.append(info)

    def isImage(self,image):
        ext = os.path.splitext(image)
        ext = ext[1]
        if ext == ".png" or ext == ".jpg" or ext == ".jpeg" or ext == ".bmp":
            return True
        else:
            return False

    def load_kitti(self,dataset_dir,subset):                                          #load_kitti dataset
        '''

        :param dataset_dir:  training set dir
        :param subset: train or val
        :return:
        '''

        assert subset in ['train','val']
        dataset_dir = os.path.join(dataset_dir,subset)
        LEFT_IMG_DIR = os.path.join(dataset_dir,'image_2')
        RIGHT_IMG_DIR =os.path.join(dataset_dir,'image_3')

        # add Class
        # 添加class ,  1:'Pedestrian', 2:'Car', 3:'Cyclist'  0:BG
        self.add_class( 1, "Pedestrian")
        self.add_class( 2, 'Car')
        self.add_class( 3, 'Cyclist')

        # read left and right images and sort them by name wihch is number eg 000000 -> 0000001   读取左右两张图片
        images_left = os.listdir(LEFT_IMG_DIR)
        images_right = os.listdir(RIGHT_IMG_DIR)
        images_left.sort()
        images_right.sort()
        len_dataset = min(len(images_left),len(images_right))
        for i in range(len_dataset):
            kitti_objects = []
            left_image = images_left[i]
            # maka sure it is a image file and the right and left match  确保是图片并且左右两边的图片匹配
            if not (self.isImage(left_image) and left_image==images_right[i]):
                continue
            id = os.path.splitext(left_image)[0]
            text_name =  id+'.txt'
            calib_file = os.path.join(dataset_dir,'calib',text_name)
            label_file = os.path.join(dataset_dir,'label_2',text_name)
            # read calib file
            calib_object = KITTICalib(calib_file)
            # read label
            with open(label_file) as label:
                for line in label.readlines():
                    line=line.strip()
                    kitti_object = KITTIObject(line,calib_object)
                    if not kitti_object.classname in self.class_dict.keys():
                        continue
                    kitti_objects.append( kitti_object )

            # read image to get w and h
            image_left_path = os.path.join(LEFT_IMG_DIR,left_image)
            images_right_path = os.path.join(RIGHT_IMG_DIR,left_image)
            height, width = skimage.io.imread(image_left_path).shape[:2]

            # add image 添加照片
            self.add_iamge(
                id,
                image_left_path,
                images_right_path,
                kitti_objects,
                width=width,
                height=height,
            )
        self.class_num = len(self.class_infos)
        self.class_ids = np.arange(self.class_num)
        self.image_num = len(self.image_infos)
        self._image_ids = np.arange(self.image_num)
        self.class_names = [c_info['name'] for c_info in self.class_infos]

    def load_image(self,image_id):
        '''
        :param iamge_id: image_id in image_ids
        :return: left_image , right_image . left_image [H,W,C]
        '''
        # Load image
        image_left = skimage.io.imread(self.image_infos[image_id]['path_left'])
        image_right = skimage.io.imread(self.image_infos[image_id]['path_right'])
        # If grayscale. Convert to RGB for consistency.
        if image_left.ndim != 3:
            image_left = skimage.color.gray2rgb(image_left)
        if image_right.ndim != 3:
            image_right = skimage.color.gray2rgb(image_right)
        # If has an alpha channel, remove it for consistency
        if image_left.shape[-1] == 4:
            image_left = image_left[..., :3]
        if image_right.shape[-1] == 4:
            image = image_right[..., :3]
        return image_left , image_right

    def load_image_gt_info(self,image_id):
        '''

        :param image_id:  image_id in image_ids
        :return:  class_ids , boxes_left:[N,(y0,x0,y1,x1)] , boxes_right , boxes_union
        '''
        '''
        kitti_object :
                    self.classname = args[0]
                    self.truncated = float(args[1])
                    self.occluded = float(args[2])
                    self.alpha = float(args[3])
                    self.dimensions = tuple([float(args[i]) for i in range(8, 11)])  # (height,width,length)
                    self.location = tuple([float(args[i]) for i in range(11, 14)])  # (x0,y0,z0)
                    self.rotation_y = float(args[14])
                    self.boxes
        '''
        def convert_boxes(box):
            '''
            :param box: [x0,y0,x1,y1]
            :return: [y0,x0,y1,x1]
            '''
            return np.stack([box[1],box[0],box[3],box[2]],axis=0)
        kitti_objects = self.image_infos[image_id]['kitti_objects']
        class_ids=[]
        boxes_left = []
        boxes_right=[]
        boxes_union=[]
        for object in kitti_objects:
            class_ids.append( self.class_dict[object.classname])
            boxes_left.append(convert_boxes(object.boxes[0].box))
            boxes_right.append(convert_boxes(object.boxes[1].box))
            boxes_union.append(convert_boxes(object.boxes[2].box))
        class_ids = np.array(class_ids)
        boxes_left = np.array(boxes_left)
        boxes_right = np.array(boxes_right)
        boxes_union = np.array(boxes_union)
        return class_ids,boxes_left,boxes_right,boxes_union

    def info_generator(self):
        for image in self.image_infos:

            info ='id:{}\n path_left:{} \n path_right:{}\n width:{}, right:{}'.format(
                image['id'],image['path_left'],image['path_right'],image['width'],image['height']
            )
            print(info)
            for object in image['kitti_objects']:
                object.print_object_info()
            yield image




#测试代码
if __name__  =='__main__':
    kitti_dataset = KITTIDataset()
    kitti_dataset.load_kitti(global_path.KITTI_DIR,'val')
    info_g = kitti_dataset.info_generator()

    for info in info_g:
        input('输入继续')