import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
import random
import threading

class Data:
    def __init__(self, name, boxes=None, img=None):
        if boxes is None:
            boxes = []
        self.name = name
        self.boxes = boxes
        self.img = img
        self.shape = img.shape


    def append_box(self, box):
        """
        向这个数据中添加标注框信息
        :param box:
        :return:
        """
        self.boxes.append(box)

    def set_name(self, name):
        self.name = name

    def set_img(self, img):
        self.img = img

    def set_boxes(self, boxes):
        self.boxes = boxes


class Box:
    """
    box类包含两个字段，一个是这个box的类别，一个是这个box的坐标信息[xmin,ymin,xmax,ymax]
    """
    def __init__(self, label, cod):
        self.label = label
        self.cod = cod

    def get_label(self):
        return self.label

    def get_cod(self):
        return self.cod


def load_data(img_path, xml_path, flog_path, save_path):
    """
    加载所有的图像和其标注，以标注文件为准，有的图像并没有标注,然后对其进行数据增强并保存
    :param flog_path: 云雾图像所在目录
    :param save_path: 数据集保存的根目录
    :param img_path: 图像文件路径， tif格式
    :param xml_path: 标注文件路径，VOC格式
    :return:
    """
    annotations = os.listdir(xml_path)
    data_list = []
    for annotation in annotations:
        xml_file = open(os.path.join(xml_path, annotation), 'br')
        boxes = load_annotations(xml_file)
        name = annotation.split(".")[0]
        img = cv2.imread(os.path.join(img_path, name + ".tif"))
        data = Data(name=name, boxes=boxes, img=img)
        data_list.append(data)

    flog_list = os.listdir(flog_path)
    flog_list = [os.path.join(flog_path, x) for x in flog_list]
    ret = []
    # 旋转，加雾，高斯模糊和反转使用并行
    t1 = threading.Thread(target=do, args=(rot90, data_list, ret))
    t1.start()
    t1.join()
    t2 = threading.Thread(target=do, args=(flip_vertical, data_list, ret))
    t2.start()
    t2.join()
    t3 = threading.Thread(target=do, args=(gaussian_blur, data_list, ret))
    t3.start()
    t3.join()
    t4 = threading.Thread(target=do, args=(add_flog, data_list, ret, flog_list))
    t4.start()
    t4.join()
    # 小目标复制, 马赛克和大目标放大并行
    data_list += ret
    ret = []
    print("processing 1")
    # t5 = threading.Thread(target=do, args=(copy_small, data_list, ret))
    # t5.start()
    # t5.join()
    # t6 = threading.Thread(target=do, args=(shrink, data_list, ret))
    # t6.start()
    # t6.join()
    # t7 = threading.Thread(target=do_mosaic, args=(data_list,ret))
    # t7.start()
    # t7.join()
    print("start copy small")
    do(copy_small, data_list, ret)
    print("start shrink ")
    do(shrink, data_list, ret)
    print("start mosaic")
    do_mosaic(data_list, ret)
    data_list += ret
    save(data_list, save_path)


def voc2data(xml_file):
    """
    解析标准VOC格式的标注文件获取其对应的box和所属的类别
    :param xml_file:
    :return:
    """
    boxes = []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for obj in root.iter("object"):
        label = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        boxes.append(Box(label=label, cod=[xmin, ymin, xmax, ymax]))
    return boxes


def load_annotations(xml_file):
    """
    解析标注文件中的信息
    :param xml_file:标注文件
    :return: 标注文件中的坐标信息和所属类别，对应的图像名称等等信息
    """
    boxes = []
    # 解析文件
    tree = ET.parse(xml_file)
    # 获取根节点
    root = tree.getroot()
    # 获取目标节点
    objects = root.find("objects")
    for obj in objects.iter("object"):
        # 找到这个标注框对应的标签
        label = obj.find("possibleresult").find("name").text
        x = []
        y = []
        # 找到这个框的坐标
        for temp in obj.find("points").iter("point"):
            xy = temp.text.split(",")
            x.append(int(float(xy[0])))
            y.append(int(float(xy[1])))
        cod = [min(x), min(y), max(x), max(y)]
        box = Box(label=label, cod=cod)
        boxes.append(box)

    return boxes

def rot90(data):
    """
    将data中的图像做一个旋转90度的处理，将其对应的标注信息也做对应的处理
    :param data:
    :return:
    """
    img = data.img
    img_new = np.rot90(img)
    boxes_new = []
    boxes = data.boxes
    for box in boxes:
        # 旋转90度，现在的x_min就是原来的y_min
        cod = box.cod
        x_min = cod[1]
        # 现在的y_min就是图像的宽减去原来的x_max
        y_min = data.shape[1] - cod[2]
        # 现在的x_max就是原来的y_max
        x_max = cod[3]
        # 现在的y_max就是图像的宽减去原来的x_min
        y_max = data.shape[1] - cod[0]
        box_new = Box(box.label, [x_min, y_min, x_max, y_max])
        boxes_new.append(box_new)

    data_new = Data(data.name + "_rot90", boxes_new, img_new)

    return data_new



def annotations_vis(data):
    """
    将data中的数据可视化
    :param data:
    :return:
    """

    img = data.img.copy()

    for box in data.boxes:
        cv2.rectangle(img, tuple(box.cod[:2]), tuple(box.cod[2:]), (0, 0, 255), 2)
    cv2.imshow(data.name, img)
    cv2.imwrite(data.name + ".png", img)


def flip_vertical(data):
    """
    将data中的图像水平旋转
    :param data:
    :return:
    """
    img = data.img.copy()
    img_new = cv2.flip(img, 0)
    boxes = data.boxes
    boxes_new = []
    for box in boxes:
        cod = box.cod
        y_min = data.shape[0] - cod[3]
        y_max = data.shape[0] - cod[1]
        x_min = cod[0]
        x_max = cod[2]
        cod_new = [x_min, y_min, x_max, y_max]
        box_new = Box(box.label, cod_new)
        boxes_new.append(box_new)
    data_new = Data(data.name + "_flip", boxes_new, img_new)
    return data_new


def random_erase(data):
    """
    随机擦除
    :param data:
    :return:
    """
    pass


def copy_small(data):
    """
    拷贝图像中的小目标，根据长宽的统计，划定面积小于300*300的为小目标
    :param data:
    :return:
    """
    change = False
    img_new = data.img.copy()
    boxes = data.boxes.copy()
    for index, box in enumerate(data.boxes):
        cod = box.cod
        length = cod[3] - cod[1]
        width = cod[2] - cod[0]
        area = length * width
        if area < 20000:
            # 如果有需要复制的目标，保存新的标注文件和图片
            change = True

            # 判定为小目标,把目标部分截取出来
            cropped_img = data.img[cod[1]:cod[3], cod[0]:cod[2]]
            # 获取随机位置
            copy_cods = getRandomCod(data.shape, boxes, 5, width, length)
            for cod in copy_cods:
                img_new[cod[1]:cod[3], cod[0]:cod[2]] = cropped_img
                boxes.append(Box(box.label, cod))
    if change:
        data_new = Data(name=data.name + "_small", img=img_new, boxes=boxes)
        return data_new

    else:
        return None



def check(data_src, data_dst):
    """
    检验增强结果是否正确
    :param data_src:
    :param data_dst:
    :return:
    """
    annotations_vis(data_dst)
    annotations_vis(data_src)
    cv2.waitKey(0)


def getRandomCod(shape, boxes, num, width, height):
    """
    根据传入的图像尺寸和原始标注框的坐标信息来获取num个随机的标注框左上角坐标
    :param height:
    :param width:
    :param num: 生成的随机标注框个数
    :param shape: 图像本身的尺寸
    :param boxes: 原始标注框的坐标信息
    :return: 返回n个cod回去
    """
    ret = []
    for i in range(num):
        while True:
            x = random.randint(0, shape[1])
            y = random.randint(0, shape[0])
            cod_tmp = [x, y, x + width, y + height]
            # 先检验是否越界
            if check_overSize(cod_tmp, shape):
                # 如果越界了，就重新生成坐标
                continue
            overlap = False
            for box in boxes:
                if check_overlap(box.cod, cod_tmp):
                    # 如果相交的话，那么就直接break
                    overlap = True
                    break
                else:
                    continue

            if not overlap and len(ret) != 0:
                for cod in ret:
                    if check_overlap(cod, cod_tmp):
                        overlap = True
                        break
            if not overlap:
                ret.append(cod_tmp)
                break

            else:
                continue

    return ret



def check_overlap(cod_1, cod_2):
    """
    判断两个标注框是否重叠
    :param cod_1:
    :param cod_2:
    :return: true or false
    """
    # 计算交集
    x_min = max(cod_1[0], cod_2[0])
    y_min = max(cod_1[1], cod_2[1])
    x_max = min(cod_1[2], cod_2[2])
    y_max = min(cod_1[3], cod_2[3])

    x = max(0, x_max - x_min)
    y = max(0, y_max - y_min)

    if x * y > 0:
        return True
    else:
        return False


def check_overSize(cod, shape):
    """
    检验标注框是否越界
    :param cod:
    :param shape:
    :return:
    """

    if cod[2] < shape[1] and cod[3] < shape[0]:
        return False
    else:
        return True


def do_mosaic(data_list, result):
    temp_list = []
    data_list = random.sample(data_list, len(data_list))
    for data in data_list:
        if len(temp_list) < 4:
            temp_list.append(resize(data))

        if len(temp_list) == 4:
            new_data = mosaic(temp_list)
            result.append(new_data)
            temp_list = temp_list[-3:]
    return result


def mosaic(data_list):
    """
    mosaic技术需要四张图像才能做
    :param data_list:
    :return:
    """
    img_1 = np.vstack((data_list[0].img.copy(), data_list[1].img.copy()))
    img_2 = np.vstack((data_list[2].img.copy(), data_list[3].img.copy()))

    img_new = np.hstack((img_1, img_2))

    boxes_new = data_list[0].boxes.copy()
    for box in data_list[2].boxes:
        cod = box.cod
        x_min = 600 + cod[0]
        y_min = cod[1]
        x_max = 600 + cod[2]
        y_max = cod[3]
        boxes_new.append(Box(label=box.label, cod=[x_min, y_min, x_max, y_max]))

    for box in data_list[1].boxes:
        cod = box.cod
        x_min = cod[0]
        y_min = cod[1] + 600
        x_max = cod[2]
        y_max = cod[3] + 600
        boxes_new.append(Box(label=box.label, cod=[x_min, y_min, x_max, y_max]))

    for box in data_list[3].boxes:
        cod = box.cod
        boxes_new.append(Box(label=box.label, cod=[x + 600 for x in cod]))

    data_new = Data(name=data_list[0].name + "_mosaic", img=img_new, boxes=boxes_new)

    return data_new




def resize(data, shape=None):
    """
    将图像缩放至600 * 600，方便拼接,
    :param shape:
    :param data:
    :return:
    """
    if shape is None:
        shape = (600, 600)

    img_new = cv2.resize(data.img.copy(), shape)
    boxes = []
    for box in data.boxes:
        cod = box.cod
        x_min = int(shape[0] / data.shape[1] * cod[0])
        y_min = int(shape[0] / data.shape[0] * cod[1])
        x_max = int(shape[1] / data.shape[1] * cod[2])
        y_max = int(shape[1] / data.shape[1] * cod[3])
        box_tmp = Box(label=box.label, cod=[x_min, y_min, x_max, y_max])
        boxes.append(box_tmp)
    data_new = Data(name=data.name, boxes=boxes, img=img_new)

    return data_new


def mask_Grid(data):
    """
    使用mask_Grid数据增强技术
    :param data:
    :return:
    """
    img_new = data.img.copy()
    # 按照img的尺寸生成一个Grid，然后做与运算
    mask = np.zeros_like(img_new)
    # 找到这个标注中最小的那个尺寸，然后以这个尺寸的1/3作为blockSize的大小


def gaussian_blur(data):
    """
    高斯模糊
    :param data:
    :return:
    """
    img_new =  data.img.copy()
    img_new = cv2.GaussianBlur(img_new, (11, 11), 0)
    data_new = Data(name=data.name + "_gaussian", boxes=data.boxes.copy(), img=img_new)

    return data_new


def shrink(data):
    """
    放大仅有小目标的图像
    :param data:
    :return:
    """
    num = 0
    for box in data.boxes:
        cod = box.cod
        area = (cod[3] - cod[1]) * (cod[2] - cod[0])
        if area < 20000:
            num += 1
    if num == 0:
        plate = np.zeros_like(data.img)
        data_new = resize(data, shape=(plate.shape[1] // 2, plate.shape[0] // 2))
        # 确保缩小图像在中间位置
        shape = data_new.shape
        boxes = []
        for box in data_new.boxes:
            cod = box.cod
            cod[0] += shape[0] // 2
            cod[1] += shape[1] // 2
            cod[2] += shape[0] // 2
            cod[3] += shape[1] // 2
            box_new = Box(label=box.label, cod=cod)
            boxes.append(box_new)
        plate[shape[0] // 2 :shape[0] // 2 + shape[0], shape[1] // 2: shape[1] + shape[1] // 2] = data_new.img.copy()
        data_new.set_img(plate)
        data_new.set_name(data.name + "_shrink")
        data_new.set_boxes(boxes)
        return data_new
    else:
        return None


def add_flog(data, flog_list):
    """

    :param data:
    :param flog_list:保存云雾图像的绝对路径列表
    :return:
    """
    img_src = data.img.copy()
    img_flog = cv2.imread(random.choice(flog_list))
    if random.randint(0, 10) > 5:
        img_flog = np.rot90(img_flog)
    img_flog = cv2.resize(img_flog, img_src.shape[:2][::-1])
    img_new = cv2.addWeighted(img_src, 0.6, img_flog, 0.4, 0)
    data_new = Data(name=data.name + "_flog", boxes=data.boxes.copy(), img=img_new)
    return data_new

def data2xml(data):
    """
    将data中的数据解析成xml格式的tree
    :param data:
    :return:
    """
    root = ET.Element("annotation")
    folder = ET.SubElement(root, "folder")
    folder.text = "VOC"
    filename = ET.SubElement(root, "filename")
    filename.text = data.name + ".tif"
    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    height = ET.SubElement(size, "height")
    depth = ET.SubElement(size, "depth")
    width.text = str(data.shape[1])
    height.text = str(data.shape[0])
    depth.text = str(data.shape[2])
    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "高分软件大赛"
    for box in data.boxes:
        obj = ET.SubElement(root, "object")
        name = ET.SubElement(obj, "name")
        name.text = box.label
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        ymin = ET.SubElement(bndbox, "ymin")
        xmax = ET.SubElement(bndbox, "xmax")
        ymax = ET.SubElement(bndbox, "ymax")
        xmin.text = str(box.cod[0])
        ymin.text = str(box.cod[1])
        xmax.text = str(box.cod[2])
        ymax.text = str(box.cod[3])
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = '0'
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = '0'

    pretty_xml(element=root, indent='\t', newline='\n')
    tree = ET.ElementTree(root)
    return tree


def pretty_xml(element, indent, newline, level=0):
    if element:
        # 如果element的text没有内容
        if element.text is None or element.text.isspace():
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
    # 将element转成list
    temp = list(element)
    for sub_element in temp:
        if temp.index(sub_element) < (len(temp) - 1):
            # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            sub_element.tail = newline + indent * (level + 1)
        else:
            # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            sub_element.tail = newline + indent * level
        #对子元素进行递归操作
        pretty_xml(sub_element, indent, newline, level=level + 1)


def save(data_list, save_path):
    annotation_path = os.path.join(save_path, "Annotations")
    img_path = os.path.join(save_path, "JPEGImages")
    if not os.path.exists(annotation_path):
        os.mkdir(annotation_path)
    if not os.path.exists(img_path):
        os.mkdir(img_path)

    for data in data_list:
        img = data.img
        name = os.path.join(img_path, data.name +".tif")
        cv2.imwrite(name, img)
        print("saving image %s to %s" %(data.name +".tif", img_path))
        tree = data2xml(data)
        tree.write(os.path.join(annotation_path, data.name +".xml"), "utf-8", True)
        print("saving annotation %s to %s" % (data.name +".xml", annotation_path))


def do(func, data_list, ret, ext=None):
    if ext is None:
        for data in data_list:
            new_data = func(data)
            if new_data is not None:
                ret.append(new_data)
    else:
        for data in data_list:
            new_data = func(data, ext)
            if new_data is not None:
                ret.append(new_data)


def random_check(save_path, num):
    """
    随机可视化检查save_path下num个样本
    :param save_path:
    :param num:
    :return:
    """
    temp_path = os.path.join(save_path, "tmp")
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    annotation_path = os.path.join(save_path, "Annotations")
    img_path = os.path.join(save_path, "JPEGImages")
    xml_list = os.listdir(annotation_path)
    xml_list = random.sample(xml_list, num)
    xml_list = [os.path.join(annotation_path, xml) for xml in xml_list]
    for xml in xml_list:
        tree = ET.parse(xml)
        root = tree.getroot()
        img = None
        img_name = ''
        try:
            img_name = root.find("filename").text
            img = cv2.imread(os.path.join(img_path, img_name))
        except IOError:
            RuntimeError("no such file")
        except TypeError:
            RuntimeError("xml tag error")
        for obj in root.iter("object"):
            name = obj.find("name").text
            box = obj.find("bndbox")
            xmin = int(box.find("xmin").text)
            ymin = int(box.find("ymin").text)
            xmax = int(box.find("xmax").text)
            ymax = int(box.find("ymax").text)
            img = cv2.putText(img, name, (xmin, ymin), font, 1.2, (255,255,255), 2)
            cv2.rectangle(img,(xmin, ymin), (xmax, ymax), (255,0,0),2)
        cv2.imwrite(os.path.join(temp_path, img_name), img)


def splitImageSets(save_path):
    """
    将文件夹构成标准的VOC格式
    :param save_path:
    :return:
    """
    imageSet = os.path.join(save_path, "imageSets")
    if not os.path.exists(imageSet):
        os.mkdir(imageSet)
    train = open(os.path.join(imageSet, "train.txt"), 'w', encoding="utf-8")
    test = open(os.path.join(imageSet, "test.txt"), 'w', encoding="utf-8")
    train_val = open(os.path.join(imageSet, "trainavl.txt"), 'w', encoding="utf-8")
    val = open(os.path.join(imageSet, "val.txt"), 'w', encoding="utf-8")
    # 切分测试集和验证集, 比例为0.9，0.1
    train_val_info = []
    xml_list = os.listdir(os.path.join(save_path, "Annotations"))
    for xml in xml_list:
        info = xml.split(".")[0]
        if np.random.uniform(0, 1) > 0.9:
            # 划分为测试集
            test.writelines(info + "\n")
        else:
            # trainval
            train_val_info.append(info)
            train_val.writelines(info + "\n")
    # 划分训练集和验证集，比例为0.9，0.1
    for info in train_val_info:
        if np.random.uniform(0, 1) > 0.9:
            # 验证集
            val.writelines(info + "\n")
        else:
            train.write(info + "\n")

    train.close()
    test.close()
    train_val.close()
    val.close()

if __name__ == '__main__':
    xmlPath = "e:\\data\\xml"
    imgPath = "e:\\data\\img"
    flogPath = "e:\\flog"
    savePath = "new_data"
    # 数据增强入口
    load_data(imgPath, xmlPath, flogPath, savePath)
    # 随机可视化检测100个样本
    random_check(savePath, 100)
    # 构建数据集
    splitImageSets(savePath)
