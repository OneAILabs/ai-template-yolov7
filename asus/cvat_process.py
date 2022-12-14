from cProfile import label
import os
from pathlib import Path
import shutil

CVAT_TRAIN_PARTTRN_1 = "data/obj_train_data/"
CVAT_TRAIN_PARTTRN_2 = "obj_train_data/"
CLASS_NAMES = "obj.names"
CLASS_DATA = "obj.data"
class CVATPreProcess():
    def __init__(self):
        self.__debug = None
        self.__write_path = None
        self.__container_dataset_path = None
        self.__source_type = None
        self.__bucket_env = None
        self.__bucket_path = None
        self.__file_type = None
        self.__file_path = None
        self.__target_base_path = None
        self.__cvat_bucket = None
        self.__target_list_path = []

    '''
    Read CVAT train.txt,the content show as below:
    
    data/obj_train_data/jc-dataset/image/000001.jpg

    From content we can know:
    train.txt path: /jc-dataset/Export/{taskid}/YOLO 1.1/train.txt
    Bucket name: jc-dataset
    Image Path: /jc-dataset/image/000001.jpg
    Label Path: /jc-dataset/Export/{taskid}/YOLO 1.1/obj_train_data/jc-dataset/image/000001.txt
    '''
    def __read_raw_file(self):
        image_full_list_path = []
        label_full_list_path = []
        with open(self.__file_path,'r') as f:
            for line in f:
                line = line.strip("\n")

                '''
                Use split to parse train.txt to get
                iamge_path & label_path form file
                Line = "data/obj_train_data/jc-dataset/image/000001.jpg"
                image_path = "image/000001.jpg"
                label_path = "obj_train_data/jc-dataset/image/000001.txt"
                Then image_full_path & label_full_path will show as below:
                image_full_path = {bucket_path}/image/000001.jpg
                label_full_path = {bucket_path}/Export/{taskid}/YOLO 1.1/obj_train_data/jc-dataset/image/000001.txt
                '''
                if(not self.__cvat_bucket):
                    self.__cvat_bucket  = line.split('/')[2]
                image_path = line.split('/',3)[-1]
                image_full_path = os.path.join(self.__bucket_path,image_path)

                label_path = str(Path(line.split('/',1)[-1]).with_suffix('.txt'))
                label_full_path = os.path.join(str(Path(self.__file_path).parent),label_path)

                image_full_list_path.append(image_full_path)
                label_full_list_path.append(label_full_path)
                
                if(self.__debug):
                    print("image_path:{}".format(image_path),flush=True)
                    print("image_full_path:{}".format(image_full_path),flush=True)
                    print("label_path:{}".format(label_path),flush=True)
                    print("label_full_path:{}".format(label_full_path),flush=True)
                    print("cvat_bucket:{}".format(self.__cvat_bucket),flush=True)
                
        self.__handle_data(image_full_list_path,label_full_list_path)


    def __handle_data(self,image_src_list,label_src_list):
        self.__handle_image(image_src_list)
        self.__handle_label(label_src_list)
        self.__write_result()

    def __handle_image(self,src_list):
        for src in src_list:
            source = src
            target = src.replace(self.__bucket_path,self.__target_base_path)
            target_path = str(Path(target).parent)
            self.__target_list_path.append(target)
            if not(os.path.exists(target_path)):
                Path(target_path).mkdir(parents=True, exist_ok=True)
        
            if(self.__debug):
                print("Handle Image \nsrc:{} \ntarget:{}".format(source,target),flush=True)
            if(not os.path.exists(target)):
                os.symlink(source, target)
        
    def __handle_label(self,src_list):
        sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep
        for src in src_list:
            source = src
            target = src.split(self.__cvat_bucket)[1].strip('/')
            target = os.path.join(self.__target_base_path,target)
            target = target.replace(sa, sb, 1)
            target_path = str(Path(target).parent)
            if not(os.path.exists(target_path)):
                Path(target_path).mkdir(parents=True, exist_ok=True)

            if(self.__debug):
                print("Handle Label \nsrc:{} \ntarget:{}".format(source,target),flush=True)
            if(os.path.exists(source)):
                if(not os.path.exists(target)):
                    shutil.copy(source,target)
            else:
                print("Warning {} not found".format(source),flush=True)

    def __write_result(self):
        if(self.__debug):
            print("Write Path:{}".format(self.__write_path))
            print("Write Result:{}".format(self.__target_list_path))
        with open(self.__write_path,'a') as f:
            for taget in self.__target_list_path:
                f.write(taget+'\n')
        self.__target_list_path = []


    def set_info(self,info,write_path,dataset_path,debug):

        self.__debug = debug
        self.__write_path = write_path
        self.__container_dataset_path = dataset_path
        self.__source_type = info["source_type"]
        self.__bucket_env = info["bucket_env"]
        self.__bucket_path = info["bucket_path"]
        self.__file_type = info["file_type"]
        self.__file_path  = info["file_path"]
        self.__target_base_path = os.path.join(self.__container_dataset_path,str(self.__bucket_path).strip('/'))
        self.__target_list_path = []
        self.__cvat_bucket  = None
        if(self.__debug):
            print("=====CVAT PROCESS=====")
            print("Write Path:{}".format(self.__write_path))
            print("Container Path:{}".format(self.__container_dataset_path))
            print("Source Type:{}".format(self.__source_type))
            print("Bucket Env:{}".format(self.__bucket_env))
            print("Bucket Path:{}".format(self.__bucket_path))
            print("File Type:{}".format(self.__file_type))
            print("File Path:{}".format(self.__file_path))
            print("Target Base Path:{}".format(self.__target_base_path))

    def get_class_names(self):
        class_names = []
        class_path = os.path.join(str(Path(self.__file_path).parent),CLASS_NAMES)
        with open(class_path,"r") as f:
            for line in f:
                line = line.strip("\n")
                class_names.append(line)
        if(self.__debug):
            print("class_path:{}".format(class_path))
            print("class_names:{}".format(class_names))
        return class_names

    def get_class_number(self):
        class_data = os.path.join(str(Path(self.__file_path).parent),CLASS_DATA)
        value = None
        with open(class_data,"r") as f:
            for line in f:
                line = line.strip("\n")
                if line.find("classes") != -1:
                    value = line.split("=")[1]
                    break
        if(self.__debug):
            if value is None:
                print("Doesn't find classes")

            else:
                print("classes:{}".format(value))
        return value

    def process(self):
        self.__read_raw_file()

            


