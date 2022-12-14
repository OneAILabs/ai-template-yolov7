
import os
from pathlib import Path
import shutil
from os import walk
import sys
import copy
DATA_YAML = os.environ.get("DATA_YAML")
TRAIN_FILE_TYPE = "train"
VALIDATE_FILE_TYPE = "val"
TEST_FILE_TYPE = "test"
TRAIN_FILE_NAME = "train.txt"
VALIDATE_FILE_NAME = "val.txt"
TEST_FILE_NAME = "test.txt"
class UserPreProcess():
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
        self.__curent_dir = None
        self.__target_dir = None
        self.__target_list_path = []
        self.__image_full_list_path = []
        self.__label_full_list_path = []


    def __handel_yaml(self):
        self.__curent_dir = os.getcwd()
        if(os.path.exists(DATA_YAML)):
            self.__target_dir = str(Path(DATA_YAML).parents[0])
        else:
            print("DATA_YAML not found")
            sys.exit(1)
        path_lists = copy.deepcopy(self.__file_path)
        for path_data in path_lists if isinstance(path_lists, list) else [path_lists]:
            os.chdir(self.__target_dir)
            self.__file_path = os.path.abspath(path_data)
            p = Path(self.__file_path)
            os.chdir(self.__curent_dir)
            if(self.__debug):
                print("handel_yaml:current_path:{},self.__file_path={},path_data={}".format(os.getcwd(),self.__file_path,path_data),flush=True)
            if p.is_dir():
                self.__generate_file(p)
            self.__read_file()
            self.__handle_data()
            self.__write_result()

    def __generate_file(self,directory):
        if(self.__file_type == TRAIN_FILE_TYPE):
            path = os.path.join(self.__target_base_path,TRAIN_FILE_NAME)
        elif(self.__file_type == VALIDATE_FILE_TYPE):
            path = os.path.join(self.__target_base_path,VALIDATE_FILE_NAME)
        elif(self.__file_type == TEST_FILE_TYPE):
            path = os.path.join(self.__target_base_path,TEST_FILE_NAME)
        else:
            print("Error: Unknown file type:{}".format(self.__file_type))
            sys.exit(1)

        if not(os.path.exists(self.__target_base_path)):
            Path(self.__target_base_path).mkdir(parents=True, exist_ok=True)

        with open(path,'w') as f:
            for root,dirs, files in os.walk(directory):
                for file in files:
                    fullpath = os.path.join(root, file)
                    if(str(Path(fullpath).suffix)!='.txt'):
                        f.writelines(fullpath+'\n')
        self.__file_path = path

    def __read_file(self):
        self.__image_full_list_path = []
        self.__label_full_list_path = []
        with open(self.__file_path,'r') as f:
            for line in f:
                line = line.strip("\n")
                line = line.rstrip()
                line = line.lstrip()
                if(line.startswith('./')):
                    parent = str(Path(self.__file_path).parent) + os.sep
                    line = line.replace('./',parent)
                image_full_path = line
                label_full_path = str(Path(image_full_path).with_suffix('.txt'))
                self.__image_full_list_path.append(image_full_path)
                self.__label_full_list_path.append(label_full_path)

                if(self.__debug):
                    print("image_full_path:{}".format(image_full_path),flush=True)
                    print("label_full_path:{}".format(label_full_path),flush=True)

    def __handle_data(self):
        self.__handle_image(self.__image_full_list_path)
        self.__handle_label(self.__label_full_list_path)


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
            src = src.replace(sa, sb, 1)
            source = src
            target = src.replace(self.__bucket_path,self.__target_base_path)
            target_path = str(Path(target).parent)
            if not(os.path.exists(target_path)):
                Path(target_path).mkdir(parents=True, exist_ok=True)

            if(self.__debug):
                print("Handle Label \nsrc:{} \ntarget:{}".format(source,target),flush=True)
            if(not os.path.exists(source)):
                print("Warning {} not found".format(source),flush=True)
            else:
                if(not os.path.exists(target)):
                    shutil.copy(source,target)

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
        self.__image_full_list_path = []
        self.__label_full_list_path = []
        if(self.__debug):
            print("=====USER PROCESS=====")
            print("Write Path:{}".format(self.__write_path))
            print("Container Path:{}".format(self.__container_dataset_path))
            print("Source Type:{}".format(self.__source_type))
            print("Bucket Env:{}".format(self.__bucket_env))
            print("Bucket Path:{}".format(self.__bucket_path))
            print("File Type:{}".format(self.__file_type))
            print("File Path:{}".format(self.__file_path))
            print("Target Base Path:{}".format(self.__target_base_path))


    def process(self):
        self.__handel_yaml()

