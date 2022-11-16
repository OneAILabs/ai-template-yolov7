
import os
from pathlib import Path
import shutil

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
        self.__target_list_path = []
        self.__image_full_list_path = []
        self.__label_full_list_path = []

    def __read_file(self):
        self.__image_full_list_path = []
        self.__label_full_list_path = []
        with open(self.__file_path,'r') as f:
            for line in f:
                line = line.strip("\n")
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
            shutil.copy(source,target)

    def __write_result(self):
        if(self.__debug):
            print("Write Result:{}".format(self.__write_path))
        with open(self.__write_path,'a') as f:
            for taget in self.__target_list_path:
                f.write(taget+'\n')

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
        self.__read_file()
        self.__handle_data()
        self.__write_result()

