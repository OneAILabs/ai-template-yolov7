import os
import random
from pathlib import Path
import re
import yaml
from cvat_process import CVATPreProcess
from user_process import UserPreProcess
TRAIN_BUCKET_KEYWORD = "train_dataset"
VALIDATE_BUCKET_KEYWORD = "validate_dataset"
TEST_BUCKET_KEYWORD = "test_dataset"

CVAT_EXPORT_DIRECTORY = "Export"
CVAT_YOLO_DIRECTORY = "/YOLO 1.1"

TRAIN_FILE_NAME = "train.txt"
VALIDATE_FILE_NAME = "val.txt"
TEST_FILE_NAME = "test.txt"

TRAIN_LIST = "train_list.txt"
TRAIN_TEMP_LIST = "train_temp_list.txt"
VALIDATE_LIST = "val_list.txt"
TEST_LIST = "test_list.txt"


CVAT_DATASET_TYPE= "CVAT"
USER_DATASET_TYPE = "USER"

TRAIN_FILE_TYPE = "train"
VALIDATE_FILE_TYPE = "val"
TEST_FILE_TYPE = "test"

class DatasetHandler:
    def __init__(self):
        # key: case insensitive
        # value: case sensitive
        self.__envlist = dict()
        for key, value in os.environ.items():
            self.__envlist[key.strip().lower()] = value.strip()
        # end-of-for
        
        self.__debug = self.__to_bool(self.__envlist.get('debug'))

        self.__cvat_train_task_id = self.__envlist.get('cvat_task_id','none')
        self.__cvat_val_task_id = self.__envlist.get('cvat_val_task_id','none')
        self.__cvat_test_task_id = self.__envlist.get('cvat_test_task_id','none')
        self.__dataset_path = self.__envlist.get('dataset','/dataset')
        self.__user_data_path = self.__envlist.get('data_yaml','default')

        self.__cvat_train_bucket = dict()
        self.__cvat_val_bucket = dict()
        self.__cvat_test_bucket = dict()

        self.__train_info = []
        self.__valid_info = []
        self.__test_info = []
        self.__train_validation_rate = self.__envlist.get('train_validation_rate','8:2')
        self.__run_test = False

        self.__class_names = []
        self.__class_number = None

        self.__container_dataset_path =  self.__envlist.get('containerdatasetpath','/datasetTemp')
        self.__train_list_path = os.path.join(self.__container_dataset_path,TRAIN_LIST)
        self.__train_temp_list_path = os.path.join(self.__container_dataset_path,TRAIN_TEMP_LIST)
        self.__val_list_path = os.path.join(self.__container_dataset_path,VALIDATE_LIST)
        self.__test_list_path = os.path.join(self.__container_dataset_path,TEST_LIST)

        self.__cvat_process = CVATPreProcess()
        self.__user_process = UserPreProcess()

    def __get_bucket(self):
        if(self.__cvat_train_task_id != 'none'):
            self.__cvat_train_task_id = self.__cvat_train_task_id.split(',')
            self.__cvat_train_bucket['cvat_train'] = self.__dataset_path

        if(self.__cvat_val_task_id != 'none'):
            self.__cvat_val_task_id = self.__cvat_val_task_id.split(',')
            self.__cvat_val_bucket['cvat_val'] = self.__dataset_path

        if(self.__cvat_test_task_id != 'none'):
            self.__cvat_test_task_id = self.__cvat_test_task_id.split(',')
            self.__cvat_test_bucket['cvat_test'] = self.__dataset_path

        if(self.__debug):
            print("CVAT Train Bucket:{}".format(self.__cvat_train_bucket))
            print("CVAT Valid Bucket:{}".format(self.__cvat_val_bucket))
            print("CVAT Test Bucket:{}".format(self.__cvat_test_bucket))


    def __set_cvat_info(self,file_type,bucket_path,bucket_env):
        if(file_type == TRAIN_FILE_TYPE):
            cvat_id = self.__cvat_train_task_id
        elif(file_type == VALIDATE_FILE_TYPE):
            cvat_id = self.__cvat_val_task_id
        else:
            cvat_id = self.__cvat_test_task_id
        scan_result = []
        export_path = os.path.join(bucket_path,CVAT_EXPORT_DIRECTORY)
        if(os.path.exists(export_path)):
            for path in Path(export_path).rglob(TRAIN_FILE_NAME):
                for id in cvat_id:
                    path_keyword = str(id)+CVAT_YOLO_DIRECTORY
                    if(path_keyword in str(path)):
                        data = {
                            "source_type":CVAT_DATASET_TYPE,
                            "bucket_env":bucket_env,
                            "bucket_path":bucket_path,
                            "file_type": file_type,
                            "file_path": str(path)
                            }
                        scan_result.append(data)
        else:
              return None
        return scan_result

    def __to_bool(self, value):
        if value == None:
            return False
        else:
            return value.lower() in {'true', 'yes', '1'}

    def __generate_info(self,info_type,cvat_bucket):
        info = []
        file_type =  info_type
        for bukect_env in cvat_bucket:
            bucket_path = cvat_bucket.get(bukect_env)
            cvat_result = self.__set_cvat_info(file_type,bucket_path,bukect_env)
            if( cvat_result != None):
                info.extend(cvat_result)

        if(file_type == TRAIN_FILE_TYPE):
            self.__train_info = info
        elif(file_type == VALIDATE_FILE_TYPE):
            self.__valid_info = info
        else:
            self.__test_info = info
        if(self.__debug):
            print("{}Info:{}".format(file_type,info))


    def is_run_test(self):
        return self.__run_test

    def get_class_names(self):
        print("get_class_names:{}".format(self.__class_names))
        return self.__class_names

    def get_class_number(self):
        print("get_class_number:{}".format(self.__class_number))
        return self.__class_number

    def run(self):
        if(os.path.exists(self.__user_data_path)):
            self.__user_process.process(self.__debug,self.__user_data_path)
        self.__get_bucket()
        self.__generate_info(TRAIN_FILE_TYPE,self.__cvat_train_bucket)
        self.__generate_info(VALIDATE_FILE_TYPE,self.__cvat_val_bucket)
        self.__generate_info(TEST_FILE_TYPE,self.__cvat_test_bucket)

        if(self.__train_info):
            for info in self.__train_info:
                if(info['source_type'] == CVAT_DATASET_TYPE):
                    self.__cvat_process.set_info(info,self.__train_temp_list_path,self.__container_dataset_path,self.__debug)
                    self.__cvat_process.process()
                    if not self.__class_names :
                        self.__class_names  = self.__cvat_process.get_class_names()
                    if not self.__class_number:
                        self.__class_number = self.__cvat_process.get_class_number()
        if(self.__valid_info):
            for info in self.__valid_info:
                if(info['source_type'] == CVAT_DATASET_TYPE):
                    self.__cvat_process.set_info(info,self.__val_list_path,self.__container_dataset_path,self.__debug)
                    self.__cvat_process.process()
                    if not self.__class_names :
                        self.__class_names  = self.__cvat_process.get_class_names()
                    if not self.__class_number:
                        self.__class_number = self.__cvat_process.get_class_number()

        if(self.__test_info):
            self.__run_test = True
            for info in self.__test_info:
                if(info['source_type'] == CVAT_DATASET_TYPE):
                    self.__cvat_process.set_info(info,self.__test_list_path,self.__container_dataset_path,self.__debug)
                    self.__cvat_process.process()
                    if not self.__class_names :
                        self.__class_names  = self.__cvat_process.get_class_names()
                    if not self.__class_number:
                        self.__class_number = self.__cvat_process.get_class_number()

        if(os.path.exists(self.__train_temp_list_path)):
            self.__generate_train_file()

    def __generate_train_file(self):
        user_val= None
        if(self.__debug):
            print("Generating CVAT Training file...")
        if(os.path.exists(self.__user_data_path)):
            with open(self.__user_data_path, 'r') as stream:
                try:
                    loaded = yaml.load(stream,Loader=yaml.SafeLoader)
                except yaml.YAMLError as exc:
                    print(exc)
            if loaded.get("val") is not None:
                user_val = loaded['val']
        if(user_val is None and not(os.path.exists(self.__val_list_path))):
            self.__generate_val_by_rate()
        else:
            if(self.__debug):
                print("Rename CVAT Template Training file...")
            os.rename(self.__train_temp_list_path,self.__train_list_path)

    def __generate_val_by_rate(self):
        if(self.__debug):
            print("Generating CVAT Val file by rate")
        train_rate = int(self.__train_validation_rate.split(':')[0])
        validation_rate = int(self.__train_validation_rate.split(':')[1])
        train_percent = float(train_rate/(train_rate+validation_rate))
        all_files_name_list= []
        with open(self.__train_temp_list_path,"r") as f:
            for line in f:
                line = line.strip("\n")
                all_files_name_list.append(line)
        num_all_files = len(all_files_name_list)
        num_train = int(num_all_files*train_percent)
        train = random.sample(all_files_name_list,num_train)
        val = [i for i in all_files_name_list if not i in train]
        if(self.__debug):
            print("Gen train_list.txt & val_list.txt by rate.\ntrain_rate:{},validation_rate:{},train_percent:{}".format(train_rate,validation_rate,train_percent))
        with open(self.__train_list_path,"w") as f:
            for name in train:
                f.write(name+'\n')

        with open(self.__val_list_path,"w") as f:
            for name in val:
                f.write(name+'\n')


if __name__ == '__main__':
    dataset = DatasetHandler()
    dataset.run()