from operator import truediv
import shutil
import os
import sys
from os import cpu_count, environ
from os.path import join
import subprocess
import AIMaker as ai
import yaml
import mlflow
import torch
from asus_utils import support_mlflow
from dataset_handler import DatasetHandler

YOLO_CFG_BASEPATH = "/yolov7/cfg/training/"
YOLO_DATA_BASEPATH = "/yolov7/data/"
YOLO_WEIGHT_BASEPATH = "/weight/"
YOLO_TEMP_BASEPATH = "/temp/"
YOLO_DATASET_TEMP_BASEPATH = "/datasetTemp/"

DEFAULT_DATA_NAME = "data.yaml"
YOLOV7_TRAIN_NAMES = 'yolov7'
YOLOV7_TEST_NAMES = 'yolov7/test'
TRAIN_FILENAME = "train_list.txt"
VAL_FILENAME = "val_list.txt"
TEST_FILENAME = "test_list.txt"


class Train():
    '''
    Lowercase-based environment
    '''
    def __init__(self):

        # key: case insensitive
        # value: case sensitive
        self.__envlist = dict()
        for key, value in os.environ.items():
            self.__envlist[key.strip().lower()] = value.strip()
        # end-of-for

        self.__model_type = self.__envlist.get('model_type','yolov7')
        self.__debug = self.__to_bool(self.__envlist.get('debug'))
        self.__extra_parameter = self.__envlist.get('extra_parameter','none')
        self.__extra_test_parameter = self.__envlist.get('extra_test_parameter','none')

        self.__imagew = self.__envlist.get('width','640')
        self.__imageh = self.__envlist.get('height','640')
        self.__batch_size = self.__envlist.get('batchsize','16')
        self.__epochs = self.__envlist.get('epochs','150')

        self.__user_weight_path = self.__envlist.get('weight','default')
        self.__user_hyp_path = self.__envlist.get('hyp_yaml','default')
        self.__user_data_path = self.__envlist.get('data_yaml','default')
        self.__user_cfg_path = self.__envlist.get('cfg_yaml','default')

        self.__dataset_handler = DatasetHandler()


        self.__src_cfg = None
        self.__src_data = None
        self.__src_hyp = None
        self.__src_weight = None

        self.__target_cfg = os.path.join(YOLO_TEMP_BASEPATH,self.__model_type+".yaml")
        self.__target_data = os.path.join(YOLO_TEMP_BASEPATH,DEFAULT_DATA_NAME)
        self.__target_hyp = os.path.join(YOLO_TEMP_BASEPATH,"hyp."+self.__model_type+".yaml")
        self.__target_weight = None

        self.__classes = None
        self.__labels = None
        self.__trainfile_path = os.path.join(YOLO_DATASET_TEMP_BASEPATH,TRAIN_FILENAME)
        self.__vaildfile_path = os.path.join(YOLO_DATASET_TEMP_BASEPATH,VAL_FILENAME)
        self.__testfile_path = os.path.join(YOLO_DATASET_TEMP_BASEPATH,TEST_FILENAME)

        self.__test_conf = self.__envlist.get('test_conf','0.001')
        self.__test_iou = self.__envlist.get('test_iou','0.65')

        self.__project = self.__envlist.get('output','/output')

        self.__result_path = os.path.join(self.__project ,YOLOV7_TRAIN_NAMES,"results.txt")
        self.__precision = 0
        self.__recall = 0
        self.__map = 0

        self.__train_file_name = None

        self.__set_dataprocess()
        self.__set_path()
        self.__get_classes()
        self.__get_labels()
        self.__overwrite_yaml()

    def __set_path(self):
        if(self.__model_type == "yolov7"):
            self.__src_cfg = os.path.join(YOLO_CFG_BASEPATH,"yolov7.yaml")
            self.__src_data = os.path.join(YOLO_DATA_BASEPATH,DEFAULT_DATA_NAME)
            self.__src_hyp = os.path.join(YOLO_DATA_BASEPATH,"hyp.scratch.p5.yaml")
            self.__src_weight = os.path.join(YOLO_WEIGHT_BASEPATH,"yolov7_training.pt")
            self.__train_file_name = "train.py"

        elif(self.__model_type == "yolov7-x"):
            self.__src_cfg = os.path.join(YOLO_CFG_BASEPATH,"yolov7x.yaml")
            self.__src_data = os.path.join(YOLO_DATA_BASEPATH,DEFAULT_DATA_NAME)
            self.__src_hyp = os.path.join(YOLO_DATA_BASEPATH,"hyp.scratch.p5.yaml")
            self.__src_weight = os.path.join(YOLO_WEIGHT_BASEPATH,"yolov7x_training.pt")
            self.__train_file_name = "train.py"

        elif(self.__model_type == "yolov7-d6"):
            self.__src_cfg = os.path.join(YOLO_CFG_BASEPATH,"yolov7-d6.yaml")
            self.__src_data = os.path.join(YOLO_DATA_BASEPATH,DEFAULT_DATA_NAME)
            self.__src_hyp = os.path.join(YOLO_DATA_BASEPATH,"hyp.scratch.p6.yaml")
            self.__src_weight = os.path.join(YOLO_WEIGHT_BASEPATH,"yolov7-d6_training.pt")
            self.__train_file_name = "train_aux.py"

        elif(self.__model_type == "yolov7-e6"):
            self.__src_cfg = os.path.join(YOLO_CFG_BASEPATH,"yolov7-e6.yaml")
            self.__src_data = os.path.join(YOLO_DATA_BASEPATH,DEFAULT_DATA_NAME)
            self.__src_hyp = os.path.join(YOLO_DATA_BASEPATH,"hyp.scratch.p6.yaml")
            self.__src_weight = os.path.join(YOLO_WEIGHT_BASEPATH,"yolov7-e6_training.pt")
            self.__train_file_name = "train_aux.py"

        elif(self.__model_type == "yolov7-w6"):
            self.__src_cfg = os.path.join(YOLO_CFG_BASEPATH,"yolov7-w6.yaml")
            self.__src_data = os.path.join(YOLO_DATA_BASEPATH,DEFAULT_DATA_NAME)
            self.__src_hyp = os.path.join(YOLO_DATA_BASEPATH,"hyp.scratch.p6.yaml")
            self.__src_weight = os.path.join(YOLO_WEIGHT_BASEPATH,"yolov7-w6_training.pt")
            self.__train_file_name = "train_aux.py"

        elif(self.__model_type == "yolov7-e6e"):
            self.__src_cfg = os.path.join(YOLO_CFG_BASEPATH,"yolov7-e6e.yaml")
            self.__src_data = os.path.join(YOLO_DATA_BASEPATH,DEFAULT_DATA_NAME)
            self.__src_hyp = os.path.join(YOLO_DATA_BASEPATH,"hyp.scratch.p6.yaml")
            self.__src_weight = os.path.join(YOLO_WEIGHT_BASEPATH,"yolov7-e6e_training.pt")
            self.__train_file_name = "train_aux.py"
        else:
            print("Error,please check MODEL_TYPE:{} env setting".format(self.__model_type),flush=True)
            sys.exit(1)         

        if(self.__debug):
            print("Type:{}\nCfg:{}\nData:{}\nHyp:{}\nWeight:{}".format(self.__model_type,self.__src_cfg,self.__src_data,self.__src_hyp,self.__src_weight))
    
    def __set_dataprocess(self):
        self.__dataset_handler.run()

    def __get_classes(self):
        value = None
        if(os.path.exists(self.__user_data_path)):
            with open(self.__user_data_path, 'r') as stream:
                try:
                    loaded = yaml.load(stream,Loader=yaml.SafeLoader)
                except yaml.YAMLError as exc:
                    print(exc)
            value = loaded['nc']
        else :
            value = self.__dataset_handler.get_class_number()
        if(self.__debug):
            if value is None:
                print("Doesn't find classes")

            else:
                print("classes:{}".format(value))
        self.__classes = int(value)

    def __get_labels(self):
        labels_list = []
        if(os.path.exists(self.__user_data_path)):
            with open(self.__user_data_path, 'r') as stream:
                try:
                    loaded = yaml.load(stream,Loader=yaml.SafeLoader)
                except yaml.YAMLError as exc:
                    print(exc)
            labels_list = loaded['names']
        else :
            labels_list = self.__dataset_handler.get_class_names()
        if(self.__debug):
            if labels_list is None:
                print("Doesn't find classes")

            else:
                print("classes:{}".format(labels_list))

        self.__labels =  labels_list

    def __overwrite_yaml(self):
        self.__overwrite_cfg()
        self.__overwrite_data()
        self.__overwrite_hyp()
        self.__overwrite_weight()

        if(self.__debug):
            print("===========================")
            print("cfg:{},{}".format(self.__src_cfg,self.__target_cfg),flush=True)
            print("data:{},{}".format(self.__src_data,self.__target_data),flush=True)
            print("hyp:{},{}".format(self.__src_hyp,self.__target_hyp),flush=True)
            print("weight:{},{}".format(self.__src_weight,self.__target_weight),flush=True)
            print("classes:{}".format(self.__classes),flush=True)
            print("labels:{}".format(self.__labels),flush=True)        

    def __overwrite_cfg(self):
        if(os.path.exists(self.__user_cfg_path)):
            self.__src_cfg = self.__user_cfg_path
        with open(self.__src_cfg, 'r') as stream:
            try:
                loaded = yaml.load(stream,Loader=yaml.SafeLoader)
            except yaml.YAMLError as exc:
                print(exc)
        loaded['nc'] = self.__classes
        # Save it again
        with open(self.__target_cfg, 'w') as stream:
            try:
                yaml.dump(loaded, stream, sort_keys=False)
            except yaml.YAMLError as exc:
                print(exc)

    def __overwrite_data(self):
        with open(self.__src_data, 'r') as stream:
            try:
                loaded = yaml.load(stream,Loader=yaml.SafeLoader)
            except yaml.YAMLError as exc:
                print(exc)
        loaded['nc'] = self.__classes
        loaded['train'] = self.__trainfile_path
        loaded['val'] = self.__vaildfile_path
        if(os.path.exists(self.__testfile_path)):
            loaded['test'] = self.__testfile_path
        loaded['names'] = self.__labels
        # Save it again
        with open(self.__target_data, 'w') as stream:
            try:
                yaml.dump(loaded, stream, sort_keys=False)
            except yaml.YAMLError as exc:
                print(exc)

    def __overwrite_hyp(self):
        if(os.path.exists(self.__user_hyp_path)):
            self.__src_hyp = self.__user_hyp_path  
        with open(self.__src_hyp, 'r') as stream:
            try:
                loaded = yaml.load(stream,Loader=yaml.SafeLoader)
            except yaml.YAMLError as exc:
                print(exc)
        for k, v in loaded.items():
            if k in self.__envlist:
                loaded[k] = float(self.__envlist.get(k))
                if self.__debug:
                    print("key:{},value:{}".format(k,loaded[k]))
        if self.__debug:
            print(loaded)
        # Save it again
        with open(self.__target_hyp, 'w') as stream:
            try:
                yaml.dump(loaded, stream, sort_keys=False)
            except yaml.YAMLError as exc:
                print(exc)

    def __overwrite_weight(self):
        if(self.__user_weight_path == 'none'):
            self.__src_weight = "' '"
        if(os.path.exists(self.__user_weight_path)):
            self.__src_weight = self.__user_weight_path
        self.__target_weight = self.__src_weight


    def __to_bool(self, value):
        if value == None:
            return False
        else:
            return value.lower() in {'true', 'yes', '1'}

    def __update_result(self):
        with open(self.__result_path,"r") as f:
            last_line = f.readlines()[-1]
        last_line = last_line.strip("\n")
        last_line = " ".join(last_line.split())
        metric = last_line.split(" ")
        self.__precision = metric[8]
        self.__recall = metric[9]
        self.__map = metric[10]
        if(self.__debug):
            print("Precision:{},Recall:{},mAP@0.5:{}".format(self.__precision ,self.__recall, self.__map))
        ai.sendUpdateRequest(float(self.__map))

    def __copy_data_yaml(self):
        data_dir = self.__project
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        src = self.__target_data
        dst = os.path.join(self.__project ,YOLOV7_TRAIN_NAMES,DEFAULT_DATA_NAME)
        if(self.__debug):
            print("Copying data from {} to {}".format(src,dst))
        try:
            shutil.copy(src,dst)
        except IOError as e:
            print(e)

    def __copy_cfg_yaml(self):
        data_dir = self.__project
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        src = self.__target_cfg
        dst = os.path.join(self.__project ,YOLOV7_TRAIN_NAMES, YOLOV7_TRAIN_NAMES+".yaml")
        if(self.__debug):
            print("Copying cfg from {} to {}".format(src,dst))
        try:
            shutil.copy(src,dst)
        except IOError as e:
            print(e)
    def __copy_file_list(self):
        if(os.path.exists(self.__trainfile_path)):
            src = self.__trainfile_path
            dst = os.path.join(self.__project ,YOLOV7_TRAIN_NAMES, TRAIN_FILENAME)
            try:
                shutil.copy(src,dst)
            except IOError as e:
                print(e)
        if(os.path.exists(self.__vaildfile_path)):
            src = self.__vaildfile_path
            dst = os.path.join(self.__project ,YOLOV7_TRAIN_NAMES, VAL_FILENAME)
            try:
                shutil.copy(src,dst)
            except IOError as e:
                print(e)
        if(os.path.exists(self.__testfile_path)):
            src = self.__testfile_path
            dst = os.path.join(self.__project ,YOLOV7_TRAIN_NAMES, TEST_FILENAME)
            try:
                shutil.copy(src,dst)
            except IOError as e:
                print(e)

    def test(self):
        if(self.__debug):
            print("=======EXECUTE TEST=======",flush=True)
        best_weights = "best.pt"
        last_weights = "last.pt"
        weight = ""
        weight_base_path = os.path.join(self.__project,YOLOV7_TRAIN_NAMES,'weights')
        if(os.path.exists(os.path.join(weight_base_path,best_weights))):
            weight = os.path.join(weight_base_path,best_weights)
        else :
            weight = os.path.join(weight_base_path,last_weights)

        cmd = \
        "python test.py " + " " + \
        "--data " + self.__target_data + " " + \
        "--img " + self.__imagew + " " + \
        "--batch-size " + self.__batch_size + " " + \
        "--conf " + self.__test_conf + " " + \
        "--iou " + self.__test_iou + " " + \
        "--device 0 "+ \
        "--weights " + weight + " " + \
        "--name " + YOLOV7_TEST_NAMES + " " + \
        "--project " + self.__project + " "+ \
        "--no-trace " + " " + \
        "--task test" + " "
        if(self.__extra_test_parameter != 'none'):
            cmd = cmd + " " + self.__extra_test_parameter
        if(self.__debug):
            print("CMD:{}".format(cmd),flush=True)
        retcode = subprocess.call(cmd, shell=True)
        print("ret:" + str(retcode), flush=True)


    def train(self):
        #cpu_count = str(os.cpu_count())
        device_count = torch.cuda.device_count()
        if(device_count == 0):
            print("No GPU found, exiting.")
            sys.exit(1)
        elif(device_count == 1):
            cmd = \
            "python " + self.__train_file_name+ " " + \
            "--workers " +str(device_count*4)+" " + \
            "--device 0 "+ \
            "--batch-size " + self.__batch_size + " " + \
            "--data " + self.__target_data + " " + \
            "--img " + self.__imagew + " " + self.__imageh + " " + \
            "--epochs " + self.__epochs + " " + \
            "--cfg " + self.__target_cfg+ " "+ \
            "--weights " + self.__src_weight + " " + \
            "--name " + YOLOV7_TRAIN_NAMES + " " + \
            "--hyp " + self.__target_hyp + " " + \
            "--project " + self.__project + " "
        else:
            device_string = ""
            for i in range(0,int(device_count)):
                if(i+1 == device_count):
                     device_string = device_string + str(i)
                else:
                    device_string = device_string + str(i)+","

            cmd = \
            "python -m torch.distributed.launch " + \
            "--nproc_per_node " + str(device_count) + " " +  \
            "--master_port 9527 " + \
            self.__train_file_name + " " + \
            "--workers " +str(device_count*4)+" " + \
            "--device " + device_string + " " + \
            "--sync-bn " + \
            "--batch-size " + self.__batch_size + " " + \
            "--data " + self.__target_data + " " + \
            "--img " + self.__imagew + " " + self.__imageh + " " + \
            "--epochs " + self.__epochs + " " + \
            "--cfg " + self.__target_cfg+ " "+ \
            "--weights " + self.__src_weight + " " + \
            "--name " + YOLOV7_TRAIN_NAMES + " " + \
            "--hyp " + self.__target_hyp + " " + \
            "--project " + self.__project + " "

        if(self.__extra_parameter !='none'):
            cmd = cmd + " " +self.__extra_parameter

        if(self.__debug):
            print("cmd:{}".format(cmd),flush=True)
        retcode = subprocess.call(cmd, shell=True)
        print("ret:" + str(retcode), flush=True)
        if (retcode == 0):
            self.__update_result()
            self.__copy_data_yaml()
            self.__copy_cfg_yaml()
            self.__copy_file_list()
            if(os.path.exists(self.__testfile_path)):
                self.test()

        else:
            sys.exit(retcode)




if __name__ == '__main__':
    print("[AIM-Train] start yolov7",flush=True)
    train_handler = Train()
    train_handler.train()