import yaml
import os
import sys
from pathlib import Path


DATA_YAML = os.environ.get("DATA_YAML")
YOLO_TEMP_BASEPATH = "/temp/"
USER_TEMP_DATA_NAME = "user_data.yaml"

class UserPreProcess():
    def __init__(self):
        self.__debug = None
        self.__target_data_yaml = os.path.join(YOLO_TEMP_BASEPATH,USER_TEMP_DATA_NAME)
        self.__curent_dir = None
        self.__target_dir = None

    def __handel_yaml(self,src_data_yaml):
        self.__curent_dir = os.getcwd()
        if(os.path.exists(src_data_yaml)):
            self.__target_dir = str(Path(src_data_yaml).parents[0])
            with open(src_data_yaml, 'r') as stream:
                try:
                    loaded = yaml.load(stream,Loader=yaml.SafeLoader)
                except yaml.YAMLError as exc:
                    print(exc)
        else:
            print("DATA_YAML:{} not found".format(src_data_yaml))
            sys.exit(1)
        if loaded.get("train") is not None:
            train_path =self.__to_absolute_path(loaded['train'])
            loaded['train'] = train_path

        if loaded.get("val") is not None:
            val_path = self.__to_absolute_path(loaded['val'])
            loaded['val'] = val_path

        if loaded.get("test") is not None:
            test_path =self.__to_absolute_path(loaded['test'])
            loaded['test'] = test_path
        with open(self.__target_data_yaml, 'w') as stream:
            try:
                yaml.dump(loaded, stream, sort_keys=False)
            except yaml.YAMLError as exc:
                print(exc)

    def __to_absolute_path(self,path):
        os.chdir(self.__target_dir)
        if isinstance(path, list):
            data_path = []
            for p in path:
                data_path.append(os.path.abspath(p))
        else:
            data_path = os.path.abspath(path)

        os.chdir(self.__curent_dir)
        if(self.__debug):
            print("Abs path:{},{},{}".format(self.__curent_dir,self.__target_dir,data_path))
        return data_path


    def process(self,debug,src_data_yaml):
        self.__debug = debug
        if(self.__debug):
            print("=====USER PROCESS=====")
        self.__handel_yaml(src_data_yaml)

