import os
import hubconf
import json
from AIMakerMonitor import counter_inc, gauge_set, api_count_inc
class ModelHandler:
    def __init__(self):
        # key: case insensitive
        # value: case sensitive
        self.__envlist = dict()
        for key, value in os.environ.items():
            self.__envlist[key.strip().lower()] = value.strip()
        # end-of-for
        self.__debug = self.__to_bool(self.__envlist.get('debug', 'true'))

        self.__model_type = self.__envlist.get('model_type','yolov7')
        self.__yolo_model_basepath = os.path.join('/model/',self.__model_type,'weights')
        self.__user_weight = self.__envlist.get('weight','default')
        
        self.__last_weight_name = self.__envlist.get('weight_name','last.pt')
        self.__best_weight_name = self.__envlist.get('weight_name','best.pt')
        
        self.__last_weight_path = os.path.join(self.__yolo_model_basepath,self.__last_weight_name)
        self.__best_weight_path = os.path.join(self.__yolo_model_basepath,self.__best_weight_name)


        self.__model = None

    def __to_bool(self, value):
        if value == None:
            return False
        else:
            return value.lower() in {'true', 'yes', '1'}


    def init(self):
        weight = ""
        if(os.path.exists(self.__user_weight)):
            weight = self.__user_weight
        else:
            if(os.path.exists(self.__best_weight_path)):
                weight = self.__best_weight_path
            else:
                weight = self.__last_weight_path
        if(self.__debug):
            print("Weight Path:{}".format(weight),flush=True)

        self.__model = hubconf.custom(path_or_model=weight)


    def detect(self,data,thresh):
        self.__model.conf = thresh
        yolo_results_json = self.__model(data).pandas().xyxy[0].to_dict(orient='records')
        if(self.__debug):
            print("detect:{}".format(yolo_results_json))
        api_count_inc()
        encoded_results = []
        for result in yolo_results_json:
            to_prom_resp = counter_inc("object_detect", result['name'])
            to_prom_resp = gauge_set("confidence", result["name"], float(result["confidence"]))
            encoded_results.append({
                'confidence': result['confidence'],
                'label': result['name'],
                'points': [
                    result['xmin'],
                    result['ymin'],
                    result['xmax'],
                    result['ymax']
                ]
            })
        if(self.__debug):
            print("result:{}".format(json.dumps(encoded_results)))
        return json.dumps(encoded_results)






    

