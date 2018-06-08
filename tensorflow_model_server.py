from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.python.saved_model import signature_constants
from google.protobuf import json_format
import json
import base64
import time
import subprocess
import select
import os
import shutil
import traceback

import parallel_prediction

class TimeoutException(Exception):
    pass

def parsePredictResponse(predictResponse):
    result = {}
    json_msg = json_format.MessageToJson(predictResponse)
    msg = json.loads(json_msg)
    outputs = msg["outputs"]
    for key in outputs:
        output = outputs[key]
        result[key]=[]
        if output['dtype'] == 'DT_STRING':
            for val in output['stringVal']:
                result[key].append(base64.b64decode(val))
        elif output['dtype'] == 'DT_INT64':
            for val in output['int64Val']:
                result[key].append(int(val))
        elif output['dtype'] == 'DT_FLOAT':
            for val in output['floatVal']:
                result[key].append(float(val))

    return result


def build_image_key_request(request, data):
    request.inputs['image_bytes'].CopyFrom(tf.contrib.util.make_tensor_proto(data["image_b64"].decode('base64'), shape=[1]))
    request.inputs['key'].CopyFrom(tf.contrib.util.make_tensor_proto(data["key"], shape=[1]))
    return request


def build_image_key_request_single(request, data):
    request.inputs['image_bytes'].CopyFrom(tf.contrib.util.make_tensor_proto(data["image_b64"].decode('base64')))
    request.inputs['key'].CopyFrom(tf.contrib.util.make_tensor_proto(data["key"], shape=[1]))
    return request


def execute_gcloud_command(cmd):
    with open(os.devnull, 'w') as dev_null:
        try:
            res = subprocess.check_output(cmd, stderr=dev_null).strip()
            return res
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise Exception('gcloud is not installed.'
                            'Please install and set up gcloud.')
            raise

def execute_command(cmd):
    with open(os.devnull, 'w') as dev_null:
        try:
            res = subprocess.check_output(cmd, stderr=dev_null).strip()
            #if not res:
                
                
            return res
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise Exception('it seems like the program used to run the command'
                            ''+' '.join(cmd)+' is not installed')
            raise


def download_model_from_gcs(gcs_link, local_filepath):
    wildcarded_gcs_link = os.path.join(gcs_link,"*")
    download_location = os.path.join(local_filepath,"1")
    if not os.path.isdir(download_location):
        os.makedirs(download_location)
    try:
        execute_gcloud_command(["gsutil", "cp", "-r", wildcarded_gcs_link, download_location])
        return wildcarded_gcs_link, download_location
    except subprocess.CalledProcessError as e:
        print "Model download failed.\n Please check if the specified GCS path: "+gcs_link+" exists and you can access to it."
        raise e
        
def download_models_from_gcs(gcs_links, local_filepath, verbose=True):
    
    config = "model_config_list: {\n"
    for k in gcs_links:
        gcs_link=gcs_links[k]
        local_position=os.path.join(local_filepath,k)
        if verbose:
            print("downloading "+k+" from "+gcs_link+" to "+local_position)
        wildcarded_gcs_link, download_location = download_model_from_gcs(gcs_link, local_position)
        local_path_to_model=os.path.abspath("/".join(download_location.split("/")[:-1]))
        if verbose:
            print("model "+k+" downloaded to "+local_path_to_model)
        config += """
        config: {
        name: \""""+k+"""\",
        base_path: \""""+local_path_to_model+"""\",
        model_platform: "tensorflow",
        },
        """
    config = config[:-1]+"\n}"
    return config
    
    
def local_filesystem_cleanup(local_filepath):
    folder = local_filepath
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

            
class TensorflowModelServer:
    """Class to handle a tensorflow_model_server lifecycle
    Typical usage:
        with TensorflowModelServer(model_path, **args) as tms:
            result = tms.predict(data)
    """
    
    def __init__(self, 
                 models, 
                 host="0.0.0.0",
                 port=9000, 
                 verbose=True, 
                 timeout_seconds=10,
                 rpc_timeout_seconds=10,
                 staging_area="./tensorflow_model_server_staging", 
                 cleanup=True,
                 request_builder=build_image_key_request,
                 response_formatter=parsePredictResponse):
        
        """Initializes the internal variables of the class
        
        Args:
            models: a key-value dictionary containing KEY: name of the model; VALUE: GCS path of the model
            host: IP addess where the tensorflow_model_server listens to
            port: port where the tensorflow_model_server listens to
            verbose: if True, prints out some statistics during the usage of this class methods
            timeout_seconds: seconds to wait during tensorflow_model_server instantiation
            rpc_timeout_seconds: seconds to wait during tensorflow_model_server predictions
            staging_area: local filepath where the model will be stored if it resides on GCP
            cleanup: if True, deletes the folder specified in "staging_area" on object disposal
            request_builder: A function that takes as input a partially built request and the necessary data to perform the prediction.
                 - This function must process the data in order to complete the request, and return it
            response_formatter: A function that takes as input a prediction object and processes it in order to change its format
        """
        
        self.models = models
        self.model_names=sorted(models.keys())
        self.host = host
        self.port = port
        self.verbose = verbose
        self.timeout=timeout_seconds
        self.rpc_timeout_seconds = rpc_timeout_seconds
        self.staging_area = staging_area
        self.cleanup = cleanup
        self.request_builder = request_builder
        self.response_formatter = response_formatter
        
        self.from_gcs = False
        self.ready = False
        self.stub = None
        
    def _reset(self):
        self.from_gcs = False
        self.ready = False
        self.stub = None
        
    def _log(self, str):
        if self.verbose:
            print str
        
    def start(self):
        """Runs the tensorflow_model_server according to the arguments specified in the class' constructor

        Raises:
            TimeoutException: If the tensorflow_model_server takes too long to start
        """
        
        if self.ready:
            return
        
            
        self.gcs_link = self.models
        self.local_model_path = self.staging_area
        self.from_gcs = True
        #self._log("downloading model from: "+self.gcs_link+" to "+self.local_model_path)
        self._log("downloading models...")
        config = download_models_from_gcs(self.gcs_link, self.local_model_path) 
        model_config_file = os.path.join(self.staging_area,"config.yaml")
        with open(model_config_file, "w") as f:
            f.write(config)
        
        cmd = [
            'tensorflow_model_server', 
            '--port='+str(self.port), 
            '--model_config_file='+model_config_file+''
        ]
        
        self._log("launching:")
        self._log(" ".join(cmd))
        self.process = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        self.poller = select.poll()
        self.poller.register(self.process.stderr,select.POLLIN)
        
        waits = 0
        
        while waits < self.timeout:
            if self.poller.poll(1):
                #print x.stdout.readline()
                res = self.process.stderr.readline()
                self._log(res)
                if "Running ModelServer at" in res: 
                    self.ready = True
                    break;
            else:
                time.sleep(1)
                waits += 1
        
        if not self.ready:
            raise TimeoutException("Timeout reached after "+str(self.timeout)+"seconds")
            
            
    def stop(self):
        """ Stops the tensorflow_model_server and frees the allocated resources.
            If cleanup is set to True and the model resides on GCS, removes the downloaded contents from disk.
        """
        
        self._log("tearing down tensorflow_model_server")
        self.poller.unregister(self.process.stderr)
        self.process.kill()  
        self._log("tear down complete")
        if self.cleanup and self.from_gcs:
            self._log("cleaning up "+self.local_model_path)
            local_filesystem_cleanup(self.local_model_path)
            self._log("clean up complete")
        self._reset()
        
    
    def predict(self, data):
        """ Performs a prediction.
        Args:
            data: a dictionary or list of dictionaries of the format expected from the "request_builder" function specified in the class' constructor
        Returns:
            The result of the "response_formatter" function(s) specified in the class' constructor
        """
        
        if not isinstance(data,list):
            data=[data]
            
        return parallel_prediction.parallel_batch_predictions(
            self.model_names,
            data,
            self.host,
            self.port, 
            request_builder=self.request_builder, 
            response_formatter=self.response_formatter)
    
        
       
        
    def __enter__(self):
        """ Enter function for the disposable pattern, it enables the "with statement" usage.
            This function is an alias for "start()"
        """
        
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_value, tb):
        """ Exit function for the disposable pattern, it enables the "with statement" usage.
            This function is an alias for "stop()"
        """
        
        self.stop()
        if exc_type is not None:
            print exc_type
            print exc_value
            traceback.print_tb(tb)
        return self
