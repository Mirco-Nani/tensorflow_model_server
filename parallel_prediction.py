from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from tensorflow.python.saved_model import signature_constants

from google.protobuf import json_format
import json
import base64
import time
from multiprocessing import Process, Queue

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

def make_prediction(host, port, model_name, data, key, request_builder, response_formatter, debug=False):
    
    start_time = time.time()
    #print("RPC CHANNEL CREATION START..")
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    #print("RPC CHANNEL CREATION END --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    #print("REQUEST BUILDING START..")
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    #request.inputs['image_bytes'].CopyFrom(tf.contrib.util.make_tensor_proto(data.decode('base64'), shape=[1]))
    #request.inputs['key'].CopyFrom(tf.contrib.util.make_tensor_proto(key, shape=[1]))
    request = request_builder(request, data)


    start_time = time.time()
    #print("PREDICTION START..")
    result = stub.Predict(request, 500.0)  # 500 secs timeout
    if debug:
        print("PREDICTION TIME --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    #print("RESPONSE BUILDING START..")
    formatted_result = response_formatter(result)
    #print("RESPONSE BUILDING END --- %s seconds ---" % (time.time() - start_time))

    return formatted_result #json_format.MessageToJson(result)


def enqueue_prediction(q, host, port, model_name, image, key, request_builder, response_formatter):
    try:
        res = make_prediction(host, port, model_name, image, key, request_builder, response_formatter)
        q.put({
            "success":True,
            "key":key,
            "result":res#{k:res[k] for k in res if k not in ["key"]}#make_prediction(host, port, model_name, image, key)
        })
    except Exception as e:
        q.put({"success":False, "key":key, "reason":str(e)})
    
def async_prediction(q, host, port, model_name, model_input, key, request_builder, response_formatter):
    p = Process(target=enqueue_prediction, args=(q,host, port, model_name, model_input, key, request_builder, response_formatter))
    p.start()
    return p


def islambda(v):
    #LAMBDA = lambda:0
    #return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__
    return callable(v)

def isdict(v):
    return isinstance(v,dict)

def parallel_batch_predictions(
    models,
    inputs,
    host="localhost",
    port="9000", 
    request_builder=build_image_key_request, 
    response_formatter=parsePredictResponse
):
    
    #print(request_builder,response_formatter)
    
    key_separator="<->"
    q = Queue()
    
    processes = []
    rb = request_builder
    rf = response_formatter
    if islambda(request_builder) and islambda(response_formatter):
        processes = [async_prediction(q, host, port, model, inp, str(i)+key_separator+model, rb, rf) 
                     for model in models
                     for i,inp in enumerate(inputs)
                    ]
    elif isdict(request_builder) and islambda(response_formatter):
        processes = [async_prediction(q, host, port, model, inp, str(i)+key_separator+model, rb[model], rf) 
                     for model in models
                     for i,inp in enumerate(inputs)
                    ]
    elif islambda(request_builder) and isdict(response_formatter):
        processes = [async_prediction(q, host, port, model, inp, str(i)+key_separator+model, rb, rf[model]) 
                     for model in models
                     for i,inp in enumerate(inputs)
                    ]
    elif isdict(request_builder) and isdict(response_formatter):
        processes = [async_prediction(q, host, port, model, inp, str(i)+key_separator+model, rb[model], rf[model]) 
                     for model in models
                     for i,inp in enumerate(inputs)
                    ]
    else:
        raise ValueError('request_builder and response_formatter must be either a key-value dict or a lambda function')

    results = [q.get() for p in processes]

    for p in processes:
        p.join()
        
    formatted_results=[{} for i in inputs]
    for res in results:
        key_values = res["key"].split(key_separator)
        img_position=int(key_values[0])
        model=key_values[1]

        formatted_results[img_position][model]={
            k:res[k] for k in res if k not in ["key"]
        }

    return formatted_results