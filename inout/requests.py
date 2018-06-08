import tensorflow as tf

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def build_estimator_request(request, data):    
    COLUMNS=data["COLUMNS"]
    FIELD_TYPES=data["FIELD_TYPES"]
    feats=data["features"]
    
    feature_dict={}

    for i,c in enumerate(COLUMNS):
        if FIELD_TYPES[c]=="string":
            feature_dict[c]=_bytes_feature(value=feats[i].encode())
        if FIELD_TYPES[c]=="number":
            feature_dict[c]=_float_feature(value=feats[i])


    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    serialized = example.SerializeToString()

    request.inputs['inputs'].CopyFrom(
        tf.contrib.util.make_tensor_proto(serialized, shape=[1]))
    
    return request


class EstimatorRequest:
    def __init__(self,COLUMNS,FIELD_TYPES):
        self.columns=COLUMNS
        self.field_types=FIELD_TYPES
        
    def __call__(self,request, data):
        return build_estimator_request(request,{
            "COLUMNS":self.columns,
            "FIELD_TYPES":self.field_types,
            "features":data
        })