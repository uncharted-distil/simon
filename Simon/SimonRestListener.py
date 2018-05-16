from flask import Flask, request
from json import JSONEncoder
import pandas
import pickle
import numpy as np
import configparser
import time

from Simon import *
from Simon.Encoder import *
from Simon.DataGenerator import *
from Simon.LengthStandardizer import *

from Simon.penny.guesser import guess

class SimonRestListener:
    """ SimonRestListener accepts a pandas dataframe (or a path to one),
    uses its predictive model to generate labels for the columns,
    and then encodes this in JSON to be returned to the caller
    """
    def __init__(self, modelName):
       self.model =  modelName
       self.encoder = JSONEncoder()

    def runModel(self, frame, p_threshold):

        # setup model as you typically would in a Simon main file
        maxlen = 20
        max_cells = 500
        p_threshold = 0.5

        DEBUG = True # boolean to specify whether or not print DEBUG information

        checkpoint_dir = "/clusterfiles/scripts/pretrained_models/"

        with open('/clusterfiles/scripts/Categories.txt','r') as f:
            Categories = f.read().splitlines()

        # orient the user a bit
        print("fixed categories are: ")
        Categories = sorted(Categories)
        print(Categories)
        category_count = len(Categories)

        execution_config=modelName

        # load specified execution configuration
        if execution_config is None:
            raise TypeError
        Classifier = Simon(encoder={}) # dummy text classifier

        config = Classifier.load_config(execution_config, checkpoint_dir)
        encoder = config['encoder']
        checkpoint = config['checkpoint']

        X = encoder.encodeDataFrame(frame)
            
        # build classifier model
        model = Classifier.generate_model(maxlen, max_cells, category_count)
        Classifier.load_weights(checkpoint, None, model, checkpoint_dir)
        model_compile = lambda m: m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
        model_compile(model)
        y = model.predict(X)
        # discard empty column edge case
        y[np.all(frame.isnull(),axis=0)]=0

        result = encoder.reverse_label_encode(y,p_threshold)

        ## LABEL COMBINED DATA AS CATEGORICAL/ORDINAL
        print("Beginning Guessing categorical/ordinal classifications...")
        start_time_guess = time.time()
        category_count = 0
        ordinal_count = 0
        raw_data = frame.as_matrix()
        for i in np.arange(raw_data.shape[1]):
            tmp = guess(raw_data[:,i], for_types ='category')
            if tmp[0]=='category':
                category_count += 1
                tmp2 = list(result[0][i])
                tmp2.append('categorical')
                result[0][i] = tuple(tmp2)
                result[1][i].append(1)
                if ('int' in result[1][i]) or ('float' in result[1][i]) \
                    or ('datetime' in result[1][i]):
                        ordinal_count += 1
                        tmp2 = list(result[0][i])
                        tmp2.append('ordinal')
                        result[0][i] = tuple(tmp2)
                        result[1][i].append(1)
        elapsed_time = time.time()-start_time_guess
        print("Total statistical variable guessing time is : %.2f sec" % elapsed_time)
        ## FINISHED LABELING COMBINED DATA AS CATEGORICAL/ORDINAL

        Classifier.clear_session()
        
        return self.encoder.encode((result))

    def predict(self, request_data,p_threshold):
        frame = pickle.loads(request_data)
                
        return self.runModel(frame,p_threshold)

    def predictFile(self, fileName,p_threshold):
        frame = pandas.read_csv(str(fileName),dtype='str')
        return self.runModel(frame,p_threshold)
        
config = configparser.ConfigParser()
config.read('config.ini')
modelName = config['DEFAULT']['modelName']

# modelName = "Base.pkl" # consider specifying explicitly before building each docker image
print("using model " + modelName + " ...")
        
listener = SimonRestListener(modelName)

app = Flask(__name__)

@app.route("/", methods=['POST'])
def predict():
    """ Listen for data being POSTed on root. The data expected to 
    be a string representation of a pickled pandas frame
    """
    request.get_data()
    p_threshold=0.5 # this will become class-dependent and tunable shortly (backwards compatible)
    return listener.predict(request.data,p_threshold)


@app.route("/fileName", methods=['POST'])
def predictFile():
    """ Listen for data being POSTed on root. The data should 
    be a string representation of a file name accessible to flask server
    """
    request.get_data()
    p_threshold=0.5 # this will become class-dependent and tunable shortly (backwards compatible)
    return listener.predictFile(request.data.decode("utf-8"),p_threshold)
    
@app.route("/fileUpload", methods=['POST'])
def predictUploadedFile():
    """ Listen for data being POSTed on root. The data should 
    be a string representation of a pickled numpy array
    """
    request.get_data()
    file = request.files['file']
    fileName = '/clusterfiles/uploaded_file.csv'
    file.save(fileName)
    p_threshold=0.5 # this will become class-dependent and tunable shortly (backwards compatible)
    result = listener.predictFile(fileName,p_threshold)
    os.remove(fileName)
    
    return result

# $env:FLASK_APP="rest/SimonRestListener.py"
# flask run