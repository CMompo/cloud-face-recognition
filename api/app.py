# TODO: Maybe wait after writing blocked to fs.
#       Maybe add a wawy for the training to know if someone is accessing data.
#   Current way could have problems if by some reason, two add petitions were performed simultaneously
#   Also, the currently.training flag must be ignored if it is x time old to avoid stucking the whole system.
#       (Implement as a configurable timeout)
#   Check if a folder is empty before training

import io
import os
import time
import random
import socket
import requests
import glob
from pathlib import Path
import mysql.connector
import shutil
import joblib
from PIL import Image
from flask import Flask
from flask_restplus import Api, Resource, fields, abort, inputs
from werkzeug.datastructures import FileStorage
from face_recognition import preprocessing
from training import train 

preprocess = preprocessing.ExifOrientationNormalize()

app = Flask(__name__)
api = Api(app, version='1.4b', title='Face Recognition Framework',
    description='A face recognition API for cloud deployment with the capability of running multiple replicas concurrently without conflicts.',
    doc='/docs')

face_recogniser = None
fr_filename = ""

IMAGE_KEY = 'image'
ID_KEY = 'ID'
IMAGE_KEY = 'image'
INCLUDE_PREDICTIONS_KEY = 'include_predictions'
CAM_ID_KEY = 'camera_id'
LOCATION_ID_KEY='location_id'
DATASET_DIR_PATH = 'nfs/dataset'
MODEL_DIR_PATH = 'nfs/models'
DB_HOST="10.0.1.1"
DB_USER="kub"
DB_PASSWORD="4yXNPkrc"
DB_DATABASE="face-recognition-db"
BLOCKED_PATH = os.path.join(DATASET_DIR_PATH,'currently.training')
            
# For Adding Images
add_img_parser = api.parser()
add_img_parser.add_argument(IMAGE_KEY, type=FileStorage, location='files', required=True,
                    help='Image to add.')

# For Face Recognition Get Results
fr_parser = api.parser()
fr_parser.add_argument(IMAGE_KEY, type=FileStorage, location='files', required=True,
                    help='Image with which the face recognition will be performed.')
fr_parser.add_argument(INCLUDE_PREDICTIONS_KEY, type=inputs.boolean, default=False,
                    help='Whether to include all possible predictions and their probability for each face in the response.')

# For Face Recognition Send Results
frsr_parser = api.parser()
frsr_parser.add_argument(IMAGE_KEY, type=FileStorage, location='files', required=True,
                    help='Image with which the face recognition will be performed.')
frsr_parser.add_argument(CAM_ID_KEY, type=str, required=True, default = False,
                    help='Camera identifier.')
frsr_parser.add_argument(LOCATION_ID_KEY, type=str, required=True, default = False,
                    help='Location identifier.')

IP_model = api.model('IPAddress', {
    'IP_address': fields.String
})

node_name_model = api.model('NodeName', {
    'node_name': fields.String
})

people = api.model('PeopleTableEntries', {
    'face_id': fields.String
})

people_model = api.model('PeopleInDB', {
    'people': fields.List(fields.Nested(people))
})

detections = api.model('DetectionsTableEntries', {
    'location': fields.String,
    'detected_by_camera_id': fields.String,
    'timestamp': fields.String
})

detections_model = api.model('LastDetection', {
    'last_known_location': fields.Nested(detections)
})

dataset_face = api.model('DatasetFaceID', {
    'id_name': fields.String,
    'number_of_images': fields.String
})

dataset_model = api.model('DatasetContents', {
    'face_ids': fields.Nested(dataset_face)
})

bounding_box = api.model('BoundingBox', {
    'left': fields.Float,
    'top': fields.Float,
    'right': fields.Float,
    'bottom': fields.Float,
})

prediction = api.model('Prediction', {
    'label': fields.String,
    'confidence': fields.Float
})

face_model = api.model('Face', {
    'top_prediction': fields.Nested(prediction),
    'bounding_box': fields.Nested(bounding_box),
    'all_predictions': fields.List(fields.Nested(prediction))
})

fr_response_model = api.model('RecognizedFaces', {
    'faces': fields.List(fields.Nested(face_model))
})

error_model = api.model('ErrorResponse', {
    'error_message': fields.String
})

faces_sent = api.model('FacesSent', {
    'top_prediction': fields.String,
    'url': fields.String,
    'response': fields.String
})

frsr_response_model = api.model('DetectionDataSent', {
    'faces': fields.Nested(faces_sent)
})

# Get the IP of the node (Debugging)
@api.route('/debug/IP')
class GetIP(Resource):
    @api.response(200, 'IP address obtained correctly', IP_model)
    @api.response(400, 'Error while obtaining ip address', error_model)
    @api.marshal_with(IP_model)
    def get(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return {'IP_address': str(IP)}

# Get the Name of the node (Debugging)
@api.route('/debug/node-name')
class GetName(Resource):
    @api.response(200, 'Node name obtained correctly', node_name_model)
    @api.response(400, 'Error while obtaining node name', error_model)
    @api.marshal_with(node_name_model)
    def get(self):
        return {'node_name': os.uname().nodename}

# Remove the dataset blocking
@api.route('/debug/unblock-dataset')
class UnblockDataset(Resource):
    @api.response(200, 'The dataset was or has been unlocked')
    @api.response(400, 'Error while unlocking the dataset', error_model)
    def post(self):
        if(os.path.isfile(BLOCKED_PATH)):
            os.remove(BLOCKED_PATH)
        return

# Get the Entries in the People Table  
@api.route('/database/people')
class DBPeople(Resource):
    @api.response(200, 'People retrieved correctly', people_model)
    @api.response(400, 'Error while retrieving people', error_model)
    @api.marshal_with(people_model)
    def get(self):
        # Configuration of the database access
        fr_db = mysql.connector.connect(
          host=DB_HOST,
          user=DB_USER,
          password=DB_PASSWORD,
          database=DB_DATABASE
        )
        db_cursor = fr_db.cursor()
        db_cursor.execute("SELECT Face_ID FROM people")
        
        json_data = {
                    'people': [
                        {
                            'face_id': str(Face_ID)
                        }
                        for Face_ID in db_cursor
                    ]
                }
        db_cursor.close()
        fr_db.close()
        return json_data
            

# Find the last know location of someone
@api.route('/database/<string:ID>/last-known-location')
class DBLastLocationID(Resource):
    @api.response(200, 'Last location retrieved correctly', detections_model)
    @api.response(400, 'Error while retrieving last location', error_model)
    @api.marshal_with(detections_model)
    def get(self, ID):
        # Change spaces with underscores
        ID = ID.lower().replace(' ', '_')
        # Configuration of the database access
        fr_db = mysql.connector.connect(
          host=DB_HOST,
          user=DB_USER,
          password=DB_PASSWORD,
          database=DB_DATABASE
        )
        db_cursor = fr_db.cursor()
        query = ("SELECT Location, Camera_ID, Timestamp FROM detections "+
        "WHERE Face_ID=%s ORDER BY Timestamp DESC LIMIT 1")
        db_cursor.execute(query, (str(ID),))
        json_data = {
                'last_known_location': [
                    {
                        'location': str(Location),
                        'detected_by_camera_id': str(Camera_ID),
                        'timestamp': str(Timestamp)
                    }
                    for (Location, Camera_ID, Timestamp) in db_cursor
                ]
            }
        db_cursor.close()
        fr_db.close()
        return json_data

# Generate/Train Model
@api.route('/model/train')
class Retrain(Resource):
    @api.response(200, 'Successfully trained')
    @api.response(400, 'Error while training', error_model)
    def post(self):
        if os.path.isfile(BLOCKED_PATH):
            abort(400, "The server is currently processing a dataset blocking request")
        Path(BLOCKED_PATH).touch()
        try:
            train.train_as_lib(DATASET_DIR_PATH, MODEL_DIR_PATH);
        except Exception as e:
            abort(400, "An error has ocurred while training: "+str(e))
        finally:
            # This must be done even if an error occurs while training
            os.remove(BLOCKED_PATH)

# Generate Embeddings
@api.route('/model/generate-emb')
class GenEmb(Resource):
    @api.response(200, 'Embeddings successfully generated')
    @api.response(400, 'Error while generating embeddings', error_model)
    def post(self):
        if os.path.isfile(BLOCKED_PATH):
            abort(400, "The server is currently processing a dataset blocking request")
        Path(BLOCKED_PATH).touch()
        try:
            train.dataset_to_embeddings_lib(DATASET_DIR_PATH);
        except Exception as e:
            abort(400, "An error has ocurred while generating the embeddings : "+str(e))
        finally:
            # This must be done even if an error occurs while training
            os.remove(BLOCKED_PATH)

# Generate/Train Model with Existing Embeddings
@api.route('/model/optimized-train')
class OptimizedRetrain(Resource):
    @api.response(200, 'Successfully trained')
    @api.response(400, 'Error while training', error_model)
    def post(self):
        if os.path.isfile(BLOCKED_PATH):
            abort(400, "The server is currently processing a dataset blocking request")
        Path(BLOCKED_PATH).touch()
        try:
            train.load_and_train_as_lib(DATASET_DIR_PATH, MODEL_DIR_PATH);
        except Exception as e:
            abort(400, "An error has ocurred while training: "+str(e))
        finally:
            # This must be done even if an error occurs while training
            os.remove(BLOCKED_PATH)

# Get ID names and the number of pictures per ID
@api.route('/dataset')
class GetIDs(Resource):
    @api.response(200, 'Successfully obtained the Face IDs in dataset', dataset_model)
    @api.response(400, 'Error while obtaining the Face IDs in dataset', error_model)
    @api.marshal_with(dataset_model)
    def get(self):
        files = os.listdir(DATASET_DIR_PATH)
        directories = []
        sizes = []
        for i in range(len(files)):
            id_dir = os.path.join(DATASET_DIR_PATH, files[i])
            if(os.path.isdir(id_dir)):
                directories.append(files[i])

        return \
            {
                'face_ids': [
                    {
                        'id_name': directory,
                        'number_of_images': str(len(os.listdir(os.path.join(DATASET_DIR_PATH, directory))))
                    }
                    for directory in directories
                ]
            }

# Add Img to ID
@api.route('/dataset/<string:ID>/image')
class AddImg(Resource):
    @api.expect(add_img_parser, validate=True)
    @api.response(200, 'Image added successfully')
    @api.response(400, 'Error adding image', error_model)
    def post(self, ID):
        args = add_img_parser.parse_args()
        if IMAGE_KEY not in args:
            abort(400, "Image field '{}' doesn't exist in request!".format(IMAGE_KEY))
        if os.path.isfile(BLOCKED_PATH):
            abort(400, "The server is currently processing a dataset blocking request")
        Path(BLOCKED_PATH).touch()
        try:
            # Change spaces with underscores
            ID = ID.lower().replace(' ', '_')
            id_path = os.path.join(DATASET_DIR_PATH,ID)
            # Access restricted to the directiries inside the dataset path
            if os.path.realpath(id_path).startswith(os.path.join(os.getcwd(),DATASET_DIR_PATH)+'/') and os.path.isdir(id_path):
                length = len([name for name in os.listdir(id_path) if os.path.isfile(id_path+'/'+name)])
                img_name = ID+'_'+str(length)+'.png'
                img = Image.open(io.BytesIO(args[IMAGE_KEY].read()))
                img_path = os.path.join(id_path,img_name)
                img.save(img_path)
                train.add_embeddings_for_img(DATASET_DIR_PATH, img_path, ID);
            else:
                abort(400, "ID "+ID+" does not exist.")
        except Exception as e:
            abort(400, "Error while adding or training image: "+str(e))
        finally:
            # This must be done even if an error occurs while training
            os.remove(BLOCKED_PATH)
       
# Add or Delete ID
@api.route('/dataset/<string:ID>')
class AddRemoveID(Resource):
    @api.response(200, 'ID added successfully')
    @api.response(400, 'Error adding ID', error_model)
    def post(self, ID):
        if os.path.isfile(BLOCKED_PATH):
            abort(400, "The server is currently processing a dataset blocking request")
        # Change spaces with underscores
        ID = ID.lower().replace(' ', '_')
        id_path = os.path.join(DATASET_DIR_PATH,ID)
        # Access restricted to the directiries inside the dataset path
        if os.path.realpath(id_path).startswith(os.path.join(os.getcwd(),DATASET_DIR_PATH)+'/') and not os.path.isdir(id_path):
            os.mkdir(id_path)
            # Configuration of the database access
            fr_db = mysql.connector.connect(
              host=DB_HOST,
              user=DB_USER,
              password=DB_PASSWORD,
              database=DB_DATABASE
            )
            db_cursor = fr_db.cursor()
            query = ("INSERT INTO `people`(`Face_ID`) "+
            "VALUES (%s)")
            db_cursor.execute(query, (str(ID),))
            fr_db.commit()
            db_cursor.close()
            fr_db.close()
            return
        else:
            abort(400, "ID "+ID+" already exists.")
            
    @api.response(200, 'ID removed successfully')
    @api.response(400, 'Error removing ID', error_model)
    def delete(self, ID):
        if os.path.isfile(BLOCKED_PATH):
            abort(400, "The server is currently processing a dataset blocking request")
        Path(BLOCKED_PATH).touch()
        try:
            # Change spaces with underscores
            ID = ID.lower().replace(' ', '_')
            id_path = os.path.join(DATASET_DIR_PATH,ID)
            # Access restricted to the directiries inside the dataset path
            if os.path.realpath(id_path).startswith(os.path.join(os.getcwd(),DATASET_DIR_PATH)+'/') and os.path.isdir(id_path):
                shutil.rmtree(id_path)
                # Remove embeddings
                train.remove_embeddings_for_id(DATASET_DIR_PATH, ID)
                # Configuration of the database access
                fr_db = mysql.connector.connect(
                  host=DB_HOST,
                  user=DB_USER,
                  password=DB_PASSWORD,
                  database=DB_DATABASE
                )
                db_cursor = fr_db.cursor()
                query = "DELETE FROM `people` WHERE Face_ID=%s"
                db_cursor.execute(query, (str(ID),))
                fr_db.commit()
                db_cursor.close()
                fr_db.close()
                return
            else:
                abort(400, "ID "+ID+" does not exist.")
        except Exception as e:
            abort(400, "Error while removing the ID and images: "+str(e))
        finally:
            # This must be done even if an error occurs while training
            os.remove(BLOCKED_PATH)

@api.route('/face-recognition/get-results')
class FaceRecognitionGetResponse(Resource):
    @api.expect(fr_parser, validate=True)
    @api.response(200, 'Face(s) recognized successfully', fr_response_model)
    @api.response(400, 'Error while recognizing faces', error_model)
    @api.marshal_with(fr_response_model)
    def post(self):
        args = fr_parser.parse_args()
        if IMAGE_KEY not in args:
            abort(400, "Image field '{}' doesn't exist in request!".format(IMAGE_KEY))

        # Get latest model
        list_of_files = glob.glob(MODEL_DIR_PATH+'/*.pkl')
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)
            global face_recogniser
            global fr_filename
            if(not fr_filename==latest_file):
                face_recogniser = joblib.load(os.path.join(latest_file))
                fr_filename=latest_file
        else:
            abort(400, "No model available. Please, train the classifier first.")
        
        img = Image.open(io.BytesIO(args[IMAGE_KEY].read()))
        img = preprocess(img)
        # convert image to RGB (stripping alpha channel if exists)
        img = img.convert('RGB')
        faces = face_recogniser(img)
        return \
            {
                'faces': [
                    {
                        'top_prediction': face.top_prediction._asdict(),
                        'bounding_box': face.bb._asdict(),
                        'all_predictions': [p._asdict() for p in face.all_predictions] if
                        args[INCLUDE_PREDICTIONS_KEY] else None
                    }
                    for face in faces
                ]
            }

@api.route('/face-recognition/send-results')
class FaceRecognitionSendResponse(Resource):
    @api.expect(frsr_parser, validate=True)
    @api.response(200, 'Face(s) recognized successfully', frsr_response_model)
    @api.response(400, 'Error while recognizing faces', error_model)
    @api.marshal_with(frsr_response_model)
    def post(self):
        args = frsr_parser.parse_args()
        if IMAGE_KEY not in args:
            abort(400, "Image field '{}' doesn't exist in request!".format(IMAGE_KEY))
        if CAM_ID_KEY not in args:
            abort(400, "Camera field '{}' doesn't exist in request!".format(CAM_ID_KEY))
        if LOCATION_ID_KEY not in args:
            abort(400, "Location field '{}' doesn't exist in request!".format(LOCATION_ID_KEY))
        # Get latest model
        list_of_files = glob.glob(MODEL_DIR_PATH+'/*.pkl')
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)
            global face_recogniser
            global fr_filename
            if(not fr_filename==latest_file):
                face_recogniser = joblib.load(os.path.join(latest_file))
                fr_filename=latest_file
        else:
            abort(400, "No model available. Please, train the classifier first.")
        img = Image.open(io.BytesIO(args[IMAGE_KEY].read()))
        img = preprocess(img)
        # convert image to RGB (stripping alpha channel if exists)
        img = img.convert('RGB')
        faces = face_recogniser(img)
        # Send information to database
        for face in faces:
            person_id = str(face.top_prediction._asdict()['label'])
            camera_id = str(args[CAM_ID_KEY])
            location_id = str(args[LOCATION_ID_KEY])
            url = 'https://us-central1-ece-597---tfm.cloudfunctions.net/addPerson'
            parameters={'id':person_id, 'camera':camera_id, 'blueprint':location_id}
            try:
                request = requests.get(url, parameters, timeout=10)
            except Exception as e:
                abort(400, "Error performing request to external server: "+str(e))
        return \
            {
                'faces': [
                    {
                        'top_prediction': str(face.top_prediction._asdict()['label']),
                        'url': url,
                        'response': str(request.text)
                    }
                    for face in faces
                ]
            }

@api.route('/face-recognition/store-results')
class FaceRecognitionStoreResponse(Resource):
    @api.expect(frsr_parser, validate=True)
    @api.response(200, 'Face(s) recognized successfully')
    @api.response(400, 'Error while recognizing faces', error_model)
    def post(self):
        args = frsr_parser.parse_args()
        if IMAGE_KEY not in args:
            abort(400, "Image field '{}' doesn't exist in request!".format(IMAGE_KEY))
        if CAM_ID_KEY not in args:
            abort(400, "Camera field '{}' doesn't exist in request!".format(CAM_ID_KEY))
        if LOCATION_ID_KEY not in args:
            abort(400, "Location field '{}' doesn't exist in request!".format(LOCATION_ID_KEY))
        # Get latest model
        list_of_files = glob.glob(MODEL_DIR_PATH+'/*.pkl')
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)
            global face_recogniser
            global fr_filename
            if(not fr_filename==latest_file):
                face_recogniser = joblib.load(os.path.join(latest_file))
                fr_filename=latest_file
        else:
            abort(400, "No model available. Please, train the classifier first.")
        img = Image.open(io.BytesIO(args[IMAGE_KEY].read()))
        img = preprocess(img)
        # convert image to RGB (stripping alpha channel if exists)
        img = img.convert('RGB')
        faces = face_recogniser(img)
        # Send information to database
        for face in faces:
            face_id = str(face.top_prediction._asdict()['label'])
            camera_id = str(args[CAM_ID_KEY])
            location = str(args[LOCATION_ID_KEY])
            try:
                # Configuration of the database access
                fr_db = mysql.connector.connect(
                    host=DB_HOST,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    database=DB_DATABASE
                )
                db_cursor = fr_db.cursor()
                query = ("INSERT INTO `detections`(`Face_ID`, `Location`, `Camera_ID`)"+
                " VALUES (%s, %s, %s)")
                db_cursor.execute(query, (str(face_id),str(location),str(camera_id)))
                fr_db.commit()
                db_cursor.close()
                fr_db.close()
            except Exception as e:
                abort(400, "Error performing request to database: "+str(e))
        return


@api.route('/face-recognition/dry-run')
class FaceRecognitionDryRun(Resource):
    @api.expect(fr_parser, validate=True)
    @api.response(200, 'Face images received successfully')
    @api.response(400, 'Error', error_model)
    def post(self):
        args = fr_parser.parse_args()
        if IMAGE_KEY not in args:
            abort(400, "Image field '{}' doesn't exist in request!".format(IMAGE_KEY))
        return
    
if __name__ == '__main__':
    app.run(host='0.0.0.0')
