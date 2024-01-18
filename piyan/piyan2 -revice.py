import os
from sys import flags
from tkinter import Frame
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
#import RPi.GPIO as GPIO
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import math
import random
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
def main(_argv):
    #setup led
    #GPIO.setmode(GPIO.BOARD)
    #LEDR=11
    #LEDG=12
    #GPIO.setup(LEDR,GPIO.OUT)
    #GPIO.setup(LEDG,GPIO.OUT)
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    tempx0 = 0
    tempy0 = 0
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    #get total frames & total times of a video
    videoCapture=cv2.VideoCapture(FLAGS.video)
    #  Frame rate (frames per second)
    totalfps = videoCapture.get(cv2.CAP_PROP_FPS)
    #  The total number of frames (frames)
    #totalframes = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    #print(" frames ："+str(totalfps))
    #print(" The total number of frames ："+str(totalframes))
    #print(" Total video duration ："+"{0:.2f}".format(totalframes/totalfps)+" second ")

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    # while video is running

    tempall = np.zeros(shape=(100,3)) 

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame = cv2.resize(frame,(1280,720))
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size,input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()
        
        '''
        x1=200
        x2=800
        y1=100
        y2=460
        y3=640
        slope=((y1-y2)/(x2-x1))
        pts=np.array([[x1,y3],[x1,y2],[x2,y1],[x2,y3]])
        cv2.polylines(frame,[pts],True,(19,89,64),5)       '''
        
        cv2.rectangle(frame,(400,720),(1280,500),(255,0,0),5)
        

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['car','cell phone','bottle','remote','motorbike']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)

        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        counter=0
        
        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        

        
        

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            
            plotx=(int(bbox[0])+int(bbox[2]))/2
            ploty=(int(bbox[1])+int(bbox[3]))/2
        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
            '''if (x2>int(bbox[0])>x1 and y3>int(bbox[1]) and (slope*(bbox[0]-x1)-(bbox[1]-y2))<0) or (x2>int(bbox[2])>x1 and y3>int(bbox[3]) and (slope*(bbox[2]-x1)-(bbox[3]-y2))<0):
               print("gg ez 7414 welcome to piyan party")
                counter+=1'''
            #if x2>int(bbox[2])>x1 and y3>int(bbox[3]) and (slope*(bbox[2]-x1)-(bbox[3]-y2))<0:
                #print("gg ez 7414 welcome to piyan party")
            if 1280>ploty>400 and 720>plotx>500:
                counter += 1
                
            #print velocity of people or car
            

            #road_width
            rw = 10/340
        
            bbox_x1 = (float(bbox[1])+float(bbox[3]))/2
            bbox_y1 = (float(bbox[0])+float(bbox[2]))/2
        
            #bbox_x1 = (int(bbox_speed[1])+int(bbox_speed[3]))/2
            disx=tempall[track.track_id][1]-bbox_x1
            #print('{}'.format(str(track.track_id)),bbox_x1,'==========',disx,'===============',int(track.track_id),'==========',tempall[int(track.track_id)][1])
            
            firstfps=1
            if tempall[track.track_id][1]==0 and tempall[track.track_id][2]==0:
                firstfps=0  
            #remove the first fps 

            tempall[track.track_id][1]=bbox_x1
            #bbox_y1 = (int(bbox_speed[0])+int(bbox_speed[2]))/2
            disy=tempall[track.track_id][2]-bbox_y1
            #print(disx)
            #print(disy)
            tempall[track.track_id][2]=bbox_y1
            #if num >= 1:
            dis = (math.sqrt(disx*disx+disy*disy))*rw
            distemp = (math.sqrt(disx*disx+disy*disy))
            #print(distemp)
            
            if firstfps==1:
                expin=float(tempall[track.track_id][1])
                print(tempall[track.track_id][1])
                if tempall[track.track_id][1]>500:
                    revice = math.exp(abs((720-250)/(expin-250))) *3600 / 1000
                else:
                    revice=1
                velocity_car =  dis * totalfps * revice
#
            #print(dis)
            #print(totalfps)
            #print(velocity_car)
            
            if firstfps==1:
                print('ID:{} car`s velocity is {:.2f} km/hr'.format(track.track_id,velocity_car))

            count = 0

            
                
            

            for t in np.arange(0,2,0.1):    
                n_trace = t * totalfps
                
                #print('{} : x ={:.2f} y ={:.2f}'.format(track.track_id, bbox_x1 - disx * n_trace, bbox_y1 - disy * n_trace))
                
                if 400 <(bbox_x1 - disx * n_trace) <1280 and 500 < (bbox_y1 - disy * n_trace) <720 and count ==0 and firstfps==1 :
                    #print('Number: {} You are in danger'.format(track.track_id))
                    #print('ID:{} car`s velocity is {:.2f} km/hr'.format(track.track_id,velocity_car))
                    count += 1

            
        '''if counter > 0 :
            print("gg ez 7414 welcome to piyan party")'''
        if counter>5:
           # GPIO.output(LEDG,GPIO.HIGH)
            #time.sleep(0.075)
           # GPIO.output(LEDG,GPIO.LOW)
            print(int(counter),">5")
        elif counter<=5:
            #GPIO.output(LEDR,GPIO.HIGH)
            #time.sleep(0.075)
            #GPIO.output(LEDR,GPIO.LOW)
            print(int(counter),"<=5")
        
        # calculate frames per second of running detections
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cv2.destroyAllWindows()


        





    #num += 1
'''def whether_touch(bbox_speed,fps_velocity,ID):
    print(bbox_x1)
    if 480 >= (int(bbox_speed[1])+int(bbox_speed[3]))/2:
        min_x = 480 - 
'''



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass