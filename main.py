"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    """
    Connect to the MQTT client
    """
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    request_id = 0

    # Flags
    is_person_in_frame = False
    detection_in_frame = 0
    single_image_mode = False
    
    # Counters
    none_detection_counter = 0
    frame_detection_counter = 0
    last_count_of_people_in_frame = 0
    total_people_count = 0

    # Time trackers
    time_tracker = 0
    time_avg = 0
    duration = 0
    
    # Initialise the class
    infer_network = Network()
    
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    load_time_start = time.time()
    # Load the model through `infer_network` 
    infer_network.load_model(args.model, args.device, CPU_EXTENSION)
    load_time_total = time.time() - load_time_start
    
    # network_shape = infer_network.get_input_shape()    # Faster RCNN Model
    # input_shape = network_shape['image_tensor']    # Faster RCNN Model
    input_shape = infer_network.get_input_shape()  
    input_name = infer_network.get_input_name()
    
    # Handle the input stream
    if args.input == "CAM":
        # Checks for Webcam
        # input_file = 0
        input_file = -1
    elif args.input.endswith(".jpg") or args.input.endswith(".png"):
        # Checks for input image 
        single_image_mode = True
        input_file = args.input
    else:
        # Checks for input video
        input_file = args.input
        
    cap = cv2.VideoCapture(input_file)
    if input_file:
        cap.open(input_file)
    
    # Loop until stream is over
    while cap.isOpened():
        # Read from the video capture
        flag, frame = cap.read()
        if not flag:
            break
          
        # Pre-process the image as needed
        p_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame.reshape(1, *p_frame.shape)
        
        # input_dict = {'image_tensor': p_frame,'image_info': (600,600,1)}    # Faster RCNN Model
        input_dict = {input_name : p_frame}
        
        infer_start_time = time.time()
        # Start asynchronous inference for specified request
        infer_network.exec_net(request_id, input_dict)
        
        # Wait for the result...
        if infer_network.wait(request_id) == 0:
            # Get the results of the inference request
            result = infer_network.get_output(request_id)
            probs = result[0, 0, :, 2]

            # Extract predictions and draw bounding boxes
            people_in_frame = 0
            h = frame.shape[0]
            w = frame.shape[1]
            for idx, prob in enumerate(probs):
                if prob > prob_threshold:
                    people_in_frame += 1
                    box = result[0, 0, idx, 3:]
                    p1 = (int(box[0] * w), int(box[1] * h))
                    p2 = (int(box[2] * w), int(box[3] * h))
                    frame = cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
                    detection_in_frame = True
                    if last_count_of_people_in_frame < people_in_frame:
                        last_count_of_people_in_frame = people_in_frame
                        
            infer_total_time = time.time() - infer_start_time
            
            # A repeated amount of frames will confirm if the person is detected or not
            if detection_in_frame:
                frame_detection_counter += 1
                none_detection_counter = 0
            else:
                none_detection_counter += 1
                frame_detection_counter = 0
                
            # Reset flag
            detection_in_frame = False
                
            
            if frame_detection_counter == 20 and is_person_in_frame == False:
                # There was no people at frame, but now there is so start timer and count person
                is_person_in_frame = True
                time_tracker = time.time()
                # Reset counters
                frame_detection_counter = 0
                none_detection_counter = 0
            
            elif none_detection_counter == 20 and is_person_in_frame == True:
                # There was people in frame, now there is not, stop timer and reset counter
                # Add # of people to counter
                is_person_in_frame = False
                total_people_count += last_count_of_people_in_frame
                time_tracker = time.time() - time_tracker
                time_avg += time_tracker
                duration = round(time_avg / total_people_count)
                # Reset counters
                frame_detection_counter = 0
                none_detection_counter = 0
                time_tracker = 0
                
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
            client.publish("person", json.dumps({"count": people_in_frame, "total": total_people_count})) 
            if duration:
                client.publish("person/duration", json.dumps({"duration": duration}))

        ### Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()

        # Write an output image if `single_image_mode`
        if single_image_mode:
            cv2.imwrite("images/output.jpg", frame)
            print("Total inference time: {0}", infer_total_time)
            print("Model time loading: {0}", load_time_total)
    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
