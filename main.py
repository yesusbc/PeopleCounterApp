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
    ### Connect to the MQTT client ###
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
    
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, CPU_EXTENSION)
    network_shape = infer_network.get_input_shape()
    input_shape = network_shape['image_tensor']
    
    
    
    ### Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    
    current_count = 0
    total_count = 0
    duration = 0
    
    person_in_frame = False
    count = 0
    none_count = 0
    detected = 0
    time_tracker = 0
    time_avg = 0
    
    
    ### Loop until stream is over ###
    while cap.isOpened():
        ### Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
            
        h = frame.shape[0]
        w = frame.shape[1]
        
        ### Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        # raise RuntimeError(p_frame.shape)
        p_frame.reshape(1, *p_frame.shape)

        
        net_input = {'image_tensor': p_frame,'image_info': (600,600,1)}
        
        ### Start asynchronous inference for specified request ###
        infer_network.exec_net(request_id, net_input)
        
        ### Wait for the result ###
        if infer_network.wait(request_id) == 0:
            ### Get the results of the inference request ###
            result = infer_network.get_output(request_id)
            probs = result[0, 0, :, 2]

            ### Extract any desired stats from the results ###
            ### Calculate and send relevant information on ###
            current_count = 0
            for i, p in enumerate(probs):
                if p > prob_threshold:
                    current_count += 1
                    box = result[0, 0, i, 3:]
                    p1 = (int(box[0] * w), int(box[1] * h))
                    p2 = (int(box[2] * w), int(box[3] * h))
                    frame = cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
                    detected = 1
            
            if detected:
                count += 1
                none_count = 0
            else:
                none_count += 1
                count = 0
            
            detected = 0
                
            if count == 5 and person_in_frame == False:
                person_in_frame = True
                current_count = 1
                time_tracker = time.time()
                count = 0
                none_count = 0
            elif none_count == 5 and person_in_frame == True:
                person_in_frame = False
                current_count = 0
                time_tracker = time.time() - time_tracker
                count = 0
                none_count = 0
                total_count += 1
                time_avg += time_tracker
                time_tracker = 0
                duration = round(time_avg / total_count)
                
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
            client.publish("person", json.dumps({"count": current_count, "total": total_count})) 
            if duration:
                client.publish("person/duration", json.dumps({"duration": duration}))

        ### Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        
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
