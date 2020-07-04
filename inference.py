#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.network_inputs = None
        self.network_outputs = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device='CPU', cpu_extension=None):
        ### TODO: Load the model ###
        self.plugin = IECore()       
        model_bin_path = os.path.splitext(model)[0] + ".bin"
        self.network = IENetwork(model=model, weights=model_bin_path)
             
        ### TODO: Check for supported layers ###
        layers_supported = self.plugin.query_network(self.network, device_name=device)
        
        layers = self.network.layers.keys()
        for layer in layers:
            if layer not in layers_supported:
                ### TODO: Add any necessary extensions ###
                self.plugin.add_extension(cpu_extension, device)
                break          
     
        self.exec_network = self.plugin.load_network(self.network, device)
        self.network_inputs = next(iter(self.network.inputs))
        self.network_outputs = next(iter(self.network.outputs))
       
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        return
    
    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.network.inputs[self.network_inputs].shape

    def exec_net(self, request_id, net_input):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.infer_request_handle = self.exec_network.start_async(request_id, inputs={self.network_inputs: net_input})
        return
    
    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.infer_request_handle.wait()

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.infer_request_handle.outputs[self.network_outputs]
