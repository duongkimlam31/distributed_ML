import sys
import glob

sys.path.append('../lib')
sys.path.append('gen-py')
sys.path.insert(0, glob.glob('../../thrift-0.19.0/lib/py/build/lib*')[0])

import ML

from distributed_ml import coordinator
from distributed_ml import compute_node
from distributed_ml.ttypes import communication_data

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
import threading
import time

signal = threading.Condition()

class ComputeNode:
  def __init__(self, port, load_probability):
    self.port = port
    self.load_probability = load_probability
    self.mlp = ML.mlp()
    
    # Make socket to contact the server
    transport = TSocket.TSocket("localhost", self.port)

    # Buffering is critical. Raw sockets are very slow
    transport = TTransport.TBufferedTransport(transport)
      
    # Wrap in a protocol
    protocol = TBinaryProtocol.TBinaryProtocol(transport)

    # Create a client to use the protocol encoder
    self.client = coordinator.Client(protocol)

    # Connect!
    transport.open()
  
  def local_train(self):
    completed_task = 0
    while True:
      try:
        # Get data from the server to initialize mlp model
        response = self.client.pull_data(float(self.load_probability))
        if response.delay:
          time.sleep(3)
        # If a task is rejected restart
        if response.fname == "":
          continue
        # set and train the model using the shared_model from the main program
        self.mlp.init_training_model(response.fname, response._V, response._W)
        
        # Save the original V and W values
        orig_V, orig_W = self.mlp.get_weights()
        # print(f"Validation error rate before training: {self.mlp.validate(response.fname)}")
        error_rate = self.mlp.train(response.eta, response.epochs)
        # print(f"Validation error rate after training: {self.mlp.validate(response.fname)}")
        
        # Get the current V and W values to calculate the gradient
        curr_V, curr_W = self.mlp.get_weights()
        
        gradient_V = ML.calc_gradient(curr_V, orig_V)
        gradient_W = ML.calc_gradient(curr_W, orig_W)
        
        # Push data back to the server
        self.client.push_data(gradient_V=gradient_V, gradient_W=gradient_W)
        completed_task += 1
        print("Completed task,", completed_task)
      except Thrift.TException as tx:
        print(f"Error: {tx.message}")
        break
    
class ServerHandler:
  def __init__(self):
    pass
  def wait_coordinator(self):
    signal.acquire()
    signal.notify()
    signal.release()
    print("Received signal from coordinator")

def start_server(port):
  handler = ServerHandler()
  processor = compute_node.Processor(handler)
  transport = TSocket.TServerSocket("localhost", port)
  tfactory = TTransport.TBufferedTransportFactory()
  pfactory = TBinaryProtocol.TBinaryProtocolFactory()
  
  server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
  print(f"Server started on port {port}")
  server.serve()
  
  
if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("python3 compute_node.py <port> <load_probability>")
    exit()
  port = sys.argv[1]
  load_probability = sys.argv[2]
  
  server_thread = threading.Thread(target=start_server, args=(port,))
  server_thread.start()
  
  # Wait for the signal from the coordinator to start
  signal.acquire()
  signal.wait()
  signal.release()
  
  compute_node = ComputeNode(9090, load_probability)
  compute_node.local_train()
  