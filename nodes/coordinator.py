import sys
import glob

sys.path.append('../lib')
sys.path.append('gen-py')
sys.path.insert(0, glob.glob('../../thrift-0.19.0/lib/py/build/lib*')[0])

import ML
import os

from distributed_ml import coordinator
from distributed_ml import compute_node
from distributed_ml.ttypes import communication_data

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
import threading
import random

class CoordinatorNode:
  def __init__(self, scheduling_policy):
    self.mlp = None
    self.shared_weight_V, self.shared_weight_W = [],[]
    self.shared_gradient_V, self.shared_gradient_W = [],[]
    self.work_queue = []
    self.job_count = 0
    self.job_completed = 0
    self.eta = 0
    self.epochs = 0
    self.scheduling_policy = scheduling_policy
    self.rejected = 0
    
    # Setting up semaphore for synchronization
    self.item_available = threading.Semaphore(0)
    self.space_available = threading.Semaphore(1024)
    self.mutex = threading.Semaphore(1)
    self.sum_mutex = threading.Semaphore(1)
    self.barrier = threading.Condition()
  
  def pull_data(self, load_probability):
    # get a file to train on 
    self.item_available.acquire()
    self.mutex.acquire()
    work = ""
    delay = False
    # Decide whether to delay a task or not (Random scheduling)
    injection = choice = random.choices(["inject", "don't inject"], weights=[load_probability, 1 - load_probability])
    if injection[0] == "inject":
      delay = True
    # If use load-balancing scheduling and no task has been rejected so far
    if self.scheduling_policy == "1" and self.rejected == 0:
      # Decide whether to reject or accept a task
      choice = random.choices(["reject", "accept"], weights=[load_probability, 1 - load_probability])[0]
      # Handle each case
      if choice == "accept": 
        work = self.work_queue.pop(0)
      else:
        self.rejected = 1
        self.mutex.release()
        self.item_available.release()
        return communication_data("", [], [], -1, -1, delay)
    # Send a task to another node if a node was recently rejected
    elif self.scheduling_policy == "1" and self.rejected == 1:
      work = self.work_queue.pop(0)
      self.rejected = 0
    # Handle random scheduling case
    elif self.scheduling_policy == "2":
        work = self.work_queue.pop(0)
    self.mutex.release()
    self.space_available.release()
    return communication_data(work, self.shared_weight_V,self.shared_weight_W, self.epochs, self.eta, delay)
  
  def push_data(self, gradient_v, gradient_w):
    # update the shared gradient
    self.sum_mutex.acquire()
    self.shared_gradient_V = ML.sum_matricies(self.shared_gradient_V, gradient_v)
    self.shared_gradient_W = ML.sum_matricies(self.shared_gradient_W, gradient_w)
    self.job_completed += 1
    self.sum_mutex.release()
    
    print(self.job_completed)
    print(self.job_count)
    # Notify barrier
    self.barrier.acquire()
    while (self.job_completed == self.job_count):
      self.barrier.notify()
      self.job_completed = 0
    self.barrier.release()
  
  def train(self, dir, rounds, epochs, h, k, eta):
    # initialize a model on the coordinator 
    self.mlp = ML.mlp()
    self.mlp.init_training_random("../letters/train_letters1.txt",k,h)
    
    # Save eta and epochs for the compute node
    self.eta = eta
    self.epochs = epochs
    
    # Traverse the data directory to get the total amount of jobs
    for file in os.listdir(dir):
      if file.startswith("validate"):
        continue
      self.job_count += 1
    
    validate = 0
    self.job_completed = 0
    
    # conduct a series of training rounds
    for i in range(rounds):
      
      # setup a shared gradient and weight model for threads
      self.shared_weight_V, self.shared_weight_W = self.mlp.get_weights()
      self.shared_gradient_V, self.shared_gradient_W = ML.scale_matricies(self.shared_weight_V, 0), ML.scale_matricies(self.shared_weight_W, 0)
      
      # add each training file to a work queue
      for file in os.listdir(dir):
        if file.startswith("validate"):
          continue
        self.space_available.acquire()
        self.mutex.acquire()
        self.work_queue.append(f"{dir}/{file}")
        self.mutex.release()
        self.item_available.release()
      
      # wait for threads to work through the queue 
      self.barrier.acquire()
      self.barrier.wait()
      self.barrier.release()
      
      # the shared gradient V and W should be the
      # average of the gradients calculated by each client
      self.shared_gradient_V = ML.scale_matricies(self.shared_gradient_V, 1 / self.job_count)
      self.shared_gradient_W = ML.scale_matricies(self.shared_gradient_W, 1 / self.job_count)
      self.mlp.update_weights(self.shared_gradient_V, self.shared_gradient_W)
      
      # You may want to print the validation error each round 
      validate = self.mlp.validate("../letters/validate_letters.txt")
      print(validate)
      
    return validate
  
def start_server():
  port = sys.argv[1]
  scheduling_policy = sys.argv[2]
  
  coordinator_node = CoordinatorNode(scheduling_policy)
  processor = coordinator.Processor(coordinator_node)
  transport = TSocket.TServerSocket(host='localhost', port=port)
  
  tfactory = TTransport.TBufferedTransportFactory()
  pfactory = TBinaryProtocol.TBinaryProtocolFactory()
  
  server = TServer.TThreadedServer(
        processor, transport, tfactory, pfactory)
  
  print('Starting the server...')
  server.serve()
  print('done.')
    
if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("python3 coordinator.py <port> <scheduling policy>")
    exit()
    
  server_thread = threading.Thread(target=start_server)
  server_thread.start()
   
  with open("../compute_nodes.txt","r") as file:
    content = file.read()
  
  nodes = content.split("\n")
  for i in range(len(nodes)):
    node = nodes[i].split(",")
    hostname = node[0]
    port = node[1]
  
    # Contact compute nodes
    transport = TSocket.TSocket(hostname, port)
    transport = TTransport.TBufferedTransport(transport)
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    client = compute_node.Client(protocol)

    transport.open()
    response = client.wait_coordinator()
    print("Contacted compute node")
    transport.close()