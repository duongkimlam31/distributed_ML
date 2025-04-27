# Machines tested: 
Server: duong210@csel-kh1250-17
Client: duong210@csel-kh1250-16
Compute Node 1: duong210@csel-remote-lnx-01
Compute Node 2: duong210@csel-kh1250-01
Compute Node 3: duong210@csel-kh1262-22
Compute Node 4: duong210@csel-kh1262-13

# System design and operation:

The system consists of 3 components: the client, the server, and the compute nodes. To simplify the synchronization between the coordinator and the compute nodes, the system
is designed in a producer-consumer fashion.

## Program execution sequence:
1) Start the compute nodes
2) Start the server
3) Start the client

## The compute node:
python3 compute_node.py <port> <load_probability>
port is the port number of the compute node and it is used to create a server

The compute node starts as a server that waits for a connection from the coordinator. Since it acts as both a server and a client a threading mechanism is used to handle
this situation. Specifically, a thread is used to handle the server section while the other is used to handle the client section. Once the coordinator contacts the compute node,
the compute node becomes a client and connect to the server. To prevent the compute node becoming the client before the server contacts it, a locking mechanism is used to block
such situation. After the compute node becomes a client, it starts to extract the data from the work queue and start training the data locally. To achieve this, it uses the RPC
mechaninsm in which pull() pulls the weights from the server while push() pushes the gradients to the server. This process is repeated indefinitely or until there is no longer any 
data left in the queue. The compute node is treated as a consumer.

## The coordinator:
python3 coordinator.py <port> <scheduling policy>

For the scheduling policy argument, 1 stands for random scheduling while 2 stands for load-balancing scheduling.

The coordinator starts by contacting all the compute nodes specified in the compute_nodes.txt file to make sure that all the compute nodes are online before initiating the training section.
Once all the compute nodes are contacted, the server waits for the client's training request. After the client gives it a request, it uses the data from that request to start the training section.
Before the training starts, the coordinator contacts all the compute nodes to turn them into clients using the wait_coordinator() function and acts as the server. Once the coordinator becomes the 
server, it spawns a thread for each compute node and each compute node starts pulling data from the work queue as they are spawned. To prevent the compute nodes from pulling an empty queue or 
pulling the same work at the same time, the system uses the semaphore mechanism for synchronization. Every training round, the server acts as the producer which pushes the work (file) to the work queue so that the compute nodes can extract them to work locally. After pushing all the work to the work queue, the server waits for all the compute nodes to finish all the work in the queue. To
achieve this, the system uses the barrier mechanism which is implemented using condition variable. The condition for the coordinator to continue is when the amount of jobs completed by the compute
nodes equal to the total amount of jobs. The training stops when the coordinator runs a specific amount of rounds indicated by the client.

## The client: 
python3 client.py <coordinator_ip> <coordinator_port> <dir_path> <rounds> <epochs>

The design of the client is simple. First, it takes in the user inputs and use these inputs to set up a connection to the server. After that, it runs the train function
on the server. In other words, it initiates the shared model training and prints the validation value once it is done. 

# Test cases and performance evaluation:

## Test case 1:
login01:        load probability = 0.2
csel-kh1262-13: load probability = 0.2
csel-kh1262-22: load probability = 0.2
csel-kh1250-01: load probability = 0.2
Scheduling policy: load-balancing scheduling
Parameters: rounds = 25, epochs = 15, h = 20, k = 26, eta = 0.0001

Results:
time taken ~= 7 minutes
validation error = 0.324

login01:        task completed = 68 tasks
csel-kh1262-13: task completed = 67 tasks
csel-kh1262-22: task completed = 69 tasks
csel-kh1250-01: task completed = 71 tasks

## Test case 2:
login01:        load probability = 0.2
csel-kh1262-13: load probability = 0.2
csel-kh1262-22: load probability = 0.2
csel-kh1250-01: load probability = 0.2
Scheduling policy: random scheduling
Parameters: rounds = 25, epochs = 15, h = 20, k = 26, eta = 0.0001

Results:
time taken ~= 7 minutes
validation error = 0.324

login01:        task completed = 74 tasks
csel-kh1262-13: task completed = 64 tasks
csel-kh1262-22: task completed = 59 tasks
csel-kh1250-01: task completed = 78 tasks

## Test case 3:
login01:        load probability = 0.8
csel-kh1262-13: load probability = 0.8
csel-kh1262-22: load probability = 0.8
csel-kh1250-01: load probability = 0.8
Scheduling policy: load-balancing scheduling
Parameters: rounds = 25, epochs = 15, h = 20, k = 26, eta = 0.0001

Results:
time taken ~= 11 minutes
validation error = 0.324

login01:        task completed = 68 tasks
csel-kh1262-13: task completed = 66 tasks
csel-kh1262-22: task completed = 69 tasks
csel-kh1250-01: task completed = 72 tasks

## Test case 4:
login01:        load probability = 0.8
csel-kh1262-13: load probability = 0.8
csel-kh1262-22: load probability = 0.8
csel-kh1250-01: load probability = 0.8
Scheduling policy: random scheduling
Parameters: rounds = 25, epochs = 15, h = 20, k = 26, eta = 0.0001

Results:
time taken ~= 9 minutes
validation error = 0.324

login01:        task completed = 72 tasks
csel-kh1262-13: task completed = 58 tasks
csel-kh1262-22: task completed = 69 tasks
csel-kh1250-01: task completed = 76 tasks

## Test case 5:
login01:        load probability = 0.4
csel-kh1262-13: load probability = 0.25
csel-kh1262-22: load probability = 0.25
csel-kh1250-01: load probability = 0.1
Scheduling policy: load-balancing scheduling
Parameters: rounds = 25, epochs = 15, h = 20, k = 26, eta = 0.0001

Results:
time taken ~= 6 minutes
validation error = 0.324

login01:        task completed = 59 tasks
csel-kh1262-13: task completed = 69 tasks
csel-kh1262-22: task completed = 58 tasks
csel-kh1250-01: task completed = 89 tasks

## Test case 6:
login01:        load probability = 0.4
csel-kh1262-13: load probability = 0.25
csel-kh1262-22: load probability = 0.25
csel-kh1250-01: load probability = 0.1
Scheduling policy: random scheduling
Parameters: rounds = 25, epochs = 15, h = 20, k = 26, eta = 0.0001

Results:
time taken ~= 7 minutes
validation error = 0.324

login01:        task completed = 67 tasks
csel-kh1262-13: task completed = 57 tasks
csel-kh1262-22: task completed = 70 tasks
csel-kh1250-01: task completed = 81 tasks