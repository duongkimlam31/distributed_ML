import sys
import glob

sys.path.append('gen-py')
sys.path.insert(0, glob.glob('../../thrift-0.19.0/lib/py/build/lib*')[0])

from distributed_ml import coordinator
from distributed_ml.ttypes import communication_data

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

if __name__ == "__main__":
  if len(sys.argv) < 6:
    print("python3 client.py <coordinator_ip> <coordinator_port> <dir_path> <rounds> <epochs>")
    exit()
  coordinator_ip = sys.argv[1]
  coordinator_port = sys.argv[2]
  dir_path = sys.argv[3]
  rounds = int(sys.argv[4])
  epochs = int(sys.argv[5])
  
  # Make socket to contact the server
  transport = TSocket.TSocket(coordinator_ip, coordinator_port)
  
  # Buffering is critical. Raw sockets are very slow
  transport = TTransport.TBufferedTransport(transport)
    
  # Wrap in a protocol
  protocol = TBinaryProtocol.TBinaryProtocol(transport)

  # Create a client to use the protocol encoder
  client = coordinator.Client(protocol)
  
  # Connect!
  transport.open()
  
  # Give task to the server
  print(client.train(dir_path, rounds, epochs, 20, 26, 0.0001))
  