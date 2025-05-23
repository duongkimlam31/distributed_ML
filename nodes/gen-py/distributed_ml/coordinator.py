#
# Autogenerated by Thrift Compiler (0.21.0)
#
# DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
#
#  options string: py
#

from thrift.Thrift import TType, TMessageType, TFrozenDict, TException, TApplicationException
from thrift.protocol.TProtocol import TProtocolException
from thrift.TRecursive import fix_spec
from uuid import UUID

import sys
import logging
from .ttypes import *
from thrift.Thrift import TProcessor
from thrift.transport import TTransport
all_structs = []


class Iface(object):
    def train(self, dir, rounds, epochs, h, k, eta):
        """
        Parameters:
         - dir
         - rounds
         - epochs
         - h
         - k
         - eta

        """
        pass

    def pull_data(self, load_probability):
        """
        Parameters:
         - load_probability

        """
        pass

    def push_data(self, gradient_V, gradient_W):
        """
        Parameters:
         - gradient_V
         - gradient_W

        """
        pass

    def contact(self):
        pass


class Client(Iface):
    def __init__(self, iprot, oprot=None):
        self._iprot = self._oprot = iprot
        if oprot is not None:
            self._oprot = oprot
        self._seqid = 0

    def train(self, dir, rounds, epochs, h, k, eta):
        """
        Parameters:
         - dir
         - rounds
         - epochs
         - h
         - k
         - eta

        """
        self.send_train(dir, rounds, epochs, h, k, eta)
        return self.recv_train()

    def send_train(self, dir, rounds, epochs, h, k, eta):
        self._oprot.writeMessageBegin('train', TMessageType.CALL, self._seqid)
        args = train_args()
        args.dir = dir
        args.rounds = rounds
        args.epochs = epochs
        args.h = h
        args.k = k
        args.eta = eta
        args.write(self._oprot)
        self._oprot.writeMessageEnd()
        self._oprot.trans.flush()

    def recv_train(self):
        iprot = self._iprot
        (fname, mtype, rseqid) = iprot.readMessageBegin()
        if mtype == TMessageType.EXCEPTION:
            x = TApplicationException()
            x.read(iprot)
            iprot.readMessageEnd()
            raise x
        result = train_result()
        result.read(iprot)
        iprot.readMessageEnd()
        if result.success is not None:
            return result.success
        raise TApplicationException(TApplicationException.MISSING_RESULT, "train failed: unknown result")

    def pull_data(self, load_probability):
        """
        Parameters:
         - load_probability

        """
        self.send_pull_data(load_probability)
        return self.recv_pull_data()

    def send_pull_data(self, load_probability):
        self._oprot.writeMessageBegin('pull_data', TMessageType.CALL, self._seqid)
        args = pull_data_args()
        args.load_probability = load_probability
        args.write(self._oprot)
        self._oprot.writeMessageEnd()
        self._oprot.trans.flush()

    def recv_pull_data(self):
        iprot = self._iprot
        (fname, mtype, rseqid) = iprot.readMessageBegin()
        if mtype == TMessageType.EXCEPTION:
            x = TApplicationException()
            x.read(iprot)
            iprot.readMessageEnd()
            raise x
        result = pull_data_result()
        result.read(iprot)
        iprot.readMessageEnd()
        if result.success is not None:
            return result.success
        raise TApplicationException(TApplicationException.MISSING_RESULT, "pull_data failed: unknown result")

    def push_data(self, gradient_V, gradient_W):
        """
        Parameters:
         - gradient_V
         - gradient_W

        """
        self.send_push_data(gradient_V, gradient_W)
        self.recv_push_data()

    def send_push_data(self, gradient_V, gradient_W):
        self._oprot.writeMessageBegin('push_data', TMessageType.CALL, self._seqid)
        args = push_data_args()
        args.gradient_V = gradient_V
        args.gradient_W = gradient_W
        args.write(self._oprot)
        self._oprot.writeMessageEnd()
        self._oprot.trans.flush()

    def recv_push_data(self):
        iprot = self._iprot
        (fname, mtype, rseqid) = iprot.readMessageBegin()
        if mtype == TMessageType.EXCEPTION:
            x = TApplicationException()
            x.read(iprot)
            iprot.readMessageEnd()
            raise x
        result = push_data_result()
        result.read(iprot)
        iprot.readMessageEnd()
        return

    def contact(self):
        self.send_contact()
        self.recv_contact()

    def send_contact(self):
        self._oprot.writeMessageBegin('contact', TMessageType.CALL, self._seqid)
        args = contact_args()
        args.write(self._oprot)
        self._oprot.writeMessageEnd()
        self._oprot.trans.flush()

    def recv_contact(self):
        iprot = self._iprot
        (fname, mtype, rseqid) = iprot.readMessageBegin()
        if mtype == TMessageType.EXCEPTION:
            x = TApplicationException()
            x.read(iprot)
            iprot.readMessageEnd()
            raise x
        result = contact_result()
        result.read(iprot)
        iprot.readMessageEnd()
        return


class Processor(Iface, TProcessor):
    def __init__(self, handler):
        self._handler = handler
        self._processMap = {}
        self._processMap["train"] = Processor.process_train
        self._processMap["pull_data"] = Processor.process_pull_data
        self._processMap["push_data"] = Processor.process_push_data
        self._processMap["contact"] = Processor.process_contact
        self._on_message_begin = None

    def on_message_begin(self, func):
        self._on_message_begin = func

    def process(self, iprot, oprot):
        (name, type, seqid) = iprot.readMessageBegin()
        if self._on_message_begin:
            self._on_message_begin(name, type, seqid)
        if name not in self._processMap:
            iprot.skip(TType.STRUCT)
            iprot.readMessageEnd()
            x = TApplicationException(TApplicationException.UNKNOWN_METHOD, 'Unknown function %s' % (name))
            oprot.writeMessageBegin(name, TMessageType.EXCEPTION, seqid)
            x.write(oprot)
            oprot.writeMessageEnd()
            oprot.trans.flush()
            return
        else:
            self._processMap[name](self, seqid, iprot, oprot)
        return True

    def process_train(self, seqid, iprot, oprot):
        args = train_args()
        args.read(iprot)
        iprot.readMessageEnd()
        result = train_result()
        try:
            result.success = self._handler.train(args.dir, args.rounds, args.epochs, args.h, args.k, args.eta)
            msg_type = TMessageType.REPLY
        except TTransport.TTransportException:
            raise
        except TApplicationException as ex:
            logging.exception('TApplication exception in handler')
            msg_type = TMessageType.EXCEPTION
            result = ex
        except Exception:
            logging.exception('Unexpected exception in handler')
            msg_type = TMessageType.EXCEPTION
            result = TApplicationException(TApplicationException.INTERNAL_ERROR, 'Internal error')
        oprot.writeMessageBegin("train", msg_type, seqid)
        result.write(oprot)
        oprot.writeMessageEnd()
        oprot.trans.flush()

    def process_pull_data(self, seqid, iprot, oprot):
        args = pull_data_args()
        args.read(iprot)
        iprot.readMessageEnd()
        result = pull_data_result()
        try:
            result.success = self._handler.pull_data(args.load_probability)
            msg_type = TMessageType.REPLY
        except TTransport.TTransportException:
            raise
        except TApplicationException as ex:
            logging.exception('TApplication exception in handler')
            msg_type = TMessageType.EXCEPTION
            result = ex
        except Exception:
            logging.exception('Unexpected exception in handler')
            msg_type = TMessageType.EXCEPTION
            result = TApplicationException(TApplicationException.INTERNAL_ERROR, 'Internal error')
        oprot.writeMessageBegin("pull_data", msg_type, seqid)
        result.write(oprot)
        oprot.writeMessageEnd()
        oprot.trans.flush()

    def process_push_data(self, seqid, iprot, oprot):
        args = push_data_args()
        args.read(iprot)
        iprot.readMessageEnd()
        result = push_data_result()
        try:
            self._handler.push_data(args.gradient_V, args.gradient_W)
            msg_type = TMessageType.REPLY
        except TTransport.TTransportException:
            raise
        except TApplicationException as ex:
            logging.exception('TApplication exception in handler')
            msg_type = TMessageType.EXCEPTION
            result = ex
        except Exception:
            logging.exception('Unexpected exception in handler')
            msg_type = TMessageType.EXCEPTION
            result = TApplicationException(TApplicationException.INTERNAL_ERROR, 'Internal error')
        oprot.writeMessageBegin("push_data", msg_type, seqid)
        result.write(oprot)
        oprot.writeMessageEnd()
        oprot.trans.flush()

    def process_contact(self, seqid, iprot, oprot):
        args = contact_args()
        args.read(iprot)
        iprot.readMessageEnd()
        result = contact_result()
        try:
            self._handler.contact()
            msg_type = TMessageType.REPLY
        except TTransport.TTransportException:
            raise
        except TApplicationException as ex:
            logging.exception('TApplication exception in handler')
            msg_type = TMessageType.EXCEPTION
            result = ex
        except Exception:
            logging.exception('Unexpected exception in handler')
            msg_type = TMessageType.EXCEPTION
            result = TApplicationException(TApplicationException.INTERNAL_ERROR, 'Internal error')
        oprot.writeMessageBegin("contact", msg_type, seqid)
        result.write(oprot)
        oprot.writeMessageEnd()
        oprot.trans.flush()

# HELPER FUNCTIONS AND STRUCTURES


class train_args(object):
    """
    Attributes:
     - dir
     - rounds
     - epochs
     - h
     - k
     - eta

    """
    thrift_spec = None


    def __init__(self, dir = None, rounds = None, epochs = None, h = None, k = None, eta = None,):
        self.dir = dir
        self.rounds = rounds
        self.epochs = epochs
        self.h = h
        self.k = k
        self.eta = eta

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.STRING:
                    self.dir = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.I32:
                    self.rounds = iprot.readI32()
                else:
                    iprot.skip(ftype)
            elif fid == 3:
                if ftype == TType.I32:
                    self.epochs = iprot.readI32()
                else:
                    iprot.skip(ftype)
            elif fid == 4:
                if ftype == TType.I32:
                    self.h = iprot.readI32()
                else:
                    iprot.skip(ftype)
            elif fid == 5:
                if ftype == TType.I32:
                    self.k = iprot.readI32()
                else:
                    iprot.skip(ftype)
            elif fid == 6:
                if ftype == TType.DOUBLE:
                    self.eta = iprot.readDouble()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        self.validate()
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('train_args')
        if self.dir is not None:
            oprot.writeFieldBegin('dir', TType.STRING, 1)
            oprot.writeString(self.dir.encode('utf-8') if sys.version_info[0] == 2 else self.dir)
            oprot.writeFieldEnd()
        if self.rounds is not None:
            oprot.writeFieldBegin('rounds', TType.I32, 2)
            oprot.writeI32(self.rounds)
            oprot.writeFieldEnd()
        if self.epochs is not None:
            oprot.writeFieldBegin('epochs', TType.I32, 3)
            oprot.writeI32(self.epochs)
            oprot.writeFieldEnd()
        if self.h is not None:
            oprot.writeFieldBegin('h', TType.I32, 4)
            oprot.writeI32(self.h)
            oprot.writeFieldEnd()
        if self.k is not None:
            oprot.writeFieldBegin('k', TType.I32, 5)
            oprot.writeI32(self.k)
            oprot.writeFieldEnd()
        if self.eta is not None:
            oprot.writeFieldBegin('eta', TType.DOUBLE, 6)
            oprot.writeDouble(self.eta)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)
all_structs.append(train_args)
train_args.thrift_spec = (
    None,  # 0
    (1, TType.STRING, 'dir', 'UTF8', None, ),  # 1
    (2, TType.I32, 'rounds', None, None, ),  # 2
    (3, TType.I32, 'epochs', None, None, ),  # 3
    (4, TType.I32, 'h', None, None, ),  # 4
    (5, TType.I32, 'k', None, None, ),  # 5
    (6, TType.DOUBLE, 'eta', None, None, ),  # 6
)


class train_result(object):
    """
    Attributes:
     - success

    """
    thrift_spec = None


    def __init__(self, success = None,):
        self.success = success

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 0:
                if ftype == TType.DOUBLE:
                    self.success = iprot.readDouble()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        self.validate()
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('train_result')
        if self.success is not None:
            oprot.writeFieldBegin('success', TType.DOUBLE, 0)
            oprot.writeDouble(self.success)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)
all_structs.append(train_result)
train_result.thrift_spec = (
    (0, TType.DOUBLE, 'success', None, None, ),  # 0
)


class pull_data_args(object):
    """
    Attributes:
     - load_probability

    """
    thrift_spec = None


    def __init__(self, load_probability = None,):
        self.load_probability = load_probability

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.DOUBLE:
                    self.load_probability = iprot.readDouble()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        self.validate()
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('pull_data_args')
        if self.load_probability is not None:
            oprot.writeFieldBegin('load_probability', TType.DOUBLE, 1)
            oprot.writeDouble(self.load_probability)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)
all_structs.append(pull_data_args)
pull_data_args.thrift_spec = (
    None,  # 0
    (1, TType.DOUBLE, 'load_probability', None, None, ),  # 1
)


class pull_data_result(object):
    """
    Attributes:
     - success

    """
    thrift_spec = None


    def __init__(self, success = None,):
        self.success = success

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 0:
                if ftype == TType.STRUCT:
                    self.success = communication_data()
                    self.success.read(iprot)
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        self.validate()
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('pull_data_result')
        if self.success is not None:
            oprot.writeFieldBegin('success', TType.STRUCT, 0)
            self.success.write(oprot)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)
all_structs.append(pull_data_result)
pull_data_result.thrift_spec = (
    (0, TType.STRUCT, 'success', [communication_data, None], None, ),  # 0
)


class push_data_args(object):
    """
    Attributes:
     - gradient_V
     - gradient_W

    """
    thrift_spec = None


    def __init__(self, gradient_V = None, gradient_W = None,):
        self.gradient_V = gradient_V
        self.gradient_W = gradient_W

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.LIST:
                    self.gradient_V = []
                    (_etype31, _size28) = iprot.readListBegin()
                    for _i32 in range(_size28):
                        _elem33 = []
                        (_etype37, _size34) = iprot.readListBegin()
                        for _i38 in range(_size34):
                            _elem39 = iprot.readDouble()
                            _elem33.append(_elem39)
                        iprot.readListEnd()
                        self.gradient_V.append(_elem33)
                    iprot.readListEnd()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.LIST:
                    self.gradient_W = []
                    (_etype43, _size40) = iprot.readListBegin()
                    for _i44 in range(_size40):
                        _elem45 = []
                        (_etype49, _size46) = iprot.readListBegin()
                        for _i50 in range(_size46):
                            _elem51 = iprot.readDouble()
                            _elem45.append(_elem51)
                        iprot.readListEnd()
                        self.gradient_W.append(_elem45)
                    iprot.readListEnd()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        self.validate()
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('push_data_args')
        if self.gradient_V is not None:
            oprot.writeFieldBegin('gradient_V', TType.LIST, 1)
            oprot.writeListBegin(TType.LIST, len(self.gradient_V))
            for iter52 in self.gradient_V:
                oprot.writeListBegin(TType.DOUBLE, len(iter52))
                for iter53 in iter52:
                    oprot.writeDouble(iter53)
                oprot.writeListEnd()
            oprot.writeListEnd()
            oprot.writeFieldEnd()
        if self.gradient_W is not None:
            oprot.writeFieldBegin('gradient_W', TType.LIST, 2)
            oprot.writeListBegin(TType.LIST, len(self.gradient_W))
            for iter54 in self.gradient_W:
                oprot.writeListBegin(TType.DOUBLE, len(iter54))
                for iter55 in iter54:
                    oprot.writeDouble(iter55)
                oprot.writeListEnd()
            oprot.writeListEnd()
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)
all_structs.append(push_data_args)
push_data_args.thrift_spec = (
    None,  # 0
    (1, TType.LIST, 'gradient_V', (TType.LIST, (TType.DOUBLE, None, False), False), None, ),  # 1
    (2, TType.LIST, 'gradient_W', (TType.LIST, (TType.DOUBLE, None, False), False), None, ),  # 2
)


class push_data_result(object):
    thrift_spec = None


    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        self.validate()
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('push_data_result')
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)
all_structs.append(push_data_result)
push_data_result.thrift_spec = (
)


class contact_args(object):
    thrift_spec = None


    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        self.validate()
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('contact_args')
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)
all_structs.append(contact_args)
contact_args.thrift_spec = (
)


class contact_result(object):
    thrift_spec = None


    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        self.validate()
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('contact_result')
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)
all_structs.append(contact_result)
contact_result.thrift_spec = (
)
fix_spec(all_structs)
del all_structs
