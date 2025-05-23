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

from thrift.transport import TTransport
all_structs = []


class communication_data(object):
    """
    Attributes:
     - fname
     - _V
     - _W
     - epochs
     - eta
     - delay

    """
    thrift_spec = None


    def __init__(self, fname = None, _V = None, _W = None, epochs = None, eta = None, delay = None,):
        self.fname = fname
        self._V = _V
        self._W = _W
        self.epochs = epochs
        self.eta = eta
        self.delay = delay

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
                    self.fname = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.LIST:
                    self._V = []
                    (_etype3, _size0) = iprot.readListBegin()
                    for _i4 in range(_size0):
                        _elem5 = []
                        (_etype9, _size6) = iprot.readListBegin()
                        for _i10 in range(_size6):
                            _elem11 = iprot.readDouble()
                            _elem5.append(_elem11)
                        iprot.readListEnd()
                        self._V.append(_elem5)
                    iprot.readListEnd()
                else:
                    iprot.skip(ftype)
            elif fid == 3:
                if ftype == TType.LIST:
                    self._W = []
                    (_etype15, _size12) = iprot.readListBegin()
                    for _i16 in range(_size12):
                        _elem17 = []
                        (_etype21, _size18) = iprot.readListBegin()
                        for _i22 in range(_size18):
                            _elem23 = iprot.readDouble()
                            _elem17.append(_elem23)
                        iprot.readListEnd()
                        self._W.append(_elem17)
                    iprot.readListEnd()
                else:
                    iprot.skip(ftype)
            elif fid == 4:
                if ftype == TType.I32:
                    self.epochs = iprot.readI32()
                else:
                    iprot.skip(ftype)
            elif fid == 5:
                if ftype == TType.DOUBLE:
                    self.eta = iprot.readDouble()
                else:
                    iprot.skip(ftype)
            elif fid == 6:
                if ftype == TType.BOOL:
                    self.delay = iprot.readBool()
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
        oprot.writeStructBegin('communication_data')
        if self.fname is not None:
            oprot.writeFieldBegin('fname', TType.STRING, 1)
            oprot.writeString(self.fname.encode('utf-8') if sys.version_info[0] == 2 else self.fname)
            oprot.writeFieldEnd()
        if self._V is not None:
            oprot.writeFieldBegin('_V', TType.LIST, 2)
            oprot.writeListBegin(TType.LIST, len(self._V))
            for iter24 in self._V:
                oprot.writeListBegin(TType.DOUBLE, len(iter24))
                for iter25 in iter24:
                    oprot.writeDouble(iter25)
                oprot.writeListEnd()
            oprot.writeListEnd()
            oprot.writeFieldEnd()
        if self._W is not None:
            oprot.writeFieldBegin('_W', TType.LIST, 3)
            oprot.writeListBegin(TType.LIST, len(self._W))
            for iter26 in self._W:
                oprot.writeListBegin(TType.DOUBLE, len(iter26))
                for iter27 in iter26:
                    oprot.writeDouble(iter27)
                oprot.writeListEnd()
            oprot.writeListEnd()
            oprot.writeFieldEnd()
        if self.epochs is not None:
            oprot.writeFieldBegin('epochs', TType.I32, 4)
            oprot.writeI32(self.epochs)
            oprot.writeFieldEnd()
        if self.eta is not None:
            oprot.writeFieldBegin('eta', TType.DOUBLE, 5)
            oprot.writeDouble(self.eta)
            oprot.writeFieldEnd()
        if self.delay is not None:
            oprot.writeFieldBegin('delay', TType.BOOL, 6)
            oprot.writeBool(self.delay)
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
all_structs.append(communication_data)
communication_data.thrift_spec = (
    None,  # 0
    (1, TType.STRING, 'fname', 'UTF8', None, ),  # 1
    (2, TType.LIST, '_V', (TType.LIST, (TType.DOUBLE, None, False), False), None, ),  # 2
    (3, TType.LIST, '_W', (TType.LIST, (TType.DOUBLE, None, False), False), None, ),  # 3
    (4, TType.I32, 'epochs', None, None, ),  # 4
    (5, TType.DOUBLE, 'eta', None, None, ),  # 5
    (6, TType.BOOL, 'delay', None, None, ),  # 6
)
fix_spec(all_structs)
del all_structs
