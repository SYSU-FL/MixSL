import os
import struct
import socket
import pickle
import time


def send_msg(sock, msg):
    msg = pickle.dumps(msg)
    length = len(msg)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)
    return length

def recv_msg(sock):
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    msg = recvall(sock, msglen)
    msg = pickle.loads(msg)
    return msg, msglen

def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data