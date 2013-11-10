'''
Very simple and limited server/client.

Server runs forever and calls into the provided handler for every
request.  Client sends a message and blocks to receive a response.

Example usage in a server process:

>>> # Reply to the client with every word, uppercased
>>> def handle(payload, fout):
        for word in payload.split():
            print >>fout, word.upper()
>>> srv = Server(handle, verbose=1)
>>> srv.start()

Example usage in a client process:

>>> client_send_receive('this is not my beautiful house')

THIS
IS
NOT
MY
BEAUTIFUL
HOUSE
'''

import socket
import SocketServer
import sys


HOST, PORT = 'localhost', 9999


def get_handler_wrapper(handler, verbose=0):
    class HandlerWrapper(SocketServer.StreamRequestHandler):
        '''
        The RequestHandler class for our server.

        It is instantiated once per connection to the server, and must
        override the handle() method to implement communication to the
        client.
        '''

        def handle(self):
            data = self.rfile.readline().strip()

            if verbose >= 1:
                print ''
                print '*** Request from {}:'.format(self.client_address[0])
                print data
                print '*' * 60
                print ''

            # self.request is the TCP socket connected to the client
            f = self.request.makefile('w')
            response = handler(data, f) or ''
            self.request.sendall(response + '\n')
            f.close()

    return HandlerWrapper


class Server(object):
    def __init__(self, handler, host=HOST, port=PORT, verbose=0):
        '''
        The parameter `handler` is a function that takes two arguments:

        1. The received request payload
        2. A file object into which to provide output
        '''
        self.socket_server = SocketServer.TCPServer(
            (host, port),
            get_handler_wrapper(handler, verbose=verbose))

    def start(self):
        self.socket_server.serve_forever()


def client_send_receive(data, out=sys.stdout, host=HOST, port=PORT, verbose=0):
    '''
    Send the payload `data` to the server, get back a response and
    print it to the output stream `out`.
    '''

    # Create a socket (SOCK_STREAM means a TCP socket)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    if verbose >= 1:
        print ''
        print '*** Sending to {}:{}'.format(host, port)
        print data
        print '*' * 60
        print ''

    try:
        # Connect to server and send data
        sock.connect((host, port))
        sock.sendall(data + '\n')

        # Receive data from the server and shut down
        for l in sock.makefile('r'):
            print >>out, l.strip()
    finally:
        sock.close()
