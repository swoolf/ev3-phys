#!/usr/bin/python

from time import sleep
import SimpleHTTPServer
import SocketServer
import os, sys
from livereload import Server, shell

global web

class webScreen:
    
    def __init__(self):
        self.file = ''
        self.PORT = 8000
        self.httpd = ''
    
    def serve(self):
        server = Server()
        server.watch('index.html')
        server.serve(port= self.PORT, host = '0.0.0.0')
    
    def endserve(self):
        self.httpd.shutdown()
        print 'ended'
    
    def openFile(self, name):
        self.file = open(name, 'w+')
    
    def closeFile(self):
        self.file.close()
    
    def deleteFile(self, name):
        os.remove(name)
    
    def writeTo(self, text):
        self.file.write("<html><body> \n")
        self.file.write('<HEAD><META HTTP-EQUIV="refresh" CONTENT=".1"></HEAD> \n')
        self.file.write(text + "\n")
        self.file.write("</body></html> \n")
    
    def post(text, name)
        openFile(name)
        writeTo(text)
        closeFile()

def main():
    global web
    
#    server = Server()
#    server.watch('index.html')
#    server.serve(port=8080, host='0.0.0.0')

    web = webScreen()
    web.serve()
    sleep(.2)
    
    web.openFile("index.html")
    web.writeTo("another test 5")
    web.closeFile()



#if __name__ == '__main__':
#    try:
#        main()
#    except KeyboardInterrupt:
#        print 'Interrupted'
#        try:
#            sys.exit(0)
#        except SystemExit:
#            os._exit(0)