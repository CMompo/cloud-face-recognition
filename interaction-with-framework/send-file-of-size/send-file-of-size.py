import os
import argparse
import requests

def parse_args():
    parser = argparse.ArgumentParser(prog="Send File of Size",
    description="This program sends a random file of a given size to the cloud face recognition system.")
    parser.add_argument('--ip-address', required=True,
                        help='IP address of the Cloud API server')
    parser.add_argument('--size', required=True, type = int,
                        help='Size in MB of the file to be sent.')
    return parser.parse_args()
    
def main():
    args = parse_args()
    url = 'http://'+str(args.ip_address)+'/face-recognition/dry-run'
    size = args.size*1024*1024
    file_gen = os.urandom(size)
    myobj = {'image': file_gen}
    request = requests.post(url, files = myobj, timeout=100)
    if(request.status_code != 200):
        print('Error message: ' + request.text)
    else:
        print('The operation was successful')
if __name__ == '__main__':
    main()
