# Function to Add Users
# AUTHOR: Adrian Sanchez-Mompo

import argparse
import requests
import os

def parse_args():
    parser = argparse.ArgumentParser(prog="Add User From Dataset",
    description="This program adds a person and its images to a cloud implementation of the Face Recognition Framework.")
    parser.add_argument('--input-folder', required=True,
                        help='Folder with the images to upload.')
    parser.add_argument('--person-id', required=True,
                        help='The identifier of the person to which the images pertain.')
    parser.add_argument('--ip-address', required=True,
                        help='IP address of the Cloud API server')
    parser.add_argument('--create-id', required=False, action='store_true',
                        help='Include to add the person ID to the system.')
    parser.add_argument('--train', required=False, action='store_true',
                        help='Send an optimized training request after the images are added.')
    return parser.parse_args()
    
def main():
    args = parse_args()
    if(args.create_id):
        url = 'http://'+str(args.ip_address)+'/dataset/'+str(args.person_id)
        request = requests.post(url, timeout=10)
        print('ID add request had response: {}'.format(request.status_code))
    url = 'http://'+str(args.ip_address)+'/dataset/'+str(args.person_id)+'/image'
    files = os.listdir(args.input_folder)
    for file in files:
        file_path = os.path.join(args.input_folder, file)
        if(os.path.isfile(file_path)):
            myobj = {'image': open(file_path, 'rb')}
            request = requests.post(url, files = myobj, timeout=10)
            print('Img add request for ' + file + ' had response: {}'.format(request.status_code))
            if(request.status_code != 200):
                print('Error message: ' + request.text)

    if(args.train):
        url = 'http://'+str(args.ip_address)+'/model/optimized-train'
        request = requests.post(url, timeout=1000)
        print('Optimized training request had response code: {}'.format(request.status_code))

if __name__ == '__main__':
    main()
