# Function to send the images in a folder to the cloud API for recognition
# AUTHOR: Adrian Sanchez-Mompo

import argparse
import requests
import os
import cv2

def parse_args():
    parser = argparse.ArgumentParser(prog="Send Images From Folder",
    description="This program sends the images stored in a folder to the cloud face recognition system.")
    parser.add_argument('--input-folder', required=True,
                        help='Folder with the images to upload.')
    parser.add_argument('--ip-address', required=True,
                        help='IP address of the Cloud API server')
    parser.add_argument('--dry-run', required=False, action='store_true',
                        help='If included, the images are sent to the cloud but no results are obtained.')
    parser.add_argument('--stats', required=False, action='store_true',
                        help='Shows statistics of the images sent.')
    return parser.parse_args()
    
def main():
    args = parse_args()
    if(args.dry_run):
        url = 'http://'+str(args.ip_address)+'/face-recognition/dry-run'
    else:
        url = 'http://'+str(args.ip_address)+'/face-recognition/get-results'
    files = os.listdir(args.input_folder)
    if(args.stats):
        file_num = 0
        comb_height = 0
        comb_width = 0
        comb_pixels = 0
    for file in files:
        file_path = os.path.join(args.input_folder, file)
        if(os.path.isfile(file_path)):
            if(args.stats):
                file_num = file_num + 1
                img = cv2.imread(file_path)
                height = img.shape[0]
                width = img.shape[1]
                comb_height = comb_height + height
                comb_width = comb_width + width
                comb_pixels = comb_pixels + height*width
            myobj = {'image': open(file_path, 'rb')}
            request = requests.post(url, files = myobj, timeout=10)
            print('Face recognition request for ' + file + ' had response: {}'.format(request.status_code))
            if(not args.dry_run):
                print(request.text)
            if(request.status_code != 200):
                print('Error message: ' + request.text)
    if(args.stats):
        # Print statistics
        print()
        print('~~~Statistics~~~')
        print('%d images have been sent to the cloud server.' %(file_num,))
        print('These images have an avg. height of %d px, an avg. width of %d px, and an avg. size of %d px' %(comb_height/file_num, comb_width/file_num, comb_pixels/file_num))    
if __name__ == '__main__':
    main()
