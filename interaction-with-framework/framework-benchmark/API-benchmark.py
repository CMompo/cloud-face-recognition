import grequests
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(prog='API-Benchmark',
    description="This program benchmarks the RAW performance of the Cloud API face recognition functions.")
    parser.add_argument('--ip-address', required=True,
                        help='IP address of the Cloud API server')
    parser.add_argument('--source-img', required=True,
                        help='Path of the image to use for the test.')
    parser.add_argument('--batch-size', type=int, default=1, required=False,
                        help='Number of concurrent requests to send.')
    parser.add_argument('--timeout', type=float, default=120, required=False,
                        help='Timeout of the requests. If batch-size is a large number, the timeout should also be increased.')

    return parser.parse_args()

def exception_handler(request, exception):
    print("Request failed.")

# Preparing requests
args = parse_args()
file_path = str(args.source_img)
url = 'http://'+str(args.ip_address)+'/face-recognition/get-results'
n_requests = args.batch_size
req_timeout = args.timeout
print('Preparing requests...')
prep_req = (grequests.post(url, files = {'image': open(file_path, 'rb')}, timeout=req_timeout) for _ in range(n_requests))
print('Requests prepared. Running benchmark...')
start_time = time.time()
result = grequests.map(prep_req, exception_handler)
process_time = time.time()-start_time

# Counting correct requests
correct = 0
for i in range(n_requests):
    if(result[i] != None and result[i].status_code==200):
        correct = correct + 1

print()
print('%d out of %d requests were successful' %(correct, n_requests))
print('Took %f seconds to complete %d requests at %f requests per second.' %(process_time, correct, correct/process_time))
print()
print('Example response:')
# Find a valid response and print it
for i in range (n_requests):
    if(result[i] != None and result[i].status_code==200):
        print(result[i].text)
        break