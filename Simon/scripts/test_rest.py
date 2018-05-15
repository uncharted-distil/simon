import requests
from json import JSONDecoder

address = "http://localhost:5000"
decoder = JSONDecoder()

test_file_name = 'o_185' #o_185, o_38

filename = "unit_test_data/"+test_file_name+".csv"

print("DEBUG::chkpt0")
print(filename)

try:
	files = {'file':open(filename,'rb')}
	print("DEBUG::chkpt1")
	r = requests.post(address+"/fileUpload", files=files)
	print("DEBUG::chkpt2")
	result = decoder.decode(r.text)
	print("DEBUG::success!!")
	print("The output from the simon docker image is:")
	print(result)
except Exception as e:
	print(e)
	print("DEBUG::Failure! Sorry! Please check and try again...")