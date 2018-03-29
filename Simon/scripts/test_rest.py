import requests
from json import JSONDecoder

address = "http://localhost:5001"
decoder = JSONDecoder()

test_file_name = '185_baseball' #185_baseball, 26_radon_seed, 30_personae, 196_autoMpg, 313_spectrometer etc

filename = "data/"+test_file_name+".csv"

print("DEBUG::chkpt0")
print(filename)

try:
	files = {'file':open(filename,'rb')}
	print("DEBUG::chkpt1")
	r = requests.post(address+"/fileUpload", files=files)
	print("DEBUG::chkpt2")
	result = decoder.decode(r.text)
	print("DEBUG::success!!")
	print("The output from the duke docker image is:")
	print(result)
except Exception as e:
	print(e)
	print("DEBUG::Failure! Sorry! Please check and try again...")

