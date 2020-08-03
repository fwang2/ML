import os
import subprocess
import json


get_cnodes = "echo $(cat {} | sort | uniq | grep -v batch | grep -v login)".format(os.environ['LSB_DJOB_HOSTFILE'])
cnodes = subprocess.check_output(get_cnodes, shell=True)
cnodes = str(cnodes)[2:-3].split(' ')
nodes_list = [c + ":2222" for c in cnodes] # Add a port number


index = int(os.environ['PMIX_RANK']) 
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': nodes_list
    },
    'task': {'type': 'worker', 'index': index} 
})

