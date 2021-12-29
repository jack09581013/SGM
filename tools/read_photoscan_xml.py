import numpy as np
import tools
import xml.etree.ElementTree as ET

root = ET.parse('data/taipei/parameters/camera.xml').getroot()
data = {}

for tag in root.findall('.//camera'):
    label = tag.get('label')[7:7+4]
    transform = tag.find('transform').text
    transform = [float(x) for x in transform.split()]
    transform = np.array(transform).reshape(4,4)
    R = transform[0:3, 0:3]
    T = transform[0:3, 3].reshape(-1, 1)
    data[label] = {
        'R': R,
        'T': T
    }

tools.save(data, 'data/taipei/parameters/camera.pkl')