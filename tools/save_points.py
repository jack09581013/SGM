import numpy as np
import tools

with open('data/taipei/feature points yml/1finalpoint2_64.yml', 'r') as file:
    text = file.read().replace('\n', '')
    index = text.find('[')
    exec('points = ' + text[index:])
    points = np.array(points).reshape(-1, 2)
    print(points)
    tools.save(points, 'data/taipei/feature points pickle/photoscan_v2_4764.points')