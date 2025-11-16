import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt

def show(predict,shape,color_map,dataset_name,model_name):
    H, W = shape
    map = np.zeros((H * W))
    for i in range(len(predict)):
        map[i] = predict[i]
    map = np.reshape(map, (H,W))
    
    cmap = colors.ListedColormap(color_map, name=dataset_name)
    
    plt.imshow(map, cmap=cmap)
    plt.axis('off')
    plt.savefig(model_name+'_'+'predicted_'+dataset_name+'.png', dpi=800, bbox_inches='tight', pad_inches=0.0)
    
    print('Image saved.')
    