import numpy as np
import im_func_lib as funci

input_data = '/Users/mattarnold/Masters/Return1/Matt/PTC'#sys.argv[1]
files = funci.get_file_list(input_data) 

file_list = files

dark_frame = np.load('dark_frame.npy')

num_files = len(file_list)
data_table = np.zeros((num_files,2))

crop_bbox = np.load('crop_box.npy')

y_low, y_high, x_low, x_high = crop_bbox[0,0],crop_bbox[0,1],crop_bbox[1,0],crop_bbox[1,1]
for I in range(num_files):
    work_img = funci.load_img(files[I])
    for J in range(np.size(work_img,2)):
        work_img[:,:,J] = np.subtract(work_img[:,:,J],dark_frame)
    crop_image = work_img [y_low : y_high, x_low : x_high,:]
    
    
    data_table[I,0] = np.mean(crop_image)
    data_table[I,1] = np.std(crop_image)
print(data_table) #np.save('mean_std_data',data_table)
    
