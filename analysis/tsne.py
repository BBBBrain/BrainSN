
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap as mp
data_dir = '###############' 
file_list = sorted(os.listdir(data_dir)) 
# labels = {'MOTOR_lf':0, 'MOTOR_rf':1,'MOTOR_lh':2,'MOTOR_rh':3,'MOTOR_t':4,
#                  'WM_0bk_body':5,'WM_0bk_faces':6,'WM_0bk_places':7,'WM_0bk_tools':8,'WM_2bk_body':5,'WM_2bk_faces':6,'WM_2bk_places':7,'WM_2bk_tools':8,
#                   'EMOTION_fear':9,'EMOTION_neut':10,'GAMBLING_loss':11,'GAMBLING_win':12,'LANGUAGE_math':13,'LANGUAGE_story':14,
#                   'RELATIONAL_match':15 ,'RELATIONAL_relation':16,'SOCIAL_mental':17,'SOCIAL_rnd':18,'rfMRI':19
#                  }

# labels = {'MOTOR':0,
#                  'WM':1,
#                   'EMOTION':2,'GAMBLING':3,'LANGUAGE':4,
#                   'RELATIONAL':5 ,'SOCIAL':6,
#                  }


# labels = {'MOVIE1':0, 'MOVIE2':1,'MOVIE3':2,'MOVIE4':3,'rfMRI':4}
labels = {'ADHD':0, 'rfMRI':1}


labels_all = []

all_samples = []

for file_name in file_list:
    print(file_name)
    file_path = os.path.join(data_dir, file_name)


    sample_data = np.load(file_path)
    print(sample_data.shape)

    # for i in labels.keys():
    #     if i in file_name:
    #         for j in range(sample_data.shape[0]):
    #             labels_all.append(labels[i])
    #         break
    # # sample_data = np.load(file_path) 
    # # print(sample_data.shape)
    for i in labels.keys():
        if i in file_name:
            for j in range(sample_data.shape[0]):
                labels_all.append(labels[i])
            break
            
            # labels_all.append(labels[i])
            # break
    # sample_data = np.load(file_path) 
    # print(sample_data.shape)

    all_samples.append(sample_data)


combined_data = np.vstack(all_samples)[:3500] 


tsne = TSNE(n_components=2, random_state=2024,method="exact")
tsne_results = tsne.fit_transform(combined_data) 
print(tsne_results.shape,np.array(labels_all).shape)




np.save("############",tsne_results)
np.save("###############",labels_all)

