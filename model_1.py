from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import os
import matplotlib as mlp
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten, Dropout, Activation
from skimage import transform
from skimage import io
import skimage.transform
from skimage.color import rgb2gray
import random 
from tensorflow import keras


#model = Sequential()

ROOT_PATH = "C:/Users/reid/OneDrive/Documents/Data_files" 
graph = tf.Graph()


def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.io.imread(f))
            labels.append(int(d))
    return images, labels

train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")
images, labels = load_data(train_data_directory)
test_images, test_labels = load_data(test_data_directory)


#print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))


transformed_images = [skimage.transform.resize(image, (32,32), mode= 'constant') for image in images]    
transformed_test_images = [skimage.transform.resize(image, (32,32), mode= 'constant') for image in test_images] 
label_names = ['poor_road_surface', 'speed_bump_0', 'sleepy_road','left_bend','right_bend','left_right_road_bend', 'right_left_road_bend', 'children_warning','cyclist_warning','cattle_Crossing','roadworkers_ahead','traffic_lights','railway_crossing_gate', 'caution', 'road_narrows_on_both_sides', 'left_narrowing_road', 'right_narrowing_road','priority_at_next_intersection_right','intersection_with_priority_to_the_right','give_away','give_away_to_traffic_from_opposite_side','stop_and_give_way_to_traffic','no_entry', 'cyclist_not_permitted','vehicles_heavier_than_indicated_limit_prohibited','no_entry_for_transport_vehicles','no_Sentry_for_vehicles_wider_than_indicated', 'no_entry_for_vehicles_higher_than_indicted', 'no_entry_in_both_directions', 'no_left_turn','no_right_turn','no_overtaking','maximum_speed_limit_indication','track_only_for_pedestrians_and_cyclist', 'straighy_ahead_only', 'left_only', 'turn_right_or_continue_straight', 'roundabout', 'mandatory_cycle-way','track_Only_for_cycles_and_pedstrians', 'no_parking', 'no_parking_or_standing', 'no_parking_allowed_between_1st-15th_days_of_the_month','no_parking_allowed_between_16th-31st_days_of_the_month','priority_over_traffic_from_opposite_side','parking', 'handicap_parking', 'parking_for_motocycles_cars_and_minibuses','parking_for_only_lorries', 'parking_for_only_buses', 'parking_mandatory_on_pavement_or_verge','begin_of_residential_area','end_of_residential_area', 'one_way_traffic', 'dead_end_ahead', 'road_workers_prohibited', 'pedestrian_crossing', 'cycle_and_moped_crossing', 'parking_indication', 'speed_bump_1', 'priority_roads_ends', 'priority_roads_ahead' ]


def display_label_images(images, label):
    limit = 24
    plt.figure(figsize=(15,15))
    i=1
    start =labels.index(label)
    end= start + labels.count(label)

    for image in images[start:end][:limit]:
        plt.subplot(3,8,i)
        plt.axis('off')
        i +=1
        plt.imshow(image)

display_label_images(images, 1)        

        

conjoin(labels, label_names)



print(len(test_images))
print(len(transformed_test_images)) 

def display_images_labels(images,labels, label_names):   

    unique_labels = set(labels) 

    plt.figure(figsize=(35,35))
    i = 1
    for label in unique_labels:          
        image = images[labels.index(label)]
        plt.subplot(8,10, i)
        plt.axis('off')
        plt.title("{0}" .format(label_names[label]))
        i += 1
        _ = plt.imshow(image)        
    plt.show()
display_images_labels(images, labels, label_names) 


for image in transformed_images[:5] :
    print("shape: {0}, min: {1}, max: {2}" .format(image.shape, image.min(), image.max())) 



#attempt model building wihtout keras, just pure tensorflow

labels_a = np.array(labels)

images_a = np.array(transformed_images)
images_a = images_a.astype('float32')

#test arrays baby
test_labels_a = np.array(test_labels)

test_images_a = np.array(transformed_test_images)
test_images_a = test_images_a.astype('float32') 

#print('length of labels:', len(labels_a), '\nimage shape:', images_a.shape)

#in_shape = transformed_images.shape[1:]

'''
model.add(Conv2D(32,(3,3), activation='relu', kernel_initializer='he_uniform'))

model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dense(120,activation='relu', kernel_initializer='he_uniform'))

model.add(Dropout(0.5))

model.add(Dense(128,activation='softmax'))'''

model = keras.Sequential([ keras.layers.Flatten(), keras.layers.Dense(120, activation='relu'),  keras.layers.Dense(120) ])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(images_a,labels_a,epochs=30, batch_size=32)

test_loss,test_acc = model.evaluate(test_images_a, test_labels_a, verbose=2 )

print('\n Test accuracy:', test_acc)


predictability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = predictability_model.predict(test_images_a)


print(np.argmax(predictions[2400]))
 


def plot_images(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color =  'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:3.0f}%  ({})". format(label_names[predicted_label], 100*np.max(predictions_array) , label_names[true_label]), color=color)        


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i] 
    plt.grid(False)
    plt.xticks(range(62))
    plt.yticks([])
    thisplot = plt.bar(range(62), predictions_array,  color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

model.save('perfected_trafficSign_model')                            



i = 2400
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_images(i,predictions[i], test_labels_a, test_images_a)
#plt.subplot(1,2,2)
#plot_value_array(i, predictions[i], test_labels_a)
plt.show()



