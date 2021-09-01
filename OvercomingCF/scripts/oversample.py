
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


def rotation_oversampling(instance,instance_label, num_of_new_instances, max_rotation):
    instance = np.array(instance)
    instance = instance.reshape((instance.shape[0], 28, 28, 1))
    instance = instance.astype('float32')

    instance_label = [instance_label]

    instances = []
    instances_labels = []
    for i in range(int(num_of_new_instances)): 
        datagen = ImageDataGenerator(rotation_range=max_rotation)
        # fit parameters from data
        datagen.fit(instance)
        for new_instance_augmented, new_instance_augmented_label in datagen.flow(instance, instance_label, batch_size=1):
            new_instance_augmented = new_instance_augmented.reshape((new_instance_augmented.shape[0], 28, 28))
            print("new_instance_augmented.dtype = ",new_instance_augmented.dtype)
            print("new_instance_augmented.shape = ",new_instance_augmented.shape)
            instances.append(new_instance_augmented)
            instances_labels = instances_labels + instance_label#instances_labels.append(new_instance_augmented_label)
            # plot
            # plt.imshow(new_instance_augmented.reshape(28, 28), cmap=plt.get_cmap('gray'))
            # plt.savefig('batch.png')
            # input("Press enter to continue")
            break # do not remove this break or you will get stuck in this for loop
    # print("instances.dtype = ",instances.dtype)
    # print("instances.shape = ",instances.shape)
    instances = np.array(instances)
    print("instances.dtype = ",instances.dtype)
    print("instances.shape = ",instances.shape)
    # input("press enter to continue")
    return instances, instances_labels

def shift_oversampling(instance,instance_label, num_of_new_instances, shift):
    instance = np.array(instance)
    instance = instance.reshape((instance.shape[0], 28, 28, 1))
    instance = instance.astype('float32')

    instance_label = [instance_label]

    instances = []
    instances_labels = []
    for i in range(int(num_of_new_instances)): 
        datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)
        # fit parameters from data
        datagen.fit(instance)
        for new_instance_augmented, new_instance_augmented_label in datagen.flow(instance, instance_label, batch_size=1):
            new_instance_augmented = new_instance_augmented.reshape((new_instance_augmented.shape[0], 28, 28))
            instances.append(new_instance_augmented)
            instances_labels = instances_labels + instance_label#.append(new_instance_augmented_label)
            print(len(instances))
            print(len(instances_labels))
            # plot
            # plt.imshow(new_instance_augmented.reshape(28, 28), cmap=plt.get_cmap('gray'))
            # plt.savefig('batch.png')
            # input("Press enter to continue")
            break # do not remove this break or you will get stuck in this for loop
    instances = np.array(instances)
    print("instances.dtype = ",instances.dtype)
    print("instances.shape = ",instances.shape)
    # input("press enter to continue")
    return instances, instances_labels

def shift_rotation_oversampling(instance,instance_label, num_of_new_instances, max_shift, max_rotation):
    instance = np.array(instance)
    instance = instance.reshape((instance.shape[0], 28, 28, 1))
    instance = instance.astype('float32')

    instance_label = [instance_label]

    instances = []
    instances_labels = []
    for i in range(int(num_of_new_instances)): 
        datagen = ImageDataGenerator(width_shift_range=max_shift, height_shift_range=max_shift, rotation_range=max_rotation)
        # fit parameters from data
        datagen.fit(instance)
        for new_instance_augmented, new_instance_augmented_label in datagen.flow(instance, instance_label, batch_size=1):
            new_instance_augmented = new_instance_augmented.reshape((new_instance_augmented.shape[0], 28, 28))
            instances.append(new_instance_augmented)
            instances_labels = instances_labels + instance_label#.append(new_instance_augmented_label)
            # print(len(instances))
            # print(len(instances_labels))
            # plot
            # plt.imshow(new_instance_augmented.reshape(28, 28), cmap=plt.get_cmap('gray'))
            # plt.savefig('batch.png')
            # input("Press enter to continue")
            break # do not remove this break or you will get stuck in this for loop
    instances = np.array(instances)
    instances_labels = np.array(instances_labels)
    # print("instances = ", instances)
    # print("instances_labels = ", instances_labels)
    # print("instances.dtype = ",instances.dtype)
    # print("instances.shape = ",instances.shape)
    # print("instances_labels.dtype = ",instances_labels.dtype)
    # print("instances_labels.shape = ",instances_labels.shape)
    # input("press enter to continue")
    # Check output is the same as the one expected by tempX and tempY
    # Stopeed here
    # input("please press")
    # plt.imshow(new_instance_augmented.reshape(28, 28), cmap=plt.get_cmap('gray'))
    # plt.savefig('batch.png')
    # input("Press enter to continue")
    return instances, instances_labels
