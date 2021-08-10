
def rotation_data_augmentation(instance,instance_label, num_of_new_instances, max_rotation):
    instance = instance.reshape((instance.shape[0], 28, 28, 1))
    instance = instance.astype('float32')
    instances = []
    instances_labels = []
    for i in range(num_of_new_instances): 
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

def shift_data_augmentation(instance,instance_label, num_of_new_instances, shift=0.2):
    instance = instance.reshape((instance.shape[0], 28, 28, 1))
    instance = instance.astype('float32')
    instances = []
    instances_labels = []
    for i in range(num_of_new_instances): 
        datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)
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
    print("instances.dtype = ",instances.dtype)
    print("instances.shape = ",instances.shape)
    # input("press enter to continue")
    return instances, instances_labels
