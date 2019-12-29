import  os

def get_train_class_list():
    base_dir = 'D:/Dataset/DAVIS/ImageSets/2017'
    ann_dir = "D:/Dataset/DAVIS/Annotations/480p"
    train_dir = base_dir+ '/train.txt'
    class_id = os.listdir(ann_dir)
    val_dir = base_dir + '/val.txt'
    train_list =[]
    # print(class_id)
    for way in open(train_dir,mode='r'):
        way = way.strip('\n')
        id = class_id.index((way)) + 1
        print(id, end=",")
        train_list.append(id)
    val_list = []
    for way in open(val_dir):
        way = way.strip('\n')
        id = class_id.index(way) + 1

        val_list.append(id)
    return set(train_list),set(val_list)

if __name__ == '__main__':
    get_train_class_list()
