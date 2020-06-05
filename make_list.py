import os
import argparse



def expand_dataset(data_path, pictures):

    image_inds = [path for path in os.listdir(os.path.join(data_path, pictures))]
    txt = ''
    for image_ind in image_inds:
        txt = txt + image_ind[:-4] + '\n'
    xml_file = open('train.txt', 'w')
    xml_file.write(txt)
    xml_file.close()
    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/pc405/Documents/TensorFlow_Faster-RCNN_for_site/dataset")
    flags = parser.parse_args()

    num = expand_dataset(flags.data_path, 'pictures')
    print('=> The number of image for train is: %d' %num)