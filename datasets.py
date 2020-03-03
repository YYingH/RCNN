import cv2
class Data(object):
    
    def __init__(self, annotations, path = "data/FDDB/originalPics/"):
        self.annotations = annotations
        self.image_data = []
        self.path = path

    
    def __getitem__(self, index):
        image_dir = self.path + self.annotations[index][0] + '.jpg'
        num_of_faces = int(self.annotations[index][1])
        positions = self.annotations[index][2:]
        major_axis_radius, minor_axis_radius, angle, center_x, center_y = [], [], [], [], []
        for pos in positions:
            position = pos.split(' ')
            major_axis_radius.append(int(float(position[0])))
            minor_axis_radius.append(int(float(position[1])))
            angle.append(float(position[2]))
            center_x.append(int(float(position[3])))
            center_y.append(int(float(position[4])))
            image = cv2.imread(image_dir)
        return (image, image_dir, num_of_faces, [major_axis_radius, minor_axis_radius, angle, center_x, center_y])


    def __len__(self):
        return len(self.annotations)