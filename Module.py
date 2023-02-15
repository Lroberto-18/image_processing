import matplotlib.pyplot as plt
import cv2


class Image_processing():
    
    def __init__(self, image, rows=3, columns=3, size=(10, 10)):

        self.image_bgr = cv2.imread(image)
        self.image_bgr_gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)
        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        self.image_fit = cv2.flip(self.image_bgr, 1)
        
        self.image_gray = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY)
        self.image_binary = cv2.threshold(self.image_gray, 
                                          128, 255, cv2.THRESH_BINARY)[1]

        self.image_bitwise_not = cv2.bitwise_not(self.image_bgr_gray)
        self.image_gausblur = cv2.GaussianBlur(self.image_bitwise_not, (85, 85), 0)
        self.image_draw = cv2.divide(self.image_bgr_gray,
                                     cv2.bitwise_not(self.image_gausblur), scale=256.0)


        self.height = self.image_bgr.shape[0]
        self.width = self.image_bgr.shape[1]
        self.channels = self.image_bgr.shape[2]
        self.rows = rows
        self.columns = columns
        self.id = 0
        self.figure = plt.figure(figsize=(size))


    def array_images(self):
        self.list_images = [self.image_rgb, self.image_bgr, self.image_fit,
                            self.image_gray, self.image_binary, self.image_binary, 
                            self.image_bitwise_not, self.image_gausblur, self.image_draw]

        return self.list_images

    def save_images(self):
        for i in range(len(self.list_images)):
            cv2.imwrite(f'image_0{i}.png', self.list_images[i])


    def plots(self,axis='off'):
        self.figure.add_subplot(self.rows, self.columns, 1)
        plt.imshow(self.image_rgb)
        plt.axis(axis) 

        self.figure.add_subplot(self.rows, self.columns, 2)
        plt.imshow(self.image_bgr) 
        plt.axis(axis) 

        self.figure.add_subplot(self.rows, self.columns, 3)
        plt.imshow(cv2.cvtColor(self.image_fit, cv2.COLOR_BGR2RGB))
        plt.axis(axis) 

        self.figure.add_subplot(self.rows, self.columns, 4)
        plt.imshow(self.image_gray, cmap='gray') 
        plt.axis(axis) 
            
        self.figure.add_subplot(self.rows, self.columns, 5)
        plt.imshow(self.image_binary, cmap='gray')
        plt.axis(axis) 

        self.figure.add_subplot(self.rows, self.columns, 6)
        plt.imshow(self.image_binary, cmap='binary')
        plt.axis(axis) 

        self.figure.add_subplot(self.rows, self.columns, 7)
        plt.imshow(cv2.cvtColor(self.image_bitwise_not, cv2.COLOR_BGR2RGB))
        plt.axis(axis) 

        self.figure.add_subplot(self.rows, self.columns, 8)
        plt.imshow(cv2.cvtColor(self.image_gausblur, cv2.COLOR_BGR2RGB))
        plt.axis(axis) 

        self.figure.add_subplot(self.rows, self.columns, 9)
        plt.imshow(cv2.cvtColor(self.image_draw, cv2.COLOR_BGR2RGB))
        plt.axis(axis) 


    def show_all(self, draw=True, save=False):
        if draw:
            self.plots()
            if save:
                plt.savefig('Image_processing.png', format='png')
            plt.show()