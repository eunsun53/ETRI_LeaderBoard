import os
from re import A
from PIL import Image
import matplotlib.pyplot as plt
ia.seed(1)
import torch
from torchvision import transforms
from skimage import io, transform, color

class Bbox(torch.utils.data.Dataset):
    def __init__(self, df, base_path):
        self.df = df
        self.base_path = base_path
        
        self.to_tensor = transforms.ToTensor()
        
        self.to_pil = transforms.ToPILImage()

    def __getitem__(self, i):
        sample = self.df.iloc[i]
        img_nm = sample['image_name']
        image = io.imread(self.base_path + sample['image_name']) #ndarray 
        if image.shape[2] != 3:
            image = color.rgba2rgb(image)
            image = 255 * image  # scale by 255
            image = image.astype(np.uint8)
        
        bbox_xmin = sample['BBox_xmin']
        bbox_ymin = sample['BBox_ymin']
        bbox_xmax = sample['BBox_xmax']
        bbox_ymax = sample['BBox_ymax']


        return (bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax), image, img_nm


df = pd.read_csv('/content/gdrive/MyDrive/FASCODE2022/task1/info_etri20_emotion_train.csv')


#def data_augmentation(category_name, category_num , ver_num):
category_name = 'Daily'
category_num = 0
ver_num = 1

split_df = df[df[category_name] == category_num]
mult_num = int(1500/ len(split_df))
#mult_num = 4

bbox_dataset = Bbox(split_df, base_path='/content/gdrive/MyDrive/data/train/') # (x1, y1, x2, y2), image
xmin = np.array([])
ymin = np.array([])
xmax = np.array([])
ymax = np.array([])

seq = iaa.Sequential([    
    iaa.OneOf([
                  iaa.Affine(rotate = 30),
                  iaa.Affine(rotate = -30),
                  iaa.Affine(rotate = 60),
                  iaa.Affine(rotate = -60),
                  iaa.Affine(rotate = 0) 
    ]),
    iaa.SomeOf(1, [
                  iaa.Fliplr(),
                  iaa.OneOf([
                                iaa.Affine(scale={"x": (1.5), "y": (1)}),
                                iaa.Affine(scale={"x": (1), "y": (1.5)})
                  ]),  
    ],  random_order=True),
    iaa.SomeOf(2, [   
                  iaa.OneOf([
                                iaa.AdditiveGaussianNoise(scale=0.1 * 255),
                                iaa.AdditiveGaussianNoise(scale=0.2*255)
                  ]),
                  iaa.GammaContrast((1.5, 2.0)),
                  iaa.Multiply((0.6, 1.2))
    ], random_order=True)
])
seq_det = seq.to_deterministic()

aug_df = pd.DataFrame(columns=list(df.columns)) # 새로 만드는 df 
for idx, sample in enumerate(bbox_dataset):
    tmp_df = split_df.iloc[[idx]] # 원본데이터 정보 dataframe
    aug_df = aug_df.append(tmp_df, ignore_index = True) #새로운 csv에도 원본 데이터 추가 & 이미지 저장 

    (x1,y1, x2, y2) = sample[0]
    img = sample[1]
    print('before', img)
    img = img.astype(np.uint8)
    img_nm = sample[2]
    print('after', img_nm, img)

    plt.imshow(img)
    ########### 원본 이미지도 저장 ##############
    for_save = Image.fromarray(img)  
    name = str(split_df.iloc[idx]['image_name']).split('/')[0] # 이미지 파일 이름에서 폴더명 분리
    dir_path = '/content/gdrive/MyDrive/FASCODE2022/task1/data_balancing_try/' # 저장 경로 지정
    path = dir_path + name
    if not os.path.isdir(path): 
      os.mkdir(path)
    
    for_save.save(dir_path+ img_nm) 
    ###############################################

    ia_bounding_boxes = []
    ia_bounding_boxes.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))

    bbs = ia.BoundingBoxesOnImage(ia_bounding_boxes, shape=img.shape)

    image_aug_lst = seq_det.augment_images([img for _ in range(mult_num)]) ## mult_num배 만큼 각 이미지 augmentation 
    bbs_aug_lst = seq_det.augment_bounding_boxes([bbs for _ in range(mult_num)])

    for k in range(mult_num):
        image_aug = image_aug_lst[k]
        bbs_aug = bbs_aug_lst[k]
        bbs_aug = bbs_aug.remove_out_of_image().cut_out_of_image()

        new_bbs = list(bbs_aug[0])
        print(k,"th new bbs: ", new_bbs)
        new_x1, new_y1 = int(new_bbs[0][0]), int(new_bbs[0][1])
        new_x2, new_y2 = int(new_bbs[1][0]), int(new_bbs[1][1])
        

        xmin = np.concatenate([xmin, [new_x1]], axis=0)
        ymin = np.concatenate([ymin, [new_y1]], axis=0)
        xmax = np.concatenate([xmax, [new_x2]], axis=0)
        ymax = np.concatenate([ymax, [new_y2]], axis=0)

        for i in range(len(bbs.bounding_boxes)):
            before = bbs.bounding_boxes[i]
            after = bbs_aug.bounding_boxes[i]
          
        #image_before = bbs.draw_on_image(img, thickness=20)
        #image_after = bbs_aug.draw_on_image(image_aug, thickness=20, color=[0, 0, 255])
        
        # 바뀐 바운딩 박스 좌표 & 이미지 이름 변경 새로운 csv 만들기 
        img_name = str(split_df.iloc[idx]['image_name']).rstrip('.jpg') +'_' +str(k) + f'_{category_name}{category_num}_aug_{ver_num}.jpg' # 새 파일명
        tmp_df['image_name'] = img_name
        tmp_df['BBox_xmin'] = new_x1
        tmp_df['BBox_ymin'] = new_y1
        tmp_df['BBox_xmax'] = new_x2
        tmp_df['BBox_ymax'] = new_y2			
        print(tmp_df)
        aug_df = aug_df.append(tmp_df, ignore_index = True)

        # 이미지 저장 
        for_save = Image.fromarray(image_aug) # np.array type -> image type
        name = str(split_df.iloc[idx]['image_name']).split('/')[0] # 이미지 파일 이름에서 폴더명 분리
        dir_path = '/content/gdrive/MyDrive/FASCODE2022/task1/data_balancing_try/' # 저장 경로 지정

        path = dir_path + name
        if not os.path.isdir(path): # 경로 안에 폴더가 없다면 만들어줌
          os.mkdir(path)

        for_save.save(dir_path + img_name) 
        print("----------------------------")
        print("augmentation info & save img")
        print("img shape:", img.shape)

################################################

# csv 저장
aug_df.to_csv(f'/content/gdrive/MyDrive/FASCODE2022/task1/data_balancing_try/{category_name}_{category_num}_aug_{ver_num}.csv', index=False)