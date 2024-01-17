from Faster_RCNN.Management import ManagementCls
from Faster_RCNN.Train import TrainCls

Oper_phase = 'training'

# Dataset 위치 경로
training_dataset_folder = './Stanford_Dataset/cars_train/'
testing_dataset_folder = './Stanford_Dataset/cars_test/'

# 앵커박스 생성 스케일 지정
anchorbox_scales = [1, 2, 3]
anchorbox_ratios = [0.5, 1, 2]
anchorbox_base_size = 132  # 적절한 크기의 앵커 박스 생성하기 위해 기본값 설정

# 총 앵커 당 앵커박스 생성 수 ==> 9개
NumberOfanchorboxes_per_anchor = len(anchorbox_scales) * len(anchorbox_ratios)
# print(NumberOfanchorboxes_per_anchor)

manager = ManagementCls()

if Oper_phase == 'training':
    traingObj = TrainCls(NumberOfanchorboxes_per_anchor, anchorbox_scales,
                         anchorbox_ratios,
                         anchorbox_base_size)

    # train객체 와 train 이미지 데이터셋 전달하여 학습할 준비
    # manager.start_imgload_training(traingObj, training_dataset_folder)
