import albumentations as A
import cv2
from pathlib import Path


class DataAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.IAAPiecewiseAffine(p=0.3), # may be wrong
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(), # may be wrong
                A.IAAEmboss(), # may be wrong 
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
        ])

    def augment(self, image):
        return self.transform(image=image)['image']


def apply_augmentation(input_dir, output_dir, num_augmentations=2): # augmentation for every of photos
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    augmenter = DataAugmentation()

    for img_path in input_path.glob('*.png'):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for i in range(num_augmentations):
            augmented_img = augmenter.augment(img)
            output_file = output_path / f"{img_path.stem}_aug_{i}.png"
            cv2.imwrite(str(output_file), cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))

    print(f"Augmentation complete. {num_augmentations} augmented images created for each original image.")
