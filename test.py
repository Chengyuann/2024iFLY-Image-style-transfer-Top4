# test.py

import cv2
import glob
import numpy as np
import torch
import os
from dataset import FoodDataset
from model import get_model
import albumentations as A
import torch.utils.data as D
import zipfile

def predict_and_save(test_loader, model, test_img, output_dir='submit'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.eval()
    idx = 0
    with torch.no_grad():
        for image, _ in test_loader:
            image = image.to('cuda')
            outputs = model(image)
            outputs = (outputs.data.cpu().numpy() * 255).astype(np.uint8)
            for output in outputs:
                output = np.transpose(output, (1, 2, 0))
                output_path = os.path.join(output_dir, test_img[idx].split('/')[-1].replace('input', 'target'))
                cv2.imwrite(output_path, output)
                idx += 1

def zip_results(output_dir='submit', zip_name='submit-MITB5.zip'):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_dir))
    print(f"Generated zip file: {zip_name}")

if __name__ == "__main__":
    test_img = glob.glob('./test/*_input.jpg')
    test_img.sort()
    
    trfm = A.Compose([A.Resize(512, 512)])
    test_ds = FoodDataset(test_img, test_img, transform=trfm)
    test_loader = D.DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=3)
    
    model = get_model().to('cuda')
    predict_and_save(test_loader, model, test_img)
    
    # Generate zip file of the predictions
    zip_results()

    # Output the number of files in the submit and test directories
    submit_files = len(os.listdir('submit'))
    test_files = len(os.listdir('test'))
    print(f"Number of files in 'submit': {submit_files}")
    print(f"Number of files in 'test': {test_files}")
