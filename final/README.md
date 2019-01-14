額外使用套件：
torch==0.4.1
scikit-learn==0.20.2
tqdm==4.28.1
cv2==3.4.2
imgaug==0.2.7
PIL==5.2.0
torchvision==0.2.1
numpy==1.14.5
pandas==0.23.4

圖片放置路徑說明：
Train Data（蛋白質圖片）須放在total_train資料夾內
Test Data (蛋白質圖片) 須放在test資料夾內

Execution:
    bash test.sh [directory of sample_submission.csv]

Result:
    The result .csv file will appear in the submit folder