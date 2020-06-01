import pandas as pd

sys.path.append('../../')
from src.preprocessing.get_data_info import generate_data_info

DATA_PATH = '../../../data/'

def main():
    """
    Extract information (body part, patient ID, anomaly on X-ray, anomaly on
    patient) about the MURA images into a pandas.DataFrame named 'data_info.csv'.
    """
    df1 = generate_data_info(DATA_PATH+'RAW/train_image_paths.csv')
    df2 = generate_data_info(DATA_PATH+'RAW/valid_image_paths.csv')
    df = pd.concat([df1,df2], axis=0).reset_index()
    df.to_csv(DATA_PATH+'data_info.csv')

if __name__ == '__main__':
    main()
