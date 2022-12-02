import os
import pandas as pd
from natsort import natsorted


def file_to_path(file=None, base_dir=None):
    """
    :param file: str with name of file to access
    :param base_dir: str base direcyory
    :return: path to the file
    """
    return os.path.join(base_dir, file)


def get_data(data_list, base_dir):
    """
    :param data_list: list of files in a directory
    :param base_dir: base directory
    :return: a data frame 
    """
    cols = ['Exp', 'Best epoch', 'AP bbox',
            'AP bbox @0.50', 'AP bbox @0.75', 'AP seg', 'AP seg @ 0.50',
            'AP seg @0.75']
    df = pd.DataFrame(columns=cols)
    print(df.columns)

    for file in data_list:
        file_path = file_to_path(file, base_dir)
        df_temp = pd.read_csv(file_path)
        print(df_temp.columns)
        df_temp = df_temp[cols]
        print(df_temp.head())

        df = pd.concat([df, df_temp])

    return df


if __name__ == '__main__':
    metrics_dir = os.path.join(os.getcwd(), 'predictions')

    metrics_list = natsorted(os.listdir(metrics_dir))

    df_results = get_data(metrics_list, metrics_dir)
    df_results = df_results.set_index('Exp')
    ranks = df_results[['AP bbox', 'AP bbox @0.50', 'AP bbox @0.75',
                        'AP seg', 'AP seg @ 0.50', 'AP seg @0.75']].rank(ascending=False)
    ranks.to_csv(os.path.join(metrics_dir,'rankings.csv'))
