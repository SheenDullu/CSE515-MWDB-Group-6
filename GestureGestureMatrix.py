import pandas as pd

import Utilities


def dotProduct(directory):
    all_vectors = pd.DataFrame.from_dict(Utilities.getAllVectors(directory, 'tf'), orient='index')
    dot_product = all_vectors.dot(all_vectors.T)
    return dot_product


if __name__ == '__main__':
    dotProduct(r'D:\ASU\Courses\MWDB\Project\Phase 2\Code\data')
