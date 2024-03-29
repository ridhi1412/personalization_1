from lightfm.cross_validation import random_train_test_split
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

import matplotlib.pyplot as plt


def lightfm_model(data, prec_at_k=10, train_split=0.8):
    """
        Code to evaluate LightFm model
        Data is a scipy sparse matrix
        
        https://arxiv.org/abs/1507.08439
    """
    model = LightFM(learning_rate=0.05, loss='bpr')
    train, test = random_train_test_split(data,
                                          test_percentage=1 - train_split)

    model.fit(train, epochs=10)

    train_precision = precision_at_k(model, train, k=10)
    test_precision = precision_at_k(model,
                                    test,
                                    k=10,
                                    train_interactions=train)

    train_auc = auc_score(model, train)
    test_auc = auc_score(model, test, train_interactions=train)

    print('Performance of LightFm Model \n')
    print(
        f'Precision \t Train: {train_precision.mean():.2f} \t Test: {test_precision.mean():.2f}'
    )
    print(
        f'AUC \t\t Train: {train_auc.mean():.2f} \t Test: {test_auc.mean():.2f}'
    )

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    ax[0, 0].hist(train_auc, bins='auto')
    ax[0, 0].title.set_text('Distribution of Train AUC score over users')
    ax[0, 0].set_ylabel('Count')
    ax[0, 0].set_xlabel('AUC Score')

    ax[0, 1].hist(test_auc, bins='auto')
    ax[0, 1].title.set_text('Distribution of Test AUC score over users')
    ax[0, 1].set_ylabel('Count')
    ax[0, 1].set_xlabel('AUC Score')

    ax[1, 0].hist(train_precision, bins='auto')
    ax[1, 0].title.set_text(
        f'Distribution of Train Precision @ {prec_at_k} for all users')
    ax[1, 0].set_ylabel('Count')
    ax[1, 0].set_xlabel(f'Precision @ {prec_at_k}')

    ax[1, 1].hist(test_precision, bins='auto')
    ax[1, 1].title.set_text(
        f'Distribution of Test Precision @ {prec_at_k} for all users')
    ax[1, 1].set_ylabel('Count')
    ax[1, 1].set_xlabel(f'Precision @ {prec_at_k}')

    plt.show()

    print('\n')
