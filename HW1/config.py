

from argparse import Namespace
from typing import Any


def dict_to_namespace(dictionary):
    for k in dictionary:
        if isinstance(dictionary[k], dict):
            dictionary[k] = dict_to_namespace(dictionary[k])
    return MyNamespace(**dictionary)

# def dict_to_namespace(dictionary):
#     for k in dictionary:
#         if isinstance(dictionary[k], dict):
#             dictionary[k] = dict_to_namespace(dictionary[k])
#     return UserDict(**dictionary)



class MyNamespace(dict):
    def __getattr__(self, key):
        if isinstance(key, str) and key in self:
            r = self[key]
        else:
            r = getattr( super(), key)
        return r
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__name, str) and __name in self:
            self[__name] = __value
        else:
            return super().__setattr__(__name, __value)

cloud_config=dict(
    task_type='regression', # classification or regression
    experts=dict(
        expert=dict(
            in_features=9,
            hidden_size=8,
        ),
        epoch=10,
        lr=1e-2,
    ),
    num_experts=6,
    data_path='data/cloud',
    output_dir='cloud_outputs',
    experts_advice=dict(
        type='fixshare', # fixshare or static
        alpha=0.2,
        lr=2e-2,
    ),
)

cloud_config = dict_to_namespace(cloud_config)

spambase_config=dict(
    task_type='classification', # classification or regression
    experts=dict(
        expert=dict(
            in_features=57,
            hidden_size=16,
        ),
        epoch=10,
        lr=1e-3,
    ),
    data_path='data/spambase',
    num_experts=6,
    output_dir='spambase_outputs',
    experts_advice=dict(
        type='static', # fixshare or static
        lr=2e-2,
    ),
)

spambase_config = dict_to_namespace(spambase_config)

eye_config=dict(
    task_type='cluster',
    num_classes=4,
    num_iters=1000,
    output_dir='cluster_outputs'
)

eye_config = dict_to_namespace(eye_config)


iris_config=dict(
    task_type='cluster',
    num_classes=4,
    num_iters=1000,
    output_dir='cluster_iris_outputs',
    num_noise=50
)

iris_config = dict_to_namespace(iris_config)


if __name__ == '__main__':
    print(cloud_config)
    print(cloud_config['experts'])
    print(cloud_config.experts)
    print(type(cloud_config.experts))
    temp = dict(**cloud_config)
