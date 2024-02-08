

from argparse import Namespace


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
        if isinstance(key, str):
            r = self[key] if key in self else None
        else:
            r = getattr( super(), key)

        return r

cloud_config=dict(
    task_type='classification', # classification or regression
    experts=dict(
        expert=dict(
            in_features=57,
            hidden_size=16,
        ),
        epoch=2,
        lr=2e-2,
    ),
    num_experts=6,
    output_dir='spambase_outputs',
    experts_advice=dict(
        type='fixshare', # fixshare or static
        alpha=0.1,
        lr=2e-2,
    ),
)

cloud_config = dict_to_namespace(cloud_config)

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
