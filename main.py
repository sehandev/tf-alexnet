import os
import random
from typing import Dict

import hydra
import numpy as np
from omegaconf import DictConfig
import tensorflow as tf
import keras.api._v2.keras as keras
from keras import layers


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run(cfg: Dict) -> None:
    # 0.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg["gpus"]))
    set_seed(cfg["seed"])
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )

    # 1. Fashion MNIST 데이터셋 불러오기
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # 2-1. Train dataset
    train_images = train_images / 255.0
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.batch(cfg["train_batch_size"])
    train_dataset = train_dataset.with_options(options)

    # 2-2. Test dataset
    test_images = test_images / 255.0
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.batch(cfg["test_batch_size"])
    test_dataset = test_dataset.with_options(options)

    with strategy.scope():
        # 3. 모델 구성
        model = keras.models.Sequential(
            [
                layers.Reshape((28, 28, 1), input_shape=(28, 28)),
                layers.Resizing(224, 224),
                layers.Conv2D(96, (11, 11), strides=4, padding="same"),
                layers.ReLU(),
                layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same"),
                layers.Conv2D(256, (5, 5), padding="same"),
                layers.ReLU(),
                layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same"),
                layers.Conv2D(384, (3, 3), padding="same"),
                layers.ReLU(),
                layers.Conv2D(384, (3, 3), padding="same"),
                layers.ReLU(),
                layers.Conv2D(256, (3, 3), padding="same"),
                layers.ReLU(),
                layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same"),
                layers.Flatten(),
                layers.Dropout(cfg["dropout"]),
                layers.Dense(4096, activation="relu"),
                layers.Dropout(cfg["dropout"]),
                layers.Dense(4096, activation="relu"),
                layers.Dense(10, activation="softmax"),
            ]
        )

        # 4. 모델 컴파일
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # 5. 모델 훈련
        model.fit(train_dataset, validation_data=test_dataset, epochs=cfg["epoch"])

        # 6. 정확도 평가하기
        print("[ Test ]")
        loss, accuracy = model.evaluate(test_dataset)
        print(f"loss : {loss}, accuracy : {accuracy}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def keras_app(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    keras_app()
