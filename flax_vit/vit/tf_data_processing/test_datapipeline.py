from input_pipeline import prepare_data, prefetch
import tensorflow_datasets as tfds
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'

def test_pipeline():
    a, _, td, info = prepare_data(
        dataset_name = "caltech101",
        validation_split = .1,
        batch_size = 2,
        image_size = 224,
        force_download = True
    )
    ds = prefetch(iter(tfds.as_numpy(td)), 2)
    for batch in ds:
        print(batch)
        break

    print(info.features['label'].num_classes)


if __name__ == "__main__":
    test_pipeline()