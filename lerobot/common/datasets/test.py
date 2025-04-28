import torch
from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset

def test_multitask_dataset():
    # Define the parameters for the test
    repo_ids = [
        "Mohamedal/put_peach_bowl_dis_test2",
        "Mohamedal/put_plum_bowl_dis_test2"
    ]
    language_embedding_model_id = "bert-base-uncased"

    # Define delta_timestamps for each dataset
    delta_timestamps = {
        "Mohamedal/put_peach_bowl_dis_test2": {"observation.state": [-0.0333, 0]},
        "Mohamedal/put_plum_bowl_dis_test2": {"observation.state": [-0.0333, 0]},
    }

    # Initialize the MultiLeRobotDataset
    dataset = MultiLeRobotDataset(
        repo_ids=repo_ids,
        root=None,  # Use default Hugging Face cache directory
        delta_timestamps=delta_timestamps,  # Pass corrected delta_timestamps here
        language_embedding_model_id=language_embedding_model_id,
        download_videos=True  # Enable downloading from Hugging Face
    )

    # Print dataset information
    print("Dataset initialized successfully!")
    print(dataset)

    # Test dataset length
    print(f"Total number of frames: {len(dataset)}")
    print(f"Total number of episodes: {dataset.num_episodes}")

    # Iterate through the last few samples
    print("\nIterating through the last few samples...")
    n = 10  # Number of samples to display
    for idx in range(max(0, len(dataset) - n), len(dataset)):  # Iterate over the last `n` samples
        sample = dataset[idx]
        dataset_index = sample['dataset_index'].item()  # Get the dataset index
        print(f"Sample {idx}:")
        print(f"  From Dataset: {repo_ids[dataset_index]}")
        print(f"  Task: {sample.get('task', 'N/A')}")
        print(f"  Task Embedding Shape: {sample.get('task_embedding', torch.tensor([])).shape}")
        print(f"  Dataset Index: {dataset_index}")

    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_multitask_dataset()