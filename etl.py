import click
import pandas as pd
from datasets import Dataset, Features, ClassLabel, Value, DatasetDict

@click.command()
# upload yes or no
@click.option("--up_hf", default=False, help="Upload the dataset to the hub", required=False)
def main(up_hf=False):
    # load data
    data = pd.read_csv("data/data.csv")

    # get the list of unique codes
    list_codes = data["code"].unique().tolist()

    # get the number of unique codes
    n_codes = len(list_codes)

    print(f"Number of unique codes: {n_codes}")
    print(f"List of unique codes: {list_codes}")

    # add a new column for the chapter
    data["chapter"] = data["code"].str[0]

    data["origin"] = "diogocarapito"

    print(data.head())
    # define class labels
    class_labels = ClassLabel(
        num_classes=n_codes,
        names=list_codes,
    )

    # Define the features
    features = Features(
        {
            "code": Value("string"),
            "text": Value("string"),
            "chapter": Value("string"),
            "origin": Value("string"),
            "label": class_labels,
        }
    )


    # create a huggingface dataset
    dataset = Dataset.from_pandas(
        data[["code", "text", "chapter", "origin", "label"]], features=features
    )

    # slpit the dataset
    train_test_split = dataset.train_test_split(
        test_size=0.2, seed=42, stratify_by_column="label"
    )

    # create a dataset dict
    dataset_dict = DatasetDict(
        {"train": train_test_split["train"], "test": train_test_split["test"]}
    )

    # save the dataset
    dataset_dict.save_to_disk("data/dataset_huggingface")

    if up_hf:
        # upload the dataset to the hub
        dataset_dict.push_to_hub(
            repo_id="diogocarapito/text-to-icpc2-nano",
        )

if __name__ == "__main__":
    main()