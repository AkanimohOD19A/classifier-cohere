import cohere
import pandas as pd
import numpy as np

co = cohere.Client()

# convert Data
data_pth = "data/train.txt"
df = pd.read_csv(data_pth, sep=";", header=None, names=['text', 'label'])
df.to_csv("data/transformed.csv", index=False)


single_label_dataset = co.datasets.create(
    name="single-label-dataset",
    data=open("data/transformed.csv", "rb"),
    type="single-label-classification-finetune-input"
)

print(co.wait(single_label_dataset).dataset.validation_status)

from cohere.finetuning.finetuning import(
    BaseModel,
    FinetunedModel,
    Settings,
)

single_label_finetune = co.finetuning.create_finetuned_model(
    request=FinetunedModel(
        name="single-label-finetune",
        settings=Settings(
            base_model=BaseModel(
                base_type="BASE_TYPE_CLASSIFICATION",
            ),
            dataset_id = single_label_dataset.id
        )
    )
)

MODEL_ID = single_label_finetune.finetuned_model.id

print(
    f"fine-tune ID: {MODEL_ID}, "
    f"fine-tune status: {single_label_finetune.finetuned_model.status}"
)

## Call Model

response = co.classify(
    inputs=[
        "i didnt feel humiliated",
        "i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake"
    ], model = MODEL_ID + "-ft"
)


print(response)