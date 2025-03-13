import cohere
# Initialize the client with your API key

co = cohere.Client()  # Replace with your actual API key

# List all fine-tuned models
finetuned_models = co.finetuning.list_finetuned_models()

# Print the models
if isinstance(finetuned_models, tuple) and len(finetuned_models) >= 2:
    # Get the list of models from the second element of the tuple
    models_list = finetuned_models[1]

    # Print each model's details
    for model in models_list:
        print(f"ID: {model.id}")
        print(f"Name: {model.name}")
        print(f"Status: {model.status}")
        print(f"Created at: {model.created_at}")
        print("-----------------------------------")
else:
    print("Unexpected response format:", finetuned_models)

