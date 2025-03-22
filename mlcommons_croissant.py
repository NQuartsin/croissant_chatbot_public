import mlcroissant as mlc
import inspect

# Get the signature of the Metadata class's constructor
signature = inspect.signature(mlc.Metadata)

# Print the parameters (properties)
print("Properties of mlc.Metadata:")
for name, param in signature.parameters.items():
    print(f"- {name}: {param.annotation}")

# Print the docstring of the Metadata class
print("\nDocumentation for mlc.Metadata:")
print(mlc.Metadata.__doc__)
# ds = mlc.Dataset("https://raw.githubusercontent.com/mlcommons/croissant/main/datasets/1.0/gpt-3/metadata.json")
# metadata = ds.metadata.to_json()
# print(f"{metadata['name']}: {metadata['description']}: {metadata['inLanguage']}")
# for x in ds.records(record_set="default"):
#     print(x)