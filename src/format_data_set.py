import csv
import json
import argparse


def format_math_dataset(input_file, output_file):
    """
    Format a math dataset for evaluation with DeepEval.
    :param input_file: forma is a CSV file with two columns: question, answer
    :param output_file: format is a JSON file with the DeepEval format as : {"prompt": question, "target": answer}
    :return:
    """
    # Load the input Dataset
    with open(input_file, "r") as file:
        reader = csv.reader(file)
        input_dataset = list(reader)

    # Convert the input Dataset to the DeepEval format
    deepeval_dataset = []
    for question, answer in input_dataset:
        deepeval_dataset.append({"prompt": question, "target": answer})

    # Save the DeepEval dataset
    with open(output_file, "w") as file:
        json.dump(deepeval_dataset, file)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Format a math dataset for evaluation with DeepEval.")
    arg_parser.add_argument("input_file", help="The path to the input CSV file.")
    arg_parser.add_argument("output_file", help="The path to the output JSON file.")
    args = arg_parser.parse_args()
    format_math_dataset(args.input_file, args.output_file)
