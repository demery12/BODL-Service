import argparse
import json


def model_output_to_json(model_output_file, out_filename):
    print("Converting", model_output_file, "to JSON")
    drinks = []
    with open(model_output_file, 'r') as f:
        contents = f.read()
        contents_split = contents.split('\\n')
        drink = {}
        for item in contents_split:
            if not item.startswith('\\t'):
                if len(drink.keys()) > 0:
                    drinks.append(drink)
                drink = {'drink_name': item, 'ingredients': []}
            else:
                item = item.lstrip('\\t-')
                drink['ingredients'].append(item)
    with open(out_filename, 'w') as f:
        f.write(json.dumps(drinks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert big string output of model to json')
    parser.add_argument('model_output_file', type=str, nargs=1,
                        help='Input file to this script, output file of the model')
    parser.add_argument('output_filename', type=str, nargs=1,
                        help='Name of file to output')

    args = parser.parse_args()

    model_output_file = args.model_output_file[0]
    if not model_output_file.endswith('.txt'):
        model_output_file = model_output_file+ '.txt'

    output_filename = args.output_filename[0]
    if not output_filename.endswith('.json'):
        output_filename = output_filename + '.json'

    model_output_to_json(model_output_file, output_filename)
