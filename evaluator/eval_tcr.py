import argparse
import json



class McTacoEvaluator:

    def __init__(self, test_file, output_file):
        self.test_file = test_file
        self.output_file = output_file

    def print_result(self):
        ref_lines = [x.strip() for x in open(self.test_file).readlines()]
        prediction_lines = [x.strip() for x in open(self.output_file).readlines()]

        result_map = {}
        for i, line in enumerate(ref_lines):
            key = " ".join(line.split("\t")[0:2])
            if key not in result_map:
                result_map[key] = []
    
            prediction = prediction_lines[i]
            label = line.split("\t")[3]
            result_map[key].append(prediction == label)

        total = 0.0
        correct = 0.0
        for question in result_map:
            val = True
            total += 1.0
            
            for i, v in enumerate(result_map[question]):
                val = val and v
            if val:
                correct += 1.0
            
        print("Exact Match: " + str(correct / total))
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file",
                        required=True,
                        help="path to the csv file with gold labels.")
    parser.add_argument("--prediction_file",
                        required=True,
                        help="path to the line-by-line file containing system predictions.")


    args = parser.parse_args()

    evaluator = McTacoEvaluator(args.test_file, args.prediction_file)
    evaluator.print_result()


if __name__ == "__main__":
    main()