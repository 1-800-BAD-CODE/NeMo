import argparse

from nemo.collections.nlp.models.token_classification.sentence_boundary_model import SentenceBoundaryDetectionModel


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exports a model", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model_path", help="Path to a .nemo model")
    parser.add_argument("output_path", help="Path to the output file. Extension .jit or .onnx to specify IR format.")

    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()
    model = SentenceBoundaryDetectionModel.restore_from(args.model_path)

    model.export(
        output=args.output_path,
        onnx_opset_version=17,
        input_example=(model.input_example(),),
        dynamic_axes={"input_ids": [0, 1], "probs": [0, 1]},
    )


if __name__ == "__main__":
    main()
