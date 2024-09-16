from pathlib import Path
from argparse import ArgumentParser
from onnxruntime.quantization import quantize_dynamic


def main() -> None:
    parser = ArgumentParser('pyprune')
    parser.add_argument("-s", "--source", type=str, required=True)
    parser.add_argument("-d", "--dest", type=str, default=None)
    args = parser.parse_args()
    
    args.source = Path(args.source).resolve().absolute()

    if args.dest is None:
        fname = f"{args.source.stem}_quant{''.join(args.source.suffixes)}"
        args.dest = args.source.parent / fname


    quantize_dynamic(args.source, args.dest)


if __name__ == "__main__":
    raise SystemExit(main())
