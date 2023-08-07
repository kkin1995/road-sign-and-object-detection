import argparse

parser = argparse.ArgumentParser(
    prog="CalculateShapeAfterConvMaxPool",
    description="Calculates the Output Shape afte Convolution or Max Pooling",
)

parser.add_argument("--input-width", type=int, action="store")
parser.add_argument("--input-height", type=int, action="store")
parser.add_argument("--input-channels", type=int, action="store")
parser.add_argument("--kernel-size", type=int, action="store")
parser.add_argument("--stride", type=int, action="store", default=1)
parser.add_argument("--padding", type=int, action="store", default=0)

args = parser.parse_args()

output_width = (
    (args.input_width - args.kernel_size + (2 * args.padding)) / (args.stride)
) + 1
output_height = (
    (args.input_height - args.kernel_size + (2 * args.padding)) / (args.stride)
) + 1

print(f"Output Dimension: {(output_height, output_width)}")
