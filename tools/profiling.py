import argparse
import os
import re
import sys


# --op ?(ví dụ 22 là --op 22)
def find_log_file(op_number, log_dir):
    """
    Find the full log file name for a given operation number.
    e.g., op_number=22 -> find file starting with "op022_".
    """
    if op_number < 0:
        return None
    prefix = f"op{op_number:03d}_"
    for filename in os.listdir(log_dir):
        if filename.startswith(prefix):
            return os.path.join(log_dir, filename)
    return None

def parse_shape_from_line(line):
    """
    Parse a shape string like '[1, 28, 28, 48]' and return a tuple.
    """
    # Regular expression to find shapes of the form [N, H, W, C]
    match = re.search(r'\[\d+,\s*(\d+),\s*(\d+),\s*(\d+)\]', line)
    if match:
        # Return (height, width, channels)
        return tuple(map(int, match.groups()))
    return None

def get_op_type_from_filename(filename):
    """
    Extract the operation type from the log file name.
    e.g., op022_CONV_2D.txt -> CONV_2D
    """
    if not filename:
        return "Unknown"
    match = re.search(r'op\d{3}_(.*?)\.txt', os.path.basename(filename))
    if match:
        return match.group(1)
    return "Unknown"

def get_weight_count(op_number, hex_dir, log_dir):
    """
    Count the number of weights in the corresponding .hex file.
    """
    log_file = find_log_file(op_number, log_dir)
    if not log_file:
        return 0
    op_type = get_op_type_from_filename(log_file)
    weight_file = os.path.join(hex_dir, f'op{op_number:03d}_{op_type}_weight_values.hex')
    if os.path.exists(weight_file):
        try:
            with open(weight_file, 'r') as f:
                return len(f.readlines())
        except Exception:
            return 0
    return 0

def parse_log_file(log_path):
    """
    Parse a log file to extract shape information and other parameters.
    """
    info = {
        'ifm_shape': None,
        'ofm_shape': None,
        'kernel_size': None,
        'stride': None,
        'padding': None
    }
    if not log_path or not os.path.exists(log_path):
        return info

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return info

    # Find IFM shape from 'IFM' or 'INPUT0' section
    for i, line in enumerate(lines):
        if '=== IFM ===' in line or '=== INPUT0 ===' in line:
            # Read the next 5 lines to find the 'shape' line
            for j in range(i + 1, min(i + 6, len(lines))):
                if 'shape' in lines[j]:
                    shape = parse_shape_from_line(lines[j])
                    if shape:
                        info['ifm_shape'] = shape
                        break
            break

    # Find OFM shape from 'OFM' section
    for i, line in enumerate(lines):
        if '=== OFM ===' in line:
            for j in range(i + 1, min(i + 6, len(lines))):
                if 'shape' in lines[j]:
                    shape = parse_shape_from_line(lines[j])
                    if shape:
                        info['ofm_shape'] = shape
                        break
            break
            
    # Find other Conv layer parameters
    for line in lines:
        # Find kernel size, e.g., "kernel_size: [3, 3]"
        if 'kernel' in line.lower() and 'size' in line.lower():
             match = re.search(r'\[(\d+),\s*(\d+)\]', line)
             if match:
                 info['kernel_size'] = tuple(map(int, match.groups()))
        # Find stride, e.g., "strides: (2, 2)"
        if 'stride' in line.lower():
            match = re.search(r'[\(\[](\d+),\s*(\d+)[\)\]]', line)
            if match:
                info['stride'] = tuple(map(int, match.groups()))
        # Find padding mode
        if 'padding' in line.lower():
            if 'SAME' in line.upper():
                info['padding'] = 'SAME'
            elif 'VALID' in line.upper():
                info['padding'] = 'VALID'

    return info

def main():
    parser = argparse.ArgumentParser(description="Get layer dimension info from debug logs.")
    parser.add_argument("--op", type=int, required=True, help="Operation number (e.g., 22 for op022).")
    args = parser.parse_args()

    op_number = args.op
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(os.path.join(script_dir, '..', 'debug_logs'))
    hex_dir = os.path.abspath(os.path.join(script_dir, '..', 'HEX_IN'))

    if not os.path.isdir(log_dir):
        print(f"Error: Log directory not found at '{log_dir}'")
        sys.exit(1)

    # --- Find log file paths ---
    current_op_log_path = find_log_file(op_number, log_dir)
    next_op_log_path = find_log_file(op_number + 1, log_dir)

    if not current_op_log_path:
        print(f"Error: Log file for op {op_number} not found.")
        sys.exit(1)

    # --- Parse log files ---
    current_op_info = parse_log_file(current_op_log_path)
    next_op_info = parse_log_file(next_op_log_path)

    # --- Consolidate info ---
    op_type = get_op_type_from_filename(current_op_log_path)
    
    # Get IFM shape from the current layer's log
    ifm_shape = current_op_info.get('ifm_shape')
    
    # Get OFM shape from the current layer's log, or from the next layer's IFM
    ofm_shape = current_op_info.get('ofm_shape')
    if not ofm_shape and next_op_info:
        ofm_shape = next_op_info.get('ifm_shape')

    # --- Print results ---
    print(f"--- Layer Info OP #{op_number} ({op_type}) ---")
    
    if ifm_shape:
        h, w, c = ifm_shape
        print(f"  - IFM Height:    {h}")
        print(f"  - IFM Width:     {w}")
        print(f"  - IFM Channels:  {c}")
    else:
        print("  - IFM Shape:     Not found")

    if ofm_shape:
        h, w, c = ofm_shape
        print(f"  - OFM Height:    {h}")
        print(f"  - OFM Width:     {w}")
        print(f"  - OFM Channels:  {c}")
    else:
        print("  - OFM Shape:     Not found")

    # Print info specific to CONV layers
    if "CONV" in op_type:
        if ofm_shape:
             print(f"  - Weight Filter: {ofm_shape[2]}")

        kernel = current_op_info.get('kernel_size')
        if not kernel and ifm_shape and ofm_shape:
            num_weights = get_weight_count(op_number, hex_dir, log_dir)
            if num_weights > 0:
                ifm_channels = ifm_shape[2]
                ofm_channels = ofm_shape[2]
                # Ensure no division by zero
                if ifm_channels > 0 and ofm_channels > 0:
                    # kernel_h * kernel_w = num_weights / (ifm_channels * ofm_channels)
                    # For depthwise, it's num_weights / ifm_channels
                    if "DEPTHWISE" in op_type:
                         kernel_size_sq = num_weights / ifm_channels
                    else: # Regular Conv
                         kernel_size_sq = num_weights / (ifm_channels * ofm_channels)

                    if kernel_size_sq > 0:
                        kernel_h = int(kernel_size_sq**0.5)
                        kernel_w = kernel_h # Assume square kernels
                        if kernel_h * kernel_w * ifm_channels * (1 if "DEPTHWISE" in op_type else ofm_channels) == num_weights:
                            kernel = (kernel_h, kernel_w)


        if kernel:
            print(f"  - Kernel Size:   {kernel[0]}x{kernel[1]}")
        else:
            print("  - Kernel Size:   Not found in log or inferred")

        stride = current_op_info.get('stride')
        if stride:
            print(f"  - Stride:        {stride[0]}")
        else:
            print("  - Stride:        Not found (default is 1)")
        
        # Manually calculate padding value
        padding = current_op_info.get('padding')
        if padding:
            print(f"  - Padding Mode:  {padding}")
            if padding == 'VALID':
                 print(f"  - Padding Value: 0")
            elif padding == 'SAME' and kernel:
                 pad_val = kernel[0] // 2
                 print(f"  - Padding Value: {pad_val}")
            else:
                 print(f"  - Padding Value: Cannot be determined")
        elif kernel: # Infer padding
            if kernel[0] == 1 and kernel[1] == 1:
                print(f"  - Padding Value: 0 (inferred for 1x1 kernel)")
            elif kernel[0] > 1:
                 # Assume 'SAME' padding for kernels > 1x1
                 pad_val = kernel[0] // 2
                 print(f"  - Padding Value: {pad_val} (inferred for {kernel[0]}x{kernel[1]} kernel with 'SAME' padding)")
        else:
            print(f"  - Padding Value: Cannot be determined")


if __name__ == "__main__":
    main()