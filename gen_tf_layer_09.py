import numpy as np
import tensorflow as tf
import argparse
import math

# BIAS_FRAC_BIT = 7
# --- BẮT ĐẦU ĐOẠN BỔ SUNG CÁC HÀM XỬ LÝ SCALE ---

# Hàm đọc file chứa 1 số thực (Scale của IFM, OFM)
def read_float_file(filename):
    with open(filename, "r") as file:
        content = file.read().strip()
        # Lấy phần tử cuối cùng nếu file có dạng "0.69..."
        val_str = content.split()[-1]
    return float(val_str)

# Hàm đọc file chứa nhiều số thực (Scale của Weight)
def read_float_array_file(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    # Lọc và lấy số thực từ mỗi dòng
    data = []
    for line in lines:
        val_str = line.strip().split()[-1]
        data.append(float(val_str))
    return np.array(data, dtype=np.float64)
def QuantizeMultiplier_hs(double_multiplier):
    if double_multiplier == 0.:
        return 0, 0
    
    q, shift = math.frexp(double_multiplier)
    
    # SỬA Ở ĐÂY: Dùng floor(x + 0.5) thay vì round()
    q_fixed = int(math.floor(q * (1 << 31) + 0.5))
    
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
        
    if shift < -31:
        shift = 0
        q_fixed = 0
        
    return q_fixed, shift
# Mô phỏng chính xác hàm QuantizeMultiplier trong genMn.cpp
def QuantizeMultiplier(double_multiplier):
    if double_multiplier == 0.:
        return 0, 0
    
    # math.frexp trả về (mantissa, exponent) tương tự std::frexp
    q, shift = math.frexp(double_multiplier)
    
    # q_fixed = round(q * 2^31)
    q_fixed = int(round(q * (1 << 31)))
    
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
        
    if shift < -31:
        shift = 0
        q_fixed = 0
        
    return q_fixed, shift



# Tương ứng với phần _mm256_mul_epi32 và _mm256_add_epi64 trong C++
def SaturatingRoundingDoublingHighMul(a, b):
    # Check tràn số đặc biệt: -2^31 * -2^31 (rất hiếm gặp nhưng logic C++ có handle)
    if a == -2147483648 and b == -2147483648:
        return 2147483647
    
    # Thực hiện nhân 64-bit
    a_64 = int(a)
    b_64 = int(b)
    ab_64 = a_64 * b_64
    
    # Cộng offset (1 << 30) - Tương ứng 'offset_vector' trong C++
    nudge = 1 << 30
    
    # Dịch phải 31 bit
    # Python >> là arithmetic shift (giữ dấu), giống hệt logic xử lý int32 của C++
    result = (ab_64 + nudge) >> 31
    
    return result

# [MỚI] Mô phỏng hàm rounding_right_shift trong C++
# Xử lý chính xác logic "nudge" và Overflow Mask
def RoundingRightShift(x, shift):
    if shift <= 0:
        return x
        
    # Tính giá trị làm tròn (nudge)
    nudge = 1 << (shift - 1)
    
    # --- LOGIC QUAN TRỌNG NHẤT GIỐNG C++ ---
    # C++ Code:
    # mask_num_plus_nudge_overflow = _mm256_cmpgt_epi32(results, INT_MAX - nudge)
    # result = overflow ? (1 << (31 - shift)) : ((results + nudge) >> shift)
    
    int32_max = 2147483647
    
    # Kiểm tra overflow nếy cộng nudge
    if x > int32_max - nudge:
        # Nếu tràn, trả về giá trị đặc biệt như logic C++
        return 1 << (31 - shift)
    else:
        # Nếu không tràn, thực hiện cộng và dịch bình thường
        return (x + nudge) >> shift

# Hàm chính kết hợp 2 bước trên
def MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift):
    # Logic tính toán shift giống TFLite:
    # shift từ QuantizeMultiplier thường là số âm (ví dụ -7).
    # Total right shift = 31 - shift_exponent.
    # Tuy nhiên, hàm này nhận đầu vào 'shift' chính là exponent từ math.frexp
    
    # 1. Tính toán lượng dịch phải thực tế
    # shift ở đây là exponent (ví dụ -6).
    # Trong công thức: Real_Mul = Quantized_Mul * 2^(shift - 31)
    # => Ta cần HighMul (nhân 2^-31) và sau đó dịch phải thêm (-shift) nữa nếu shift < 0
    
    left_shift = shift if shift > 0 else 0
    right_shift = -shift if shift < 0 else 0
    
    # Bước 1: Left Shift (thường là 0 với Output Layer)
    x_shifted = x * (1 << left_shift)
    
    # Bước 2: Fixed Point Multiply (High Mul)
    # Tương ứng đoạn _mm256_mul_epi32... đến khi ra results trong C++
    high_mul = SaturatingRoundingDoublingHighMul(x_shifted, quantized_multiplier)
    
    # Bước 3: Rounding Right Shift
    # Tương ứng đoạn lambda rounding_right_shift trong C++
    result = RoundingRightShift(high_mul, right_shift)
    
    return result


# # Mô phỏng chính xác hàm MultiplyByQuantizedMultiplier trong genMn.cpp
# def MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift):
#     # x: int32, quantized_multiplier: int32, shift: int
    
#     total_shift = 31 - shift
    
#     # Tính round: 1 << (total_shift - 1)
#     round_val = 1 << (total_shift - 1)
    
#     # Thực hiện nhân 64-bit: result = x * multiplier + round
#     result = x * quantized_multiplier + round_val
    
#     # Dịch phải
#     result = result >> total_shift
    
#     # Trong Python số nguyên tự mở rộng, nhưng ta cần mô phỏng int32 nếu cần
#     # Tuy nhiên kết quả phép này thường được dùng để cộng ZP rồi clip
#     return result

# --- KẾT THÚC ĐOẠN BỔ SUNG ---

# --- [BỔ SUNG 1] Hàm đọc giá trị ZP từ file ---
def read_zp_file(filename):
    try:
        with open(filename, "r") as file:
            # Đọc toàn bộ nội dung, ví dụ: "F2"
            content = file.read().strip()
            # Lấy chuỗi cuối cùng (F2) để tránh các ký tự thừa
            val_str = content.split()[-1]
            
        val = int(val_str)
        
        # Xử lý số âm 8-bit (nếu > 127 thì trừ 256)
        # Ví dụ: F2 (242) -> -14
        if val > 0x7F:
            val -= 0x100
        return val
    except Exception as e:
        print(f"Cảnh báo: Không đọc được file {filename}, dùng ZP=0. Lỗi: {e}")
        return 0
    
# Hàm đọc dữ liệu từ file HEX với thứ tự hàng → cột → channel → filter
def read_bias_file(filename, length):
    with open(filename, "r") as file:
        # hex_values = file.readlines()
        lines = file.readlines()    
    # 1. Đọc dữ liệu Hex (Python tự hiểu 0xFF...F là số dương lớn)
    # Chúng ta dùng int(x, 16) sẽ ra số dương unsigned nếu x >= 0x80000000
    # data = np.array([int(x.strip(), 16) for x in hex_values], dtype=np.int64)
    data = np.array([int(x.strip()) for x in lines], dtype=np.int64)
    # 2. Xử lý số âm (Signed 32-bit conversion)
    # Nếu giá trị >= 2^31 (0x80000000), tức là số âm trong hệ bù 2 32-bit
    for i in range(len(data)):
        if data[i] >= 0x80000000: 
            data[i] -= 0x100000000  # Trừ đi 2^32 để về số âm

    # 3. Ép kiểu về đúng int32 để đưa vào Model
    data = data.astype(np.int32)

    # LƯU Ý QUAN TRỌNG: 
    # Đã XÓA đoạn: data[j] = data[j] * (1 << BIAS_FRAC_BIT)
    # Bias là số nguyên cộng trực tiếp vào accumulator, không cần shift.

    return data.reshape((length,))
# def read_bias_file(filename, length):
#     with open(filename, "r") as file:
#         hex_values = file.readlines()
    
#     # Chuyển đổi từ HEX thành số nguyên 8-bit có dấu
#     data = np.array([int(x.strip(), 16) for x in hex_values], dtype=np.int16)

#     # Đảm bảo dữ liệu trong phạm vi số nguyên có dấu 8-bit
#     for i in range(len(data)):
#         if data[i] > 0x7F:  # Nếu giá trị > 127, chuyển thành số âm
#             data[i] -= 0x100  # 0x100 là 256, nên ta trừ đi để có giá trị âm
            
#     for j in range(len(data)):
#         data[j] = data[j] * (1 << BIAS_FRAC_BIT)
        
#     return data.reshape((length,))
def read_hex_file_weight(filename, shape):
    with open(filename, "r") as file:
        # hex_values = file.readlines()
        lines = file.readlines()    
    # Chuyển đổi từ HEX thành số nguyên 8-bit có dấu
    # data = np.array([int(x.strip(), 16) for x in hex_values], dtype=np.int16)
    data = np.array([int(x.strip()) for x in lines], dtype=np.int16)
    # Đảm bảo dữ liệu trong phạm vi số nguyên có dấu 8-bit
    for i in range(len(data)):
        if data[i] > 0x7F:  # Nếu giá trị > 127, chuyển thành số âm
            data[i] -= 0x100  # 0x100 là 256, nên ta trừ đi để có giá trị âm

    H, W, C, F = shape
    reshaped_data = np.zeros((H, W, C, F), dtype=np.int16)
    index = 0
    for f in range(F):
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    reshaped_data[h, w, c, f] = data[index]
                    index += 1
    return reshaped_data

def read_hex_file(filename, shape):
    with open(filename, "r") as file:
        # hex_values = file.readlines()
        lines = file.readlines()    
    # Chuyển đổi từ HEX thành số nguyên 8-bit có dấu
    # data = np.array([int(x.strip(), 16) for x in hex_values], dtype=np.int32)
    data = np.array([int(x.strip()) for x in lines], dtype=np.int32)
    # Đảm bảo dữ liệu trong phạm vi số nguyên có dấu 8-bit
    # Nếu giá trị lớn hơn 127, chúng ta sẽ chuyển thành số âm
    for i in range(len(data)):
        if data[i] > 0x7F:  # Nếu giá trị > 127, chuyển thành số âm
            data[i] -= 0x100  # 0x100 là 256, nên ta trừ đi để có giá trị âm
    H, W, C = shape
    reshaped_data = np.zeros((H, W, C), dtype=np.int32)
    index = 0
    for h in range(H):
        for w in range(W):
            for c in range(C):
                reshaped_data[h, w, c] = data[index]

                index += 1

    return reshaped_data

# Hàm ghi dữ liệu ra file HEX
# Sửa trong gen_tf_layer_01.py
# Sửa trong gen_tf_layer_01.py
def write_hex_file(filename, data):
    H, W, C = data.shape
    with open(filename, "w") as file:
        for h in range(H):          # Loop Channel trước (để khớp với genhex.py)
            for w in range(W):
                for c in range(C):
                    int_value = int(round(data[h, w, c]))
                    
                    # SỬA LẠI: Mask 32-bit và format 8 ký tự
                    hex_value = int_value & 0xFFFFFFFF 
                    file.write(f"{int_value}\n")
                    # file.write(f"{hex_value:08X}\n")

# def write_hex_file(filename, data):
#     H, W, C = data.shape
#     with open(filename, "w") as file:
#         for c in range(C):
#             for h in range(H):
#                 for w in range(W):
#                     int_value = int(round(data[h, w, c]))
#                     # Sửa ở đây: Mask 0xFF và format 02X cho chuẩn 8-bit
#                     hex_value = int_value & 0xFF 
#                     file.write(f"{hex_value:02X}\n")
# def write_hex_file(filename, data):
#     H, W, C = data.shape
#     with open(filename, "w") as file:
#         for c in range(C):
#             for h in range(H):
#                 for w in range(W):
#                     int_value = int(round(data[h, w, c]))
#                     hex_value = int_value & 0xFFFF
#                     file.write(f"{hex_value:04X}\n")

# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifm_height", type=int, required=True)
    parser.add_argument("--ifm_width", type=int, required=True)
    parser.add_argument("--ifm_channel", type=int, required=True)
    parser.add_argument("--weight_filter", type=int, required=True)
    parser.add_argument("--padding1", type=int, default=1)  # Padding P
    parser.add_argument("--stride1", type=int, default=1)   # Stride S
    args = parser.parse_args()

    # Tính kích thước OFM với padding và stride
    output_feature_height = (args.ifm_height - 1 + 2 * args.padding1) // args.stride1 + 1
    output_feature_width = (args.ifm_width - 1 + 2 * args.padding1) // args.stride1 + 1
    output_feature_channel = args.weight_filter

    # File paths cố định
    #input_file = "hex/input_out.hex"
    input_file = "HEX_IN/op017_CONV_2D_ifm_values.hex"
    #weight_file = "hex/weight_hex_out.txt"
    weight_file = "HEX_IN/op017_CONV_2D_weight_values.hex"
    output_file = "OFM/op017_CONV_2D_ofm.hex"
    bias_file = "HEX_IN/op017_CONV_2D_bias_values.hex"

    
    # Đọc dữ liệu
    input_data = read_hex_file(input_file, (args.ifm_height, args.ifm_width, args.ifm_channel))
    weight_data_flat = read_hex_file_weight(weight_file, (1, 1, args.ifm_channel, args.weight_filter))
    weight_data = weight_data_flat.reshape(1, 1, args.ifm_channel, args.weight_filter)
    bias_data = read_bias_file(bias_file, args.weight_filter).astype(np.float32)
    zp_file = "HEX_IN/op017_CONV_2D_ifm_zp.hex" 
    zp_in = read_zp_file(zp_file)
    bias_data_1 = np.zeros(args.weight_filter, dtype=np.float32)
    print(f"Info: ZP input = {zp_in}")
    # 2. Tính toán và thực hiện Padding thủ công
    if args.padding1 > 0:
        # Tính toán lượng cần pad theo chuẩn TensorFlow 'SAME'
        pad_h_total = max((output_feature_height - 1) * args.stride1 + 1 - args.ifm_height, 0)
        pad_w_total = max((output_feature_width - 1) * args.stride1 + 1 - args.ifm_width, 0)
        
        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left
        
        print(f"Padding info: Top={pad_top}, Bottom={pad_bottom}, Left={pad_left}, Right={pad_right}")
        
        # Padding input data bằng giá trị ZP_IN
        input_data_padded = np.pad(
            input_data,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=zp_in
        )
    else:
        input_data_padded = input_data

    # Cập nhật shape mới sau khi pad
    padded_height, padded_width, _ = input_data_padded.shape

    # 3. Tạo mô hình với padding='valid' (vì đã pad tay rồi)
    input_layer = tf.keras.layers.Input(shape=(padded_height, padded_width, args.ifm_channel))
    conv_layer = tf.keras.layers.Conv2D(filters=args.weight_filter,
                                        kernel_size=(1, 1),
                                        strides=(args.stride1, args.stride1),
                                        padding='valid', # QUAN TRỌNG: Đổi thành VALID
                                        activation=None)(input_layer)
    
    model = tf.keras.Model(inputs=input_layer, outputs=conv_layer)
    model.layers[1].set_weights([weight_data.astype(np.float32), bias_data])

    # 4. Dự đoán với input đã pad
    output_data = model.predict(input_data_padded.reshape(1, padded_height, padded_width, args.ifm_channel).astype(np.float32))
    output_data = output_data.reshape(output_feature_height, output_feature_width, output_feature_channel)

    # 1. Đọc giá trị zp_in
    # Lưu ý: Đảm bảo file ifm_zp.hex nằm đúng thư mục hoặc sửa đường dẫn cho đúng
    
    # 2. Tính tổng trọng số (Sum Weight) cho từng Filter
    # weight_data có shape (H=3, W=3, Channel, Filter)
    # Ta cộng gộp 3 chiều đầu (0, 1, 2) để ra mảng (Filter,) chứa tổng weight của từng filter
    sum_w = np.sum(weight_data, axis=(0, 1, 2))
    
    # 3. Tính lượng bù (Correction) = zp_in * sum_w
    # Kết quả 'correction' là một mảng có số phần tử bằng số Filter (args.weight_filter)
    correction = zp_in * sum_w
    
    # 4. Trừ bù vào kết quả Output (Thực hiện cho TOÀN BỘ Layer)
    # Numpy Broadcasting sẽ tự động lấy từng pixel tại Filter i trừ đi correction[i]
    output_data = output_data - correction

    scale_ifm = read_float_file("HEX_IN/op017_CONV_2D_ifm_scale.hex")
    scale_ofm = read_float_file("HEX_IN/op017_CONV_2D_ofm_scale.hex")
    scale_weights = read_float_array_file("HEX_IN/op017_CONV_2D_weight_scale.hex")
    zp_ofm = read_zp_file("HEX_IN/op017_CONV_2D_ofm_zp.hex")

    print(f"Info: IFM Scale: {scale_ifm}")
    print(f"Info: OFM Scale: {scale_ofm}")
    print(f"Info: OFM ZP: {zp_ofm}")
    
    # 2. Chuẩn bị mảng chứa kết quả 8-bit
    H_out, W_out, C_out = output_data.shape
    output_final = np.zeros((H_out, W_out, C_out), dtype=np.int32)
    # 3. Thực hiện vòng lặp Requantize cho từng Filter (Channel)
    # Vì mỗi Filter có weight_scale khác nhau -> M và n khác nhau
    for f in range(C_out):
        # a. Tính Effective Scale: (S_in * S_w) / S_out
        effective_scale = (scale_ifm * scale_weights[f]) / scale_ofm
        
        # b. Tính M (quantized_multiplier) và n (shift) từ C++ logic
        multiplier_m, shift_n = QuantizeMultiplier(effective_scale)
        
        # In debug cho filter đầu tiên
        if f == 0:
            print(f"Debug Filter 0: Eff_Scale={effective_scale}, M={multiplier_m}, n={shift_n}")

        # c. Lấy dữ liệu của filter f (đã trừ bù ZP ở bước trước)
        acc_data = output_data[:, :, f]

        # d. Áp dụng MultiplyByQuantizedMultiplier (vectorized - loop giả lập)
        # Vì hàm MultiplyByQuantizedMultiplier viết cho scalar (số đơn), ta dùng vectorization của numpy
        # hoặc loop đơn giản để đảm bảo chính xác logic if/else overflow.
        
        # Để đảm bảo chính xác 100% logic C++ (đặc biệt là check overflow), 
        # ta nên loop qua từng phần tử hoặc dùng numpy apply_along_axis (nhưng loop for dễ debug hơn)
        
        acc_data = output_data[:, :, f].flatten() # Duỗi ra 1D để xử lý
        res_scaled_flat = np.zeros_like(acc_data)
        
        for i in range(len(acc_data)):
             res_scaled_flat[i] = MultiplyByQuantizedMultiplier(acc_data[i], multiplier_m, shift_n)
             
        res_scaled = res_scaled_flat.reshape(H_out, W_out)    


        # e. Cộng ZP Output
        res_final = res_scaled + zp_ofm
        
        # f. Clip về khoảng 8-bit Signed [-128, 127]
        res_final = np.clip(res_final, -128, 127)
        
        # g. Lưu vào mảng kết quả
        output_final[:, :, f] = res_final.astype(np.int32)

    # ======================================================
    # Ghi kết quả
    write_hex_file(output_file, output_final)
    # write_hex_file(output_file, output_data)
    print(f"Kết quả đã được ghi vào {output_file}")
    