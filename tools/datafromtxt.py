import os
import re
import ast

# --- CẤU HÌNH ---
INPUT_FOLDER = 'debug_logs'
HEX_OUTPUT_FOLDER = 'HEX_IN'
GOLDEN_OUTPUT_FOLDER = 'GoldenOFM'

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_values_array(values_text):
    clean_text = values_text.replace('[', ' ').replace(']', ' ').replace('\n', ' ')
    try:
        values = [int(x) for x in clean_text.split() if x.strip() not in ['', '...']]
        return values
    except ValueError:
        return []

def write_decimal_file(filepath, data_list):
    with open(filepath, 'w') as f:
        for item in data_list:
            f.write(str(item) + '\n')

def extract_section_data(content):
    scales = []
    zps = []
    values = []
    
    quant_match = re.search(r"quant\s*:\s*({.*?})", content, re.DOTALL)
    if quant_match:
        try:
            quant_dict = ast.literal_eval(quant_match.group(1))
            scales = quant_dict.get('scales', [])
            zps = quant_dict.get('zero_points', [])
        except:
            pass

    values_match = re.search(r"values\(int\):\s*(.*)", content, re.DOTALL)
    if values_match:
        values = parse_values_array(values_match.group(1))
        
    return scales, zps, values

def process_file(filepath):
    filename_with_ext = os.path.basename(filepath)
    file_prefix = os.path.splitext(filename_with_ext)[0]
    
    print(f"Processing: {filename_with_ext}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Tách file thành các section chính
    raw_parts = re.split(r'(===\s*.*?\s*===)', content)
    
    blocks = []
    start_idx = 0
    if raw_parts and not raw_parts[0].startswith("==="):
        start_idx = 1
        
    for i in range(start_idx, len(raw_parts) - 1, 2):
        header_part = raw_parts[i].strip()
        body_part = raw_parts[i+1]
        header_name = header_part.strip("= \n\t")
        blocks.append({"header": header_name, "content": body_part})

    final_sections = []
    
    # 2. Xử lý tách section ẩn (Deep Scan cải tiến)
    for block in blocks:
        text = block["content"]
        header = block["header"]
        
        # Tìm các vị trí bắt đầu dòng có "name :"
        # Dùng regex multiline để tìm ^name\s*: hoặc \nname\s*:
        matches = list(re.finditer(r'(^|\n)name\s*:', text))
        split_indices = [m.start() for m in matches]
        
        # Nếu không có hoặc chỉ có 1 "name:" nằm ngay đầu (hoặc gần đầu), giữ nguyên block
        if not split_indices:
             final_sections.append({"type": header, "content": text})
             continue
             
        # Nếu chỉ có 1 "name:" và nó nằm ở đầu text (bỏ qua whitespace), đó là section chuẩn
        if len(split_indices) == 1:
            # Kiểm tra xem trước đó có text gì quan trọng không
            pre_text = text[:split_indices[0]].strip()
            if not pre_text: # Nếu phía trước chỉ là khoảng trắng -> Chuẩn
                final_sections.append({"type": header, "content": text})
                continue

        # TRƯỜNG HỢP CÓ SECTION ẨN (VD: ADD layer bị gộp)
        # Nếu split_indices[0] > 0 (tức là có nội dung trước name: đầu tiên), ta giữ phần đó lại
        parts = []
        last_idx = 0
        
        # Nếu "name:" đầu tiên nằm ngay đầu, ta bắt đầu cắt từ đó.
        # Nếu "name:" đầu tiên nằm xa, phần trước đó thuộc về header cũ.
        
        # Để đơn giản: Ta coi mỗi "name:" là dấu hiệu bắt đầu 1 section con
        # Phần text từ đầu đến "name:" đầu tiên (nếu có) vẫn thuộc header chính
        if split_indices[0] > 0:
             # Có thể là phần values của header cũ trôi xuống, hoặc header cũ không có name
             # Nhưng thường header cũ sẽ ôm trọn đến name: tiếp theo
             pass

        # Cắt chuỗi
        for i, idx in enumerate(split_indices):
            # Điểm cuối của phần này là điểm đầu của phần sau (hoặc hết bài)
            end_idx = split_indices[i+1] if i + 1 < len(split_indices) else len(text)
            
            # Điều chỉnh index để lấy cả từ khóa "name :" (vì finditer trả về vị trí bắt đầu match)
            # match bao gồm cả \n nếu có, nên ta lấy từ match.start() + (1 nếu là \n)
            # Thực tế cứ cắt từ idx là an toàn vì parse_values sẽ lo phần rác
            
            # Tuy nhiên, cần xác định xem phần đầu tiên (từ 0 đến split_indices[0]) là gì?
            # Thường là rỗng. Nếu không rỗng, gán nó vào header hiện tại.
            if i == 0 and idx > 0:
                 final_sections.append({"type": header, "content": text[:idx]})

            part_content = text[idx:end_idx]
            
            # Xác định Type cho phần này
            if i == 0:
                # Phần chứa name: đầu tiên -> Chính là Header hiện tại
                sub_type = header
            else:
                # Các phần sau -> Section ẩn
                # Nếu là phần cuối cùng -> Khả năng cao là OFM
                if i == len(split_indices) - 1:
                    sub_type = "OFM"
                else:
                    sub_type = f"IMPLICIT_{i}"
            
            final_sections.append({"type": sub_type, "content": part_content})

    # 3. Đếm lại IFM sau khi đã bung các section ẩn
    ifm_count = sum(1 for s in final_sections if "INPUT" in s["type"] or "IFM" in s["type"])

    for sec in final_sections:
        name = sec["type"]
        content = sec["content"]
        
        scales, zps, values = extract_section_data(content)
        
        if not values and not scales:
            continue

        # Đặt tên file
        file_type = ""
        is_ofm = False
        
        # Logic nhận diện loại
        if "INPUT" in name or "IFM" in name:
            if ifm_count > 1:
                match = re.search(r'\d+$', name)
                idx = match.group() if match else "0"
                file_type = f"ifm{idx}"
            else:
                file_type = "ifm"
        elif "WEIGHT" in name:
            file_type = "weight"
        elif "BIAS" in name:
            file_type = "bias"
        elif "OUTPUT" in name or "OFM" in name:
            file_type = "ofm"
            is_ofm = True
        else:
            # Fallback
            file_type = name.lower().replace(" ", "_")

        # Ghi file
        if scales:
            path = os.path.join(HEX_OUTPUT_FOLDER, f"{file_prefix}_{file_type}_scale.hex")
            write_decimal_file(path, scales)

        if zps:
            path = os.path.join(HEX_OUTPUT_FOLDER, f"{file_prefix}_{file_type}_zp.hex")
            write_decimal_file(path, zps)

        if values:
            if is_ofm:
                path = os.path.join(GOLDEN_OUTPUT_FOLDER, f"{file_prefix}_{file_type}_values.hex")
            else:
                path = os.path.join(HEX_OUTPUT_FOLDER, f"{file_prefix}_{file_type}_values.hex")
            write_decimal_file(path, values)

def main():
    ensure_dir(HEX_OUTPUT_FOLDER)
    ensure_dir(GOLDEN_OUTPUT_FOLDER)
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Folder '{INPUT_FOLDER}' not found.")
        return

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.txt')]
    for f in files:
        try:
            process_file(os.path.join(INPUT_FOLDER, f))
        except Exception as e:
            print(f"Error processing {f}: {e}")
        
    print(f"\nDone! Processed {len(files)} files.")

if __name__ == "__main__":
    main()