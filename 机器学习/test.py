def extract_oneu_from_image(image_path, debug=True):
    if not os.path.exists(image_path):
        return []

    try:
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            return []
        pre_img = preprocess_image(raw_img)
        result = ocr.ocr(pre_img, cls=True)
    except Exception as e:
        if debug:
            print(f"   ❌ OCR失败: {e}")
        return []

    if not result:
        return []

    img = raw_img
    img_h, img_w = img.shape[:2]

    all_texts = []
    for line in result:
        if line is None:
            continue
        for box in line:
            if len(box) < 2:
                continue
            text = box[1][0]
            conf = box[1][1]
            bbox = box[0]
            cx = (bbox[0][0] + bbox[2][0]) / 2
            cy = (bbox[0][1] + bbox[2][1]) / 2
            left = bbox[0][0]
            right = bbox[2][0]
            all_texts.append((text, conf, cx, cy, left, right, bbox))

    if debug:
        print(f"   📐 {img_w}x{img_h}")

    # ===== 新增：先尝试直接匹配完整集装箱码 =====
    full_containers = []
    for txt, conf, cx, cy, left, right, bbox in all_texts:
        clean = re.sub(r'[\s\-_\.]', '', txt).upper()
        # 匹配完整集装箱码（4字母+7数字）
        if re.match(r'^[A-Z]{4}\d{7}$', clean):
            if is_valid_container_code(clean):
                full_containers.append((clean, conf, cx, cy, bbox, txt))
                if debug:
                    print(f"   ✅ 直接匹配完整箱号: {clean}")

    # 如果有完整的有效集装箱码，直接返回
    if full_containers:
        # 按置信度排序，取最可信的
        full_containers.sort(key=lambda x: x[1], reverse=True)
        box_code, conf, cx, cy, bbox, original_text = full_containers[0]

        # 提取箱型
        type_code = None
        y_diff_min = max(12, img_h * 0.012)
        y_diff_max = max(100, img_h * 0.18)

        for txt, _, tc_cx, tc_cy, _, _, _ in all_texts:
            if abs(cy - tc_cy) < y_diff_max and tc_cx > cx:
                clean_txt = re.sub(r'[\s\-_\.]', '', txt).upper()
                m = re.search(r'(2[0-9][A-Z0-9][0-9]|4[0-9][A-Z0-9][0-9])', clean_txt)
                if m:
                    tc = m.group(1).replace('6', 'G')
                    if tc not in PREFIX_BLACKLIST:
                        type_code = tc
                        if debug:
                            print(f"   🔧 箱型: {tc}")
                        break

        return [{
            'code': box_code,
            'type_code': type_code,
            'confidence': conf,
            'original_text': original_text,
            'bbox': bbox,
            'y': cy,
            'check_valid': is_valid_container_code(box_code),
            'right_x': None
        }]

    # ===== 原有逻辑：寻找前缀 =====
    prefix = None
    prefix_y = None
    prefix_right = None
    prefix_bbox = None
    prefix_conf = None
    prefix_text = None

    # 先尝试找单独的前缀（4位字母）
    for txt, conf, cx, cy, left, right, bbox in all_texts:
        clean = re.sub(r'[\s\-_\.]', '', txt).upper()
        if re.match(r'^[A-Z]{4}$', clean) and clean not in PREFIX_BLACKLIST:
            if cy < img_h * 0.6:
                prefix = clean
                prefix_y = cy
                prefix_right = right
                prefix_bbox = bbox
                prefix_conf = conf
                prefix_text = txt
                if debug:
                    print(f"   🔍 前缀: {prefix}")
                break

    if not prefix:
        if debug:
            print(f"   ❌ 无前缀")
        return []

    # 从OCR结果中提取数字
    y_threshold = max(50, img_h * 0.08)
    digits = []

    for txt, conf, cx, cy, left, right, bbox in all_texts:
        # 处理可能的前缀+数字组合（如 CBHU1234567）
        clean = re.sub(r'[\s\-_\.]', '', txt).upper()

        # 检查是否以当前前缀开头
        if clean.startswith(prefix):
            # 提取后面的数字部分
            suffix = clean[len(prefix):]
            if re.match(r'^\d+$', suffix):
                # 检查位置是否合理（通常在同一个水平线）
                if abs(cy - prefix_y) < y_threshold:
                    # 如果后缀包含7位数字，直接使用
                    if len(suffix) >= 7:
                        digits = [(suffix[:7], conf, left, right, cx)]
                        if debug:
                            print(f"   🔢 从前缀组合中提取数字: {suffix[:7]}")
                        break
                    else:
                        # 如果不足7位，添加到数字列表中
                        digits.append((suffix, conf, left, right, cx))
                        if debug:
                            print(f"   🔢 从前缀组合中提取数字: {suffix}")
                        continue

        # 原有的数字匹配逻辑
        if abs(cy - prefix_y) < y_threshold and left > prefix_right:
            clean_num = re.sub(r'[\s\-_\.]', '', txt)
            if re.match(r'^\d+$', clean_num):
                digits.append((clean_num, conf, left, right, cx))
                if debug:
                    print(f"   🔢 数字: '{clean_num}' (x={left:.0f})")

    # 如果直接从组合中提取了完整的7位数字
    if len(digits) == 1 and len(digits[0][0]) >= 7:
        final_digits = digits[0][0][:7]
    else:
        digits.sort(key=lambda x: x[2])
        all_digits = ''.join([d[0] for d in digits])

        if debug:
            print(f"   🔗 拼接数字: {all_digits}")

        # ===== 核心修复：如果只有6位，从原图裁剪右侧区域识别 =====
        if len(all_digits) == 6:
            if debug:
                print(f"   🔍 只有6位，尝试从原图裁剪右侧区域识别...")

            missing = extract_missing_digit_from_image(img, prefix_y, prefix_right, img_h, img_w, debug=debug)

            if missing:
                first_digit = missing[0]
                final_digits = all_digits + first_digit
                if debug:
                    print(f"   🔧 补全数字: {final_digits}")
            else:
                if debug:
                    print(f"   ❌ 裁剪区域未识别到数字")
                return []
        elif len(all_digits) >= 7:
            final_digits = all_digits[:7]
        else:
            if debug:
                print(f"   ❌ 数字不足6位")
            return []

    box_code = prefix + final_digits

    if is_valid_container_code(box_code):
        if debug:
            print(f"   ✅ {box_code}")
    else:
        # 修正校验位
        first10 = prefix + final_digits[:6]
        correct_check = compute_container_check_digit(first10)
        if correct_check:
            box_code = first10 + correct_check
            if debug:
                print(f"   🔧 修正: {box_code}")

    # 找箱型
    type_code = None
    y_diff_min = max(12, img_h * 0.012)
    y_diff_max = max(100, img_h * 0.18)

    for txt, conf, cx, cy, left, right, bbox in all_texts:
        if prefix_y + y_diff_min < cy < prefix_y + y_diff_max:
            clean = re.sub(r'[\s\-_\.]', '', txt).upper()
            m = re.search(r'(2[0-9][A-Z0-9][0-9]|4[0-9][A-Z0-9][0-9])', clean)
            if m:
                tc = m.group(1).replace('6', 'G')
                if tc not in PREFIX_BLACKLIST:
                    type_code = tc
                    if debug:
                        print(f"   🔧 箱型: {tc}")
                    break

    result_list = [{
        'code': box_code,
        'type_code': type_code,
        'confidence': prefix_conf,
        'original_text': prefix_text,
        'bbox': prefix_bbox,
        'y': prefix_y,
        'check_valid': is_valid_container_code(box_code),
        'right_x': prefix_right
    }]

    return result_list