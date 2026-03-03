def cal_patch_index(patch_size, image_size):
    # 依据裁剪块大小和步长确定每一小块的起始位置，返回位置集合
    stride = patch_size // 2
    end_h = (image_size[0] - stride) // stride * stride
    end_w = (image_size[1] - stride) // stride * stride
    h_list = [i for i in range(0, end_h, stride)]
    w_list = [i for i in range(0, end_w, stride)]
    if (image_size[0] - stride) % stride != 0:
        h_list.append(image_size[0] - patch_size)
    if (image_size[1] - stride) % stride != 0:
        w_list.append(image_size[1] - patch_size)
    return h_list, w_list

def cal_patch_index_half(patch_size, image_size):
    # 依据裁剪块大小和步长确定每一小块的起始位置，返回位置集合
    stride = patch_size // 4
    end_h = (image_size[0] - stride) // stride * stride
    end_w = (image_size[1] - stride) // stride * stride
    h_list = [i for i in range(0, end_h, stride)]
    w_list = [i for i in range(0, end_w, stride)]
    if (image_size[0] - stride) % stride != 0:
        h_list.append(image_size[0] - patch_size)
    if (image_size[1] - stride) % stride != 0:
        w_list.append(image_size[1] - patch_size)
    return h_list, w_list

def test_fill_index(i,j,h_start,w_start,h_list,w_list,patch_size):
    # 预测时每个patch的预测结果只保留中间一半
    h_end = h_start + patch_size
    w_end = w_start + patch_size
    patch_h_start = 0
    patch_h_end = patch_size
    patch_w_start = 0
    patch_w_end = patch_size

    if i != 0:
        h_start = h_start + patch_size // 4
        patch_h_start = patch_size // 4

    if i != len(h_list) - 1:
        h_end = h_end - patch_size // 4
        patch_h_end = patch_h_end - patch_size // 4

    if j != 0:
        w_start = w_start + patch_size // 4
        patch_w_start = patch_size // 4

    if j != len(w_list) - 1:
        w_end = w_end - patch_size // 4
        patch_w_end = patch_w_end - patch_size // 4

    fill_index=[h_start,h_end, w_start,w_end]
    patch_index=[patch_h_start, patch_h_end, patch_w_start, patch_w_end]

    return fill_index,patch_index
