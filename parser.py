import numpy as np


 # NOTE: this is needed, because the bg of the left panel contains noise
def diff_rbg(rgb, shifted):
    diff = rgb - shifted
    # NOTE: the second cast is for -128
    diff = np.abs(diff.astype(np.int8, copy=False)).astype(np.uint8, copy=False)
    diff = np.sum(diff, axis=2, dtype=np.uint16)
    diff = diff <= 6
    return diff


def vlines(rgb):
    shift_right = np.roll(rgb, 1, axis=1)
    shift_right[:,0,:] = rgb[:,0,:]
    return diff_rbg(rgb, shift_right)


# run-length encoding
def rl_encode(diff, min_len):
    col_list = []
    h, w = diff.shape
    for x in range(w):
        col = diff[:,x]
        shift_down = np.roll(col, 1)
        shift_down[0] = 1

        turning, = np.where(col != shift_down)
        if len(turning) % 2 == 1:
            turning = np.append(turning, h)

        begin = turning[::2]
        end = turning[1::2]
        valid_index = (end - begin) >= min_len

        col_list.append((begin[valid_index], end[valid_index]))

    return col_list


def rect_lr(vrle, vdiff, x_begin, x_end, rect_w, threshold_min, threshold_max):
    counters = np.zeros(len(vrle), dtype=np.int32)
    for x in range(x_begin, x_end):
        begin, end = vrle[x]
        if len(begin) == 0:
            continue

        length = end - begin
        mid = begin + length // 2
        is_ok = length >= threshold_min
        is_ok &= length <= threshold_max
        if x + rect_w < x_end:
            is_ok &= vdiff[mid, x + rect_w] == 0
            is_ok &= vdiff[end - 1, x + rect_w] == 0
        counters[x] += is_ok.sum()

    counters[counters <= 1] = 0
    idx = np.argsort(counters)[::-1]
    idx = idx[counters[idx] > 0]
    return np.asarray(idx)


def rect_tb_even(hrle, rect_x, rect_w, rect_h, rect_dist, threshold_min, threshold_max):
    h = len(hrle)
    y_ok = np.zeros(shape=(h,), dtype=bool)
    for y in range(h):
        begin, end = hrle[y]
        if len(begin) == 0:
            continue

        length = end - begin
        is_ok = (threshold_min <= length) & (length <= threshold_max)
        is_ok &= begin > rect_x - 10
        is_ok &= end < rect_x + rect_w + 10
        y_ok[y] = is_ok.any()

    ys, = np.where(y_ok)
    unique_off, cnt_off = np.unique(ys % rect_dist, return_counts=True)
    idx_top1, idx_top2 = np.argpartition(cnt_off, -2)[-2:]
    off_top1, off_top2 = unique_off[idx_top1], unique_off[idx_top2]
    off_diff = abs(off_top1 - off_top2)
    assert off_diff == rect_h or off_diff == rect_dist - rect_h
    y_top = ys[ys % rect_dist == off_top1]
    y_bot = ys[ys % rect_dist == off_top2]
    assert len(y_top) == len(y_bot)     # FIXME: align y_top and y_bot
    if y_top[0] > y_bot[0]:
        y_top, y_bot = y_bot, y_top
    if y_bot[0] - y_top[0] == rect_dist - rect_h:   # gap
        y_top, y_bot = y_bot[:-1], y_top[1:]
    return y_top


def rect_tb(hrle, rect_x, rect_w, rect_h, threshold_min, threshold_max):
    h = len(hrle)
    y_ok = np.zeros(shape=(h,), dtype=bool)
    for y in range(h):
        begin, end = hrle[y]
        if len(begin) == 0:
            continue

        length = end - begin
        is_ok = (threshold_min <= length) & (length <= threshold_max)
        is_ok &= begin > rect_x - 10
        is_ok &= end < rect_x + rect_w + 10
        y_ok[y] = is_ok.any()

    y_shifted = np.roll(y_ok, -rect_h)
    y_shifted[-rect_h:] = 0
    y_ok &= y_shifted

    ys, = np.where(y_ok)
    # check non-overlapping
    assert (ys[1:] - ys[:-1] > rect_h).all()
    return ys


def rect_tb_center(hrle, center_x, center_err, rect_h, threshold_min, threshold_max):
    h = len(hrle)
    y_ok = np.zeros(shape=(h,), dtype=bool)
    for y in range(h):
        begin, end = hrle[y]
        if len(begin) == 0:
            continue

        length = end - begin
        center = (begin + end) // 2
        is_ok = (threshold_min <= length) & (length <= threshold_max)
        is_ok &= center > center_x - center_err
        is_ok &= center < center_x + center_err
        y_ok[y] = is_ok.any()

    y_shifted = np.roll(y_ok, -rect_h)
    y_shifted[-rect_h:] = 0
    y_ok &= y_shifted

    ys, = np.where(y_ok)
    # check non-overlapping
    assert (ys[1:] - ys[:-1] > rect_h).all()
    return ys


def find_line(rle, i_begin, i_end, threshold_min):
    for x in range(i_begin, i_end):
        begin, end = rle[x]
        if len(begin) == 0:
            continue
        for b, e in zip(begin, end):
            if e - b >= threshold_min:
                return x
    return None


def find_space_y(edges, y_begin, y_end, x_begin, x_end):
    for y in range(y_begin, y_end):
        if (edges[y, x_begin:x_end] == 1).all():
            return y
    return None


def find_non_space_x(edges, x_begin, x_end, y_begin, y_end):
    for x in range(x_begin, x_end):
        if (edges[y_begin:y_end, x] == 0).any():
            return x
    return None


def find_space_x(edges, x_begin, x_end, y_begin, y_end):
    for x in range(x_begin, x_end):
        if (edges[y_begin:y_end, x] == 1).all():
            return x
    return None


def crop2u32(rgb, x, y, w, h):
    r = rgb[y:y+h, x:x+w, 0].astype(np.uint32)
    g = rgb[y:y+h, x:x+w, 1].astype(np.uint32)
    b = rgb[y:y+h, x:x+w, 2].astype(np.uint32)
    return r | (g << 8) | (b << 16)


class Parser:
    TYPE_LEFT_CHAT = 1
    TYPE_RIGHT_CHAT = 2
    TYPE_TS = 3
    TYPE_END = 4

    left_panel_icon_w = 40
    left_panel_icon_h = 40
    left_panel_icon_dist = 64
    left_panel_icon_threshold_min = 25
    left_panel_icon_threshold_max = 42

    chat_icon_w = 34
    chat_icon_h = 34
    chat_icon_threshold_min = 30
    chat_icon_threshold_max = 40

    group_chat_msg_offset_y = 16

    f14 = None
    f12 = None

    def __init__(self) -> None:
        if self.f14 is None:
            # XXX: hack
            import sys
            sys.path.insert(0, '../img2txt')
            import img2txt
            self.f14 = img2txt.Font()
            self.f14.load_bin_file('../img2txt/yahei.14.bin')
            self.f12 = img2txt.Font()
            self.f12.load_bin_file('../img2txt/yahei.12.bin')

        self.r_left_panel_icon_x = None
        self.r_left_panel_icon_ys = None
        self.r_left_panel_boundary = None
        self.r_left_panel_data = None
        self.r_chat_icon_left_x = None
        self.r_chat_icon_right_x = None
        self.r_editor_boundary = None
        self.r_content_ys = None
        self.r_content_ys_end = None
        self.r_content_left = None
        self.r_content_right = None
        self.r_content_types = None
        self.r_content_data = None
        self.r_content_is_group = None

    def run(self, rgb, training: bool):
        # rgb: uint8 array of dimension (h, w, 3)

        vdiff = vlines(rgb)
        hdiff = vlines(rgb.transpose(1, 0, 2)).transpose()

        h, w = vdiff.shape
        vrle = rl_encode(vdiff, 14)
        hrle = rl_encode(hdiff.transpose(), 14)

        # item left or right
        if training:
            xs = rect_lr(
                vrle, vdiff, 0, w,
                self.left_panel_icon_w, self.left_panel_icon_threshold_min, self.left_panel_icon_threshold_max,
            )
            assert len(xs) >= 1
            self.r_left_panel_icon_x = xs[0]
            print('r_left_panel_icon_x', self.r_left_panel_icon_x)

        # item top or bottom at r_left_panel_icon_x
        self.r_left_panel_icon_ys = rect_tb_even(
            hrle, self.r_left_panel_icon_x, self.left_panel_icon_w, self.left_panel_icon_h, self.left_panel_icon_dist,
            self.left_panel_icon_threshold_min, self.left_panel_icon_threshold_max,
        )
        print('r_left_panel_icon_ys', self.r_left_panel_icon_ys)

        # left panel boundary
        if training:
            self.r_left_panel_boundary = find_line(
                vrle, self.r_left_panel_icon_x + 200, self.r_left_panel_icon_x + 200 + 100,
                self.r_left_panel_icon_ys[-1] - self.r_left_panel_icon_ys[0],
            )
            assert self.r_left_panel_boundary is not None
            print('r_left_panel_boundary', self.r_left_panel_boundary)

        # left and right chat icons
        if training:
            xs = rect_lr(
                vrle, vdiff, self.r_left_panel_boundary + 10, w,
                self.chat_icon_w, self.chat_icon_threshold_min, self.chat_icon_threshold_max,
            )
            xs = xs[:2]
            assert len(xs) == 2     # training data must contain the right chat icon
            self.r_chat_icon_left_x, self.r_chat_icon_right_x = xs
            if self.r_chat_icon_left_x > self.r_chat_icon_right_x:
                self.r_chat_icon_left_x, self.r_chat_icon_right_x = self.r_chat_icon_right_x, self.r_chat_icon_left_x
            print('r_chat_icon_left_x', self.r_chat_icon_left_x)
            print('r_chat_icon_right_x', self.r_chat_icon_right_x)

        left_ys = rect_tb(
            hrle, self.r_chat_icon_left_x, self.chat_icon_w, self.chat_icon_h,
            self.chat_icon_threshold_min, self.chat_icon_threshold_max,
        )
        right_ys = rect_tb(
            hrle, self.r_chat_icon_right_x, self.chat_icon_w, self.chat_icon_h,
            self.chat_icon_threshold_min, self.chat_icon_threshold_max,
        )
        print('left_ys', left_ys)
        print('right_ys', right_ys)

        # timestamp in the center
        ts_h = 19
        ts_w_min = 40
        ts_w_max = 150

        ts_ys = rect_tb_center(
            hrle, (self.r_chat_icon_left_x + self.chat_icon_w + self.r_chat_icon_right_x) // 2, 20, ts_h,
            ts_w_min, ts_w_max,
        )
        print('ts_ys', ts_ys)

        # editor box boundary
        if training:
            y_bound = max(
                left_ys[-1] if len(left_ys) else 0,
                right_ys[-1] if len(right_ys) else 0,
            )
            y_bound = y_bound or h
            self.r_editor_boundary = find_line(
                hrle, y_bound, h,
                self.r_chat_icon_right_x - self.r_chat_icon_left_x,
            )
            assert self.r_editor_boundary is not None
            print('r_editor_boundary', self.r_editor_boundary)

        # bounding box of each element in chat window
        ys = np.concatenate((
            left_ys,
            right_ys,
            ts_ys,
            [self.r_editor_boundary],
        ))
        types = np.concatenate((
            np.full((len(left_ys),), self.TYPE_LEFT_CHAT, dtype=np.uint8),
            np.full((len(right_ys),), self.TYPE_RIGHT_CHAT, dtype=np.uint8),
            np.full((len(ts_ys),), self.TYPE_TS, dtype=np.uint8),
            np.full((1,), self.TYPE_END, dtype=np.uint8),
        ))
        sorted_idx = np.argsort(ys)
        assert len(sorted_idx) == 0 or types[sorted_idx[-1]] == 4

        vhdiff = vdiff & hdiff
        # import PIL.Image
        # PIL.Image.fromarray(vhdiff).save('dbg.png')

        ys_end = np.zeros_like(ys)
        content_left = np.zeros_like(ys)
        content_right = np.zeros_like(ys)
        self.r_content_is_group = None
        for idx1, idx2 in zip(sorted_idx[:-1], sorted_idx[1:]):
            y1, y2 = ys[idx1], ys[idx2]
            tp = types[idx1]
            if tp == self.TYPE_TS:
                y2 = y1 + ts_h
            elif tp == self.TYPE_LEFT_CHAT or tp == self.TYPE_RIGHT_CHAT:
                x1 = self.r_chat_icon_left_x + self.left_panel_icon_w + 5
                x2 = self.r_chat_icon_right_x - 5
                # FIXME: group chat layout
                if self.r_content_is_group is None:
                    self.r_content_is_group = (
                        vhdiff[y1 + 12, x1:x2] == 1
                    ).all()
                    print('r_content_is_group', self.r_content_is_group)
                y2 = find_space_y(
                    vhdiff, y1 + self.chat_icon_h, y2,
                    x1, x2,
                )
                if y2 is None:
                    break   # the last one is not complete
            else:
                assert 0

            ys_end[idx1] = y2
            content_left[idx1] = find_non_space_x(
                vhdiff, self.r_chat_icon_left_x + self.chat_icon_w + 2, self.r_chat_icon_right_x - 2,
                y1, y2,
            )
            content_right[idx1] = find_space_x(
                vhdiff, content_left[idx1] + 2, self.r_chat_icon_right_x - 2,
                y1, y2,
            )

        if len(sorted_idx) > 1 and ys_end[sorted_idx[-2]] == 0:
            last_idx = sorted_idx[-2]       # the incomplete one
            ys = np.delete(ys, last_idx)
            ys_end = np.delete(ys_end, last_idx)
            content_left = np.delete(content_left, last_idx)
            content_right = np.delete(content_right, last_idx)
            types = np.delete(types, last_idx)

        assert types[-1] == self.TYPE_END
        ys = ys[:-1]
        ys_end = ys_end[:-1]
        content_left = content_left[:-1]
        content_right = content_right[:-1]
        types = types[:-1]

        self.r_content_ys = ys
        self.r_content_ys_end = ys_end
        self.r_content_left = content_left
        self.r_content_right = content_right
        self.r_content_types = types
        print('r_chat_ys_end', ys_end)

        self.r_content_data = []
        for y, yend, x1, x2, tp in zip(ys, ys_end, content_left, content_right, types):
            # TODO: group chat name
            pass

            if tp == self.TYPE_LEFT_CHAT or tp == self.TYPE_RIGHT_CHAT:
                if self.r_content_is_group:
                    y += self.group_chat_msg_offset_y
                yend -= 2   # boundary

            crop = crop2u32(rgb, x1, y, x2 - x1, yend - y).ravel()
            crop_ptr, _ = crop.__array_interface__['data']
            font = self.f12 if tp == self.TYPE_TS else self.f14
            char_result = font.run(crop_ptr, x2 - x1, yend - y)
            line_result = font.make_lines(crop_ptr, x2 - x1, yend - y, char_result)

            contents = []
            self.r_content_data.append(contents)
            for tx, ty, tw, th, txt in line_result.tolist():
                # # XXX: filter out noise
                # if tp == self.TYPE_TS and ty != 1:
                #     continue
                contents.append((x1 + tx, y + ty, tw, th, txt))

            # TODO: other content types, image, rp...

        self.r_left_panel_data = []
        crop_h = self.left_panel_icon_h + 1
        for y in self.r_left_panel_icon_ys:
            x1 = self.r_left_panel_icon_x + self.left_panel_icon_w
            x2 = self.r_left_panel_boundary
            crop = crop2u32(rgb, x1, y, x2 - x1, crop_h)
            crop[crop > 0xb9b9b9] = 0xffffff     # make bg constant

            # import PIL.Image
            # t = np.empty((crop.shape[0], crop.shape[1], 3), dtype=np.uint8)
            # t[:, :, 0] = crop.astype(np.uint8)
            # t[:, :, 1] = (crop >> 8).astype(np.uint8)
            # t[:, :, 2] = (crop >> 16).astype(np.uint8)
            # PIL.Image.fromarray(t).save('dbg.png') if y == 200 else None

            crop = crop.ravel()
            crop_ptr, _ = crop.__array_interface__['data']

            char_result = self.f14.run(crop_ptr, x2 - x1, crop_h)
            line_result = self.f14.make_lines(crop_ptr, x2 - x1, crop_h, char_result)
            # print(y, line_result.tolist())
            title = ''
            for cx, cy, cw, _, ctxt in line_result.tolist():
                if cx == 10 and cy == 0:
                    title = ctxt
                    break
            if title[-1] == '‥':
                title = title[:-1] + '…'

            char_result = self.f12.run(crop_ptr, x2 - x1, crop_h)
            line_result = self.f12.make_lines(crop_ptr, x2 - x1, crop_h, char_result)
            # print('xxx', y, line_result.tolist())
            ts = ''
            detail = ''
            for cx, cy, cw, _, ctxt in line_result.tolist():
                if cx + cw == 186 and cy == 0:
                    ts = ctxt
                elif cx == 10 and cy == 24:
                    # FIXME: unrecognized char in the middle
                    detail = ctxt
                    break
            if detail.endswith('‥.'):
                detail = detail[:-2] + '…'

            self.r_left_panel_data.append((title, ts, detail))


def add_rect(rgb, rect_x, rect_y, rect_w, rect_h, color=np.asarray([0xff, 0x40, 0x40])):
    x_slice = slice(rect_x, rect_x + rect_w)
    rgb[rect_y - 1, x_slice] = color
    rgb[rect_y + rect_h, x_slice] = color
    x_pair = (rect_x - 1, rect_x + rect_w)
    rgb[rect_y : rect_y + rect_h, x_pair] = color


def main():
    import PIL.Image

    im = PIL.Image.open('wx5.png')
    rgb = np.asarray(im)

    # vdiff = vlines(rgb)
    # hdiff = vlines(rgb.transpose(1, 0, 2)).transpose()

    # # debug
    # PIL.Image.fromarray(vdiff).save('v_wx5.png')
    # PIL.Image.fromarray(hdiff).save('h_wx5.png')

    p = Parser()
    p.run(rgb, training=True)

    import sys
    input_img = sys.argv[1]
    assert '/' not in input_img

    im = PIL.Image.open(input_img)
    rgb = np.asarray(im)
    p.run(rgb, training=False)

    # demo
    for y in p.r_left_panel_icon_ys:
        add_rect(rgb, p.r_left_panel_icon_x, y, p.left_panel_icon_w, p.left_panel_icon_h)
    t2c = {
        p.TYPE_LEFT_CHAT: [0, 0, 0xff],
        p.TYPE_RIGHT_CHAT: [0xff, 0, 0xff],
        p.TYPE_TS: [0, 0xff, 0],
    }
    for y, yend, x1, x2, tp in zip(p.r_content_ys, p.r_content_ys_end, p.r_content_left, p.r_content_right, p.r_content_types):
        if tp == p.TYPE_LEFT_CHAT:
            add_rect(rgb, p.r_chat_icon_left_x, y, p.chat_icon_w, p.chat_icon_h, color=t2c[tp])
        if tp == p.TYPE_RIGHT_CHAT:
            add_rect(rgb, p.r_chat_icon_right_x, y, p.chat_icon_w, p.chat_icon_h, color=t2c[tp])
        if p.r_content_is_group and (tp == p.TYPE_LEFT_CHAT or tp == p.TYPE_RIGHT_CHAT):
            y += p.group_chat_msg_offset_y
        add_rect(rgb, x1, y, x2 - x1, yend - y, color=t2c[tp])

    for y, (title, ts, detail) in zip(p.r_left_panel_icon_ys, p.r_left_panel_data):
        print(y, (title, ts, detail))

    for idx in np.argsort(p.r_content_ys):
        for cx, cy, cw, ch, ctxt in p.r_content_data[idx]:
            print(p.r_content_types[idx], (cx, cy, cw, ch), ctxt)
            # if len(ctxt) == 1:
            #     print('yyy', p.r_content_types[idx], (cx, cy, cw, ch), ord(ctxt))
            #     add_rect(rgb, cx, cy, cw, ch)

    PIL.Image.fromarray(rgb).save('mark_' + input_img)


if __name__ == '__main__':
    main()
