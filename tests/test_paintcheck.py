from PIL import Image, ImageDraw

from argus.core.paintcheck import ink_stats, judge_paint

W, H = 1280, 800


def screen(draw_fn=None):
    img = Image.new("RGB", (W, H), (32, 32, 32))                   # dark window chrome
    d = ImageDraw.Draw(img)
    d.rectangle([120, 170, 1200, 740], fill=(255, 255, 255))        # the white canvas
    d.rectangle([0, 0, W, 90], fill=(45, 45, 45))                    # toolbar with some light text
    d.text((20, 30), "File Edit View", fill=(240, 240, 240))
    if draw_fn:
        draw_fn(d)
    return img


def test_canvas_is_found_and_blank_canvas_fails():
    st = ink_stats(screen())
    x0, y0, x1, y1 = st["canvas"]
    assert abs(x0 - 120) <= 8 and abs(y0 - 170) <= 8 and abs(x1 - 1200) <= 8 and abs(y1 - 740) <= 8
    assert st["ink_px"] == 0
    assert judge_paint(st, "line")[0] is False and "almost no ink" in judge_paint(st, "line")[1]


def test_a_long_line_passes_as_a_line():
    st = ink_stats(screen(lambda d: d.line([300, 300, 760, 450], fill=(0, 0, 0), width=3)))
    ok, why = judge_paint(st, "line")
    assert ok and st["ink_w"] >= 440 and st["ink_h"] >= 140, why


def test_a_tiny_scribble_is_not_a_line_or_a_zigzag():
    # what the live zigzag looked like: three short strokes from the same spot
    def scribble(d):
        d.line([700, 400, 700, 437], fill=(0, 0, 0), width=3)
        d.line([700, 400, 820, 437], fill=(0, 0, 0), width=3)
        d.line([710, 405, 830, 516], fill=(0, 0, 0), width=3)

    st = ink_stats(screen(scribble))
    assert judge_paint(st, "zigzag")[0] is False
    assert judge_paint(st, "line")[0] is False


def test_a_straight_diagonal_is_a_line_but_not_a_zigzag():
    st = ink_stats(screen(lambda d: d.line([300, 300, 760, 450], fill=(0, 0, 0), width=3)))
    assert st["turns"] == 0
    assert judge_paint(st, "line")[0] is True and judge_paint(st, "zigzag")[0] is False


def test_a_real_zigzag_passes_as_zigzag():
    pts = [(300, 300), (420, 420), (540, 300), (660, 420)]
    st = ink_stats(screen(lambda d: d.line(pts, fill=(0, 0, 0), width=3)))
    assert st["turns"] == 2
    assert judge_paint(st, "zigzag")[0] is True
    assert judge_paint(st, "line")[0] is True


def test_no_white_canvas_means_paint_is_not_open():
    st = ink_stats(Image.new("RGB", (W, H), (30, 30, 30)))
    assert st["canvas"] is None and judge_paint(st, "line") == (False, "no white canvas found (Paint not open?)")


def test_reports_how_much_of_the_screen_is_white_even_when_no_canvas_is_found():
    dark = ink_stats(Image.new("RGB", (W, H), (30, 30, 30)))
    assert dark["canvas"] is None and dark["bright_pct"] == 0.0
    paint = ink_stats(screen())
    assert paint["canvas"] is not None and 40 < paint["bright_pct"] < 70      # 1080x570 canvas on 1280x800
    tiny = ink_stats(screen_with_small_canvas())
    assert tiny["canvas"] is None and 0 < tiny["bright_pct"] < 4


def screen_with_small_canvas():
    img = Image.new("RGB", (W, H), (32, 32, 32))
    ImageDraw.Draw(img).rectangle([100, 100, 260, 190], fill=(255, 255, 255))
    return img
